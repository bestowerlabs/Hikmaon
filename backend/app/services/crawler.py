"""Autonomous public-web crawler for misuse discovery.

Lawful-and-safe by construction:

- **robots.txt compliance** — every URL is checked against the site's
  robots.txt (cached per host) before fetching; disallowed paths are skipped.
- **Politeness** — at most one request per host per ``POLITENESS_SECONDS``,
  page and media size caps, bounded page count and depth per job.
- **SSRF hardening** — only http/https; every hostname is DNS-resolved and
  rejected if it points at private, loopback, link-local, or reserved
  address space, so the crawler can never be steered at internal services.
- **Scoped** — crawls stay inside the job's allowed domains (defaulting to
  the seed domains and their subdomains).

Pipeline per discovered media file: download → perceptual fingerprint →
public-sighting index → compare against registered media → if the match
percentage clears the incident threshold, run the standard detection cycle
(evidence + owner alert + consent workflow).

Jobs run as asyncio background tasks. An optional autonomous schedule
re-crawls configured seeds on an interval:
    HIKMAON_CRAWLER_SEEDS=https://example.com,https://news.site
    HIKMAON_CRAWLER_INTERVAL_MINUTES=60
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import httpx

from app import net_guard
from app.models import CrawlJob, CrawlJobCreate
from app.storage import InMemoryStore

USER_AGENT = "HikmaonBot/1.0 (+https://github.com/bestowerlabs/Hikmaon; authenticity monitoring)"
POLITENESS_SECONDS = 1.0
PAGE_TIMEOUT = 15.0
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_MEDIA_BYTES = 40 * 1024 * 1024
MEDIA_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
    ".mp4", ".webm", ".mov", ".m4v",
    ".mp3", ".m4a", ".ogg", ".wav", ".flac",
)
MEDIA_ATTR_TAGS = {"img": "src", "video": "src", "audio": "src", "source": "src", "embed": "src"}
META_MEDIA_PROPERTIES = {"og:image", "og:image:url", "og:video", "og:video:url", "og:audio", "twitter:image"}


def _host_is_public(hostname: str) -> bool:
    """Resolve a hostname and require every address to be globally routable."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return False
    return True


def _url_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return _host_is_public(parsed.hostname)


def _in_scope(url: str, allowed_domains: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


class _PageParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self.media: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("href"):
            self.links.append(urljoin(self.base_url, attributes["href"]))
        media_attr = MEDIA_ATTR_TAGS.get(tag)
        if media_attr and attributes.get(media_attr):
            self.media.append(urljoin(self.base_url, attributes[media_attr]))
        if tag == "meta" and attributes.get("property") in META_MEDIA_PROPERTIES and attributes.get("content"):
            self.media.append(urljoin(self.base_url, attributes["content"]))
        if tag == "link" and attributes.get("rel") == "image_src" and attributes.get("href"):
            self.media.append(urljoin(self.base_url, attributes["href"]))


class CrawlerService:
    def __init__(self, store: InMemoryStore, monitoring_service, pipeline, match_threshold: float) -> None:
        self.store = store
        self.monitoring = monitoring_service
        self.pipeline = pipeline
        self.match_threshold = match_threshold
        self._robots: dict[str, tuple[float, robotparser.RobotFileParser]] = {}
        self._robots_ttl = 3600.0
        self._last_hit: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

    # --------------------------------------------------------------- jobs
    def create_job(self, payload: CrawlJobCreate, owner_user_id: str) -> CrawlJob:
        seeds = [u for u in payload.seed_urls if _url_allowed(u)]
        if not seeds:
            raise ValueError("No crawlable seed URLs (public http/https only)")
        allowed = [d.lower().lstrip(".") for d in payload.allowed_domains] or sorted(
            {urlparse(u).hostname.lower() for u in seeds}
        )
        job = CrawlJob(
            job_id=f"crawl_{uuid.uuid4().hex[:12]}",
            owner_user_id=owner_user_id,
            seed_urls=seeds,
            allowed_domains=allowed,
            max_pages=payload.max_pages,
            max_depth=payload.max_depth,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.crawl_jobs[job.job_id] = job
        self.store.persist()
        return job

    def start_job(self, job: CrawlJob) -> None:
        task = asyncio.get_running_loop().create_task(self.run_job(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run_job(self, job: CrawlJob) -> CrawlJob:
        job.status = "running"
        self.store.persist()
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                follow_redirects=False,  # redirects are followed manually + re-validated (SSRF guard)
                timeout=PAGE_TIMEOUT,
            ) as client:
                await self._crawl(client, job)
            job.status = "completed"
        except Exception as exc:  # a job must never take the API down
            job.status = "failed"
            job.errors.append(str(exc)[:300])
        job.finished_at = datetime.now(tz=timezone.utc)
        self.store.persist()
        return job

    # -------------------------------------------------------------- crawl
    async def _crawl(self, client: httpx.AsyncClient, job: CrawlJob) -> None:
        queue: list[tuple[str, int]] = [(url, 0) for url in job.seed_urls]
        visited_pages: set[str] = set()
        seen_media: set[str] = set()

        while queue and job.pages_crawled < job.max_pages:
            url, depth = queue.pop(0)
            if url in visited_pages or not _in_scope(url, job.allowed_domains):
                continue
            visited_pages.add(url)
            if not _url_allowed(url) or not await self._robots_allowed(client, url):
                continue

            await self._be_polite(url)
            try:
                response = await self._safe_get(client, url)
            except (httpx.HTTPError, net_guard.UnsafeURLError) as exc:
                job.errors.append(f"{url}: {exc}"[:200])
                continue
            if response.status_code != 200:
                continue

            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            body = response.content[:MAX_PAGE_BYTES]
            job.pages_crawled += 1

            if content_type.startswith(("image/", "video/", "audio/")):
                await self._process_media(client, job, str(response.url), preloaded=body)
            elif content_type == "text/html":
                parser = _PageParser(str(response.url))
                try:
                    parser.feed(body.decode(response.encoding or "utf-8", errors="replace"))
                except Exception:
                    continue
                for media_url in parser.media:
                    if media_url not in seen_media and _url_allowed(media_url):
                        seen_media.add(media_url)
                        await self._process_media(client, job, media_url)
                if depth < job.max_depth:
                    for link in parser.links:
                        link = link.split("#")[0]
                        if link not in visited_pages and _in_scope(link, job.allowed_domains):
                            if link.lower().endswith(MEDIA_EXTENSIONS):
                                if link not in seen_media and _url_allowed(link):
                                    seen_media.add(link)
                                    await self._process_media(client, job, link)
                            else:
                                queue.append((link, depth + 1))
            self.store.persist()

    async def _process_media(
        self, client: httpx.AsyncClient, job: CrawlJob, media_url: str, preloaded: bytes | None = None
    ) -> None:
        media_bytes = preloaded
        if media_bytes is None:
            if not await self._robots_allowed(client, media_url):
                return
            await self._be_polite(media_url)
            try:
                response = await self._safe_get(client, media_url)
            except (httpx.HTTPError, net_guard.UnsafeURLError):
                return
            if response.status_code != 200 or len(response.content) > MAX_MEDIA_BYTES:
                return
            media_bytes = response.content
        if not media_bytes:
            return

        # Fingerprint + index the sighting (skips heavy AI unless it matches).
        import base64

        self.monitoring.index_public_media(media_url, media_bytes, source=f"crawler:{job.job_id}")
        job.media_indexed += 1

        result = self.pipeline.run_detection_cycle(
            media_type="image",
            filename=media_url.rsplit("/", 1)[-1][:120] or "crawled-media",
            content_b64=base64.b64encode(media_bytes).decode(),
        )
        if result.get("event") == "incident_created":
            job.matches_found += 1
            job.incidents.append(result["incident"]["incident_id"])

    # ------------------------------------------------------------ helpers
    async def _robots_allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._robots.get(origin)
        if cached is not None and time.time() - cached[0] < self._robots_ttl:
            robots = cached[1]
        else:
            robots = robotparser.RobotFileParser()
            try:
                response = await self._safe_get(client, f"{origin}/robots.txt")
                if response.status_code == 200:
                    robots.parse(response.text.splitlines())
                else:
                    robots.parse([])  # no robots.txt -> allowed
            except (httpx.HTTPError, net_guard.UnsafeURLError):
                robots.parse([])
            self._robots[origin] = (time.time(), robots)
        return robots.can_fetch(USER_AGENT, url)

    async def _safe_get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        """Fetch with redirects followed manually and every hop re-validated
        against the public-host guard (SSRF protection)."""
        return await net_guard.safe_get_async(client, url, url_validator=_url_allowed)

    async def _be_polite(self, url: str) -> None:
        host = urlparse(url).netloc
        elapsed = time.time() - self._last_hit.get(host, 0.0)
        if elapsed < POLITENESS_SECONDS:
            await asyncio.sleep(POLITENESS_SECONDS - elapsed)
        self._last_hit[host] = time.time()


async def autonomous_schedule(crawler: CrawlerService, seeds: list[str], interval_minutes: float) -> None:
    """Continuously re-crawl configured seeds (started from app startup)."""
    from app.models import CrawlJobCreate

    while True:
        try:
            job = crawler.create_job(CrawlJobCreate(seed_urls=seeds), owner_user_id="system")
            await crawler.run_job(job)
        except Exception:
            pass
        await asyncio.sleep(interval_minutes * 60)
