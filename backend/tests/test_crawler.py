from __future__ import annotations

import asyncio
import base64

import httpx
from fastapi.testclient import TestClient

from app.main import app, crawler_service, store
from app.models import CrawlJobCreate
from app.services import crawler as crawler_module

client = TestClient(app)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _fake_site(media_bytes: bytes, robots: str = "User-agent: *\nAllow: /\n"):
    """An httpx MockTransport simulating a small public website."""
    pages = {
        "https://fakesite.example/robots.txt": (robots, "text/plain"),
        "https://fakesite.example/": (
            '<html><body><a href="/gallery">gallery</a>'
            '<meta property="og:image" content="/promo.jpg"/></body></html>',
            "text/html",
        ),
        "https://fakesite.example/gallery": (
            '<html><body><img src="/stolen.jpg"/><a href="/">home</a></body></html>',
            "text/html",
        ),
    }
    binaries = {
        "https://fakesite.example/stolen.jpg": media_bytes,
        "https://fakesite.example/promo.jpg": media_bytes,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in pages:
            text, content_type = pages[url]
            return httpx.Response(200, text=text, headers={"content-type": content_type})
        if url in binaries:
            return httpx.Response(200, content=binaries[url], headers={"content-type": "image/jpeg"})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _patch_network(monkeypatch, transport: httpx.MockTransport) -> None:
    original_client = httpx.AsyncClient

    def patched_client(**kwargs):
        kwargs["transport"] = transport
        return original_client(**kwargs)

    monkeypatch.setattr(crawler_module.httpx, "AsyncClient", patched_client)
    # The fake domain has no DNS entry; scope/public checks are patched in.
    monkeypatch.setattr(crawler_module, "_host_is_public", lambda hostname: True)
    monkeypatch.setattr(crawler_module, "POLITENESS_SECONDS", 0.0)
    crawler_service._robots.clear()  # isolate robots.txt cache between tests


def test_crawler_finds_stolen_media_and_opens_incident(monkeypatch, make_user, make_photo, to_bytes):
    headers, user, _ = make_user()
    original = to_bytes(make_photo(81))
    registered = client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "mine.png", "content_b64": base64.b64encode(original).decode()},
        headers=headers,
    ).json()

    # The site reposts a re-encoded copy of the registered photo.
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(original)).convert("RGB").resize((200, 200))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=65)
    _patch_network(monkeypatch, _fake_site(buffer.getvalue()))

    job = crawler_service.create_job(
        CrawlJobCreate(seed_urls=["https://fakesite.example/"], max_pages=10, max_depth=2),
        owner_user_id=user["user_id"],
    )
    finished = _run(crawler_service.run_job(job))

    assert finished.status == "completed"
    assert finished.pages_crawled >= 2
    assert finished.media_indexed >= 1
    assert finished.matches_found >= 1
    incident_id = finished.incidents[0]
    incident = store.incidents[incident_id]
    assert incident.matched_media_id == registered["media_id"]
    assert any("fakesite.example" in url for url in incident.matched_urls)


def test_crawler_respects_robots_txt(monkeypatch, make_user, make_photo, to_bytes):
    headers, user, _ = make_user()
    media = to_bytes(make_photo(82))
    _patch_network(
        monkeypatch,
        _fake_site(media, robots="User-agent: *\nDisallow: /\n"),
    )
    job = crawler_service.create_job(
        CrawlJobCreate(seed_urls=["https://fakesite.example/"], max_pages=10),
        owner_user_id=user["user_id"],
    )
    finished = _run(crawler_service.run_job(job))
    assert finished.status == "completed"
    assert finished.pages_crawled == 0  # everything disallowed
    assert finished.media_indexed == 0


def test_crawler_stays_in_scope(monkeypatch, make_user, make_photo, to_bytes):
    headers, user, _ = make_user()
    media = to_bytes(make_photo(83))
    transport = _fake_site(media)

    fetched: list[str] = []
    original_handler = transport.handler

    def spying_handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        return original_handler(request)

    _patch_network(monkeypatch, httpx.MockTransport(spying_handler))
    job = crawler_service.create_job(
        CrawlJobCreate(seed_urls=["https://fakesite.example/"], max_pages=10),
        owner_user_id=user["user_id"],
    )
    _run(crawler_service.run_job(job))
    assert all("fakesite.example" in url for url in fetched)


def test_ssrf_guard_blocks_private_addresses():
    assert crawler_module._host_is_public("localhost") is False
    assert crawler_module._url_allowed("http://127.0.0.1/admin") is False
    assert crawler_module._url_allowed("http://169.254.169.254/latest/meta-data") is False
    assert crawler_module._url_allowed("ftp://example.com/x") is False


def test_crawler_refuses_redirect_to_internal_host(monkeypatch, make_user, make_photo, to_bytes):
    """A crawled public page that 302-redirects to an internal host must not
    be followed (SSRF redirect-bypass regression)."""
    headers, user, _ = make_user()

    internal_hit = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "127.0.0.1":
            internal_hit["count"] += 1
            return httpx.Response(200, content=b"INTERNAL-SECRET")
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", headers={"content-type": "text/plain"})
        # Public page redirects an image fetch to the metadata/internal host.
        return httpx.Response(302, headers={"location": "http://127.0.0.1/latest/meta-data"})

    _patch_network(monkeypatch, httpx.MockTransport(handler))
    # Only the fake public domain is "public"; 127.0.0.1 must be judged internal.
    monkeypatch.setattr(crawler_module, "_host_is_public", lambda host: host == "fakesite.example")

    job = crawler_service.create_job(
        CrawlJobCreate(seed_urls=["https://fakesite.example/"], max_pages=5, max_depth=1),
        owner_user_id=user["user_id"],
    )
    finished = _run(crawler_service.run_job(job))

    assert finished.status == "completed"
    assert internal_hit["count"] == 0  # the crawler never fetched the internal host
    assert finished.media_indexed == 0


def test_crawl_job_api_requires_auth_and_validates(make_user):
    headers, _, _ = make_user()
    bad = client.post(
        "/api/crawler/jobs",
        json={"seed_urls": ["ftp://nope.example"]},
        headers=headers,
    )
    assert bad.status_code == 400
