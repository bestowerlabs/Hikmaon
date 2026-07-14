"""Media sync: pull recent uploads from connected platforms and auto-register.

``POST /api/connectors/{id}/sync`` walks the provider's media API with the
stored OAuth token, downloads each new item, and feeds it through the
standard ingest pipeline (hash -> fingerprint -> Hikmalayer anchor ->
certificate). Items already registered (same content hash) are skipped.

Adapters implemented against the real provider APIs: Instagram/Facebook
(Graph), Google Drive, Dropbox, OneDrive (Microsoft Graph), X. Remaining
providers return a precise "not implemented" pointer rather than failing
silently.
"""
from __future__ import annotations

import hashlib

import httpx

from app import net_guard
from app.integrations.oauth import OAuthManager
from app.models import ConnectorAccount, ConnectorIngestEvent
from app.storage import InMemoryStore

MAX_ITEMS_PER_SYNC = 25
IMAGE_MIME_PREFIXES = ("image/",)
VIDEO_MIME_PREFIXES = ("video/",)


class MediaSyncService:
    def __init__(self, store: InMemoryStore, oauth: OAuthManager, pipeline) -> None:
        self.store = store
        self.oauth = oauth
        self.pipeline = pipeline

    def sync(self, account: ConnectorAccount) -> dict:
        adapter = getattr(self, f"_list_{account.provider}", None)
        if adapter is None:
            return {
                "provider": account.provider,
                "synced": 0,
                "detail": f"media sync adapter for {account.provider} not implemented yet "
                f"(add _list_{account.provider} to app/integrations/sync.py)",
            }

        token = self.oauth.access_token(account)
        items = adapter(token)

        known_hashes = {r.content_hash for r in self.store.registrations.values()}
        registered, skipped = [], 0
        for item in items[:MAX_ITEMS_PER_SYNC]:
            media_bytes = self._download(item["url"], token if item.get("authorized_download") else None)
            if media_bytes is None:
                continue
            if hashlib.sha256(media_bytes).hexdigest() in known_hashes:
                skipped += 1
                continue
            import base64

            result = self.pipeline.ingest_from_connector(
                ConnectorIngestEvent(
                    connector_id=account.connector_id,
                    media_type=item.get("media_type", "image"),
                    filename=item.get("filename", "synced-media"),
                    content_b64=base64.b64encode(media_bytes).decode(),
                    source_url=item.get("source_url", item["url"]),
                )
            )
            registered.append(result["media_id"])

        return {
            "provider": account.provider,
            "found": len(items),
            "synced": len(registered),
            "already_registered": skipped,
            "media_ids": registered,
        }

    # ------------------------------------------------------------ adapters
    def _list_instagram(self, token: str) -> list[dict]:
        response = httpx.get(
            "https://graph.facebook.com/v19.0/me/media",
            params={"fields": "id,media_type,media_url,timestamp,permalink", "access_token": token},
            timeout=20.0,
        )
        response.raise_for_status()
        items = []
        for entry in response.json().get("data", []):
            if not entry.get("media_url"):
                continue
            items.append(
                {
                    "url": entry["media_url"],
                    "source_url": entry.get("permalink", entry["media_url"]),
                    "filename": f"instagram_{entry['id']}",
                    "media_type": "video" if entry.get("media_type") == "VIDEO" else "image",
                }
            )
        return items

    def _list_facebook(self, token: str) -> list[dict]:
        response = httpx.get(
            "https://graph.facebook.com/v19.0/me/photos/uploaded",
            params={"fields": "id,images,created_time,link", "access_token": token},
            timeout=20.0,
        )
        response.raise_for_status()
        items = []
        for entry in response.json().get("data", []):
            images = entry.get("images") or []
            if not images:
                continue
            items.append(
                {
                    "url": images[0]["source"],  # largest rendition first
                    "source_url": entry.get("link", images[0]["source"]),
                    "filename": f"facebook_{entry['id']}",
                    "media_type": "image",
                }
            )
        return items

    def _list_google_drive(self, token: str) -> list[dict]:
        response = httpx.get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": "mimeType contains 'image/' or mimeType contains 'video/'",
                "orderBy": "createdTime desc",
                "pageSize": MAX_ITEMS_PER_SYNC,
                "fields": "files(id,name,mimeType,webViewLink)",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        response.raise_for_status()
        items = []
        for entry in response.json().get("files", []):
            items.append(
                {
                    "url": f"https://www.googleapis.com/drive/v3/files/{entry['id']}?alt=media",
                    "source_url": entry.get("webViewLink", ""),
                    "filename": entry.get("name", entry["id"]),
                    "media_type": "video" if entry.get("mimeType", "").startswith("video/") else "image",
                    "authorized_download": True,
                }
            )
        return items

    def _list_dropbox(self, token: str) -> list[dict]:
        response = httpx.post(
            "https://api.dropboxapi.com/2/files/list_folder",
            json={"path": "", "recursive": True, "limit": MAX_ITEMS_PER_SYNC * 4},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        response.raise_for_status()
        items = []
        for entry in response.json().get("entries", []):
            name = entry.get("name", "").lower()
            if entry.get(".tag") != "file":
                continue
            if not name.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov")):
                continue
            link = httpx.post(
                "https://api.dropboxapi.com/2/files/get_temporary_link",
                json={"path": entry["path_lower"]},
                headers={"Authorization": f"Bearer {token}"},
                timeout=20.0,
            )
            if link.status_code != 200:
                continue
            items.append(
                {
                    "url": link.json()["link"],
                    "source_url": f"dropbox://{entry['path_lower']}",
                    "filename": entry["name"],
                    "media_type": "video" if name.endswith((".mp4", ".mov")) else "image",
                }
            )
        return items

    def _list_onedrive(self, token: str) -> list[dict]:
        response = httpx.get(
            "https://graph.microsoft.com/v1.0/me/drive/recent",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        response.raise_for_status()
        items = []
        for entry in response.json().get("value", []):
            mime = (entry.get("file") or {}).get("mimeType", "")
            if not mime.startswith(IMAGE_MIME_PREFIXES + VIDEO_MIME_PREFIXES):
                continue
            items.append(
                {
                    "url": f"https://graph.microsoft.com/v1.0/me/drive/items/{entry['id']}/content",
                    "source_url": entry.get("webUrl", ""),
                    "filename": entry.get("name", entry["id"]),
                    "media_type": "video" if mime.startswith("video/") else "image",
                    "authorized_download": True,
                }
            )
        return items

    def _list_x(self, token: str) -> list[dict]:
        me = httpx.get(
            "https://api.x.com/2/users/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        me.raise_for_status()
        user_id = me.json()["data"]["id"]
        response = httpx.get(
            f"https://api.x.com/2/users/{user_id}/tweets",
            params={"expansions": "attachments.media_keys", "media.fields": "url,type,variants"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        response.raise_for_status()
        media = (response.json().get("includes") or {}).get("media", [])
        items = []
        for entry in media:
            url = entry.get("url")
            if not url and entry.get("variants"):
                mp4 = [v for v in entry["variants"] if v.get("content_type") == "video/mp4"]
                url = mp4[0]["url"] if mp4 else None
            if not url:
                continue
            items.append(
                {
                    "url": url,
                    "source_url": url,
                    "filename": f"x_{entry.get('media_key', 'media')}",
                    "media_type": "video" if entry.get("type") == "video" else "image",
                }
            )
        return items

    # ----------------------------------------------------------- download
    def _download(self, url: str, token: str | None) -> bytes | None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = net_guard.safe_get(url, headers=headers)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, net_guard.UnsafeURLError):
            return None
