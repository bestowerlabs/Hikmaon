from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from app.storage import InMemoryStore


class MonitoringService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def index_public_media(self, media_url: str, media_bytes: bytes) -> dict:
        content_hash = hashlib.sha256(media_bytes).hexdigest()
        fingerprint = hashlib.sha256(media_bytes[:4096]).hexdigest()
        record = {
            "url": media_url,
            "fingerprint": fingerprint,
            "first_seen_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.store.crawler_index.setdefault(content_hash, []).append(record)
        return {"content_hash": content_hash, **record}
