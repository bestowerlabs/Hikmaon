from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from app.perceptual import fingerprint_media
from app.storage import InMemoryStore


class MonitoringService:
    """Public-media index built on perceptual fingerprints.

    Indexed items are matched by perceptual-hash distance (not exact bytes),
    so re-encoded/edited copies discovered on the public internet still link
    back to registered originals. Production replaces the ingest endpoint
    with a robots.txt-compliant crawler and platform-API ingestion feeding
    this same index.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def index_public_media(self, media_url: str, media_bytes: bytes, source: str = "manual") -> dict:
        fingerprint = fingerprint_media(media_bytes)
        record = {
            "url": media_url,
            "content_hash": hashlib.sha256(media_bytes).hexdigest(),
            "media_kind": fingerprint.media_kind,
            "phash_hex": fingerprint.phash_hex,
            "dhash_hex": fingerprint.dhash_hex,
            "embedding": fingerprint.embedding,
            "chunks": fingerprint.chunks,
            "frame_phashes": fingerprint.frame_phashes,
            "audio_bits": fingerprint.audio_bits,
            "source": source,
            "first_seen_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        self.store.crawler_index.append(record)
        self.store.persist()
        return {
            k: v
            for k, v in record.items()
            if k not in ("chunks", "embedding", "frame_phashes", "audio_bits")
        }
