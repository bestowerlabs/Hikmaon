from __future__ import annotations

import base64
import hashlib
import uuid
from datetime import datetime, timezone

import numpy as np

from app.models import RegistrationCreate, RegistrationRecord
from app.storage import InMemoryStore


class RegistrationService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def register_media(self, payload: RegistrationCreate) -> RegistrationRecord:
        raw_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        fingerprint_commitment = self._fingerprint(payload.media_type, raw_bytes)
        embedding = self._embedding(raw_bytes)

        txid = self._submit_hikmalayer_tx(
            content_hash=content_hash,
            fingerprint_commitment=fingerprint_commitment,
            owner_public_key=payload.owner_public_key,
            metadata_pointer=f"registrations/{payload.owner_id}/{payload.filename}",
        )

        record = RegistrationRecord(
            media_id=str(uuid.uuid4()),
            owner_id=payload.owner_id,
            owner_public_key=payload.owner_public_key,
            media_type=payload.media_type,
            filename=payload.filename,
            content_hash=content_hash,
            fingerprint_commitment=fingerprint_commitment,
            embedding=embedding,
            metadata=payload.metadata,
            blockchain_txid=txid,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.registrations[record.media_id] = record
        return record

    def _fingerprint(self, media_type: str, raw_bytes: bytes) -> str:
        # Simplified perceptual fingerprint placeholder with media-type salting.
        fingerprint_source = media_type.encode("utf-8") + raw_bytes[:4096]
        return hashlib.sha256(fingerprint_source).hexdigest()

    def _embedding(self, raw_bytes: bytes, dim: int = 512) -> list[float]:
        seed = int(hashlib.sha256(raw_bytes).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(0, 1, dim)
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 0 else vector
        return normalized.astype(float).tolist()

    def _submit_hikmalayer_tx(
        self,
        content_hash: str,
        fingerprint_commitment: str,
        owner_public_key: str,
        metadata_pointer: str,
    ) -> str:
        txid = f"hkml_{uuid.uuid4().hex[:18]}"
        self.store.blockchain_records[txid] = {
            "type": "MEDIA_REGISTRATION",
            "payload": {
                "content_hash": content_hash,
                "fingerprint_commitment": fingerprint_commitment,
                "owner_pubkey": owner_public_key,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "metadata_pointer": metadata_pointer,
            },
        }
        return txid
