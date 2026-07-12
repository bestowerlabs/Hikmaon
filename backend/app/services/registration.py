from __future__ import annotations

import base64
import hashlib
import os
import uuid
from datetime import datetime, timezone

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from app.hikmalayer import HikmalayerClient
from app.models import RegistrationCreate, RegistrationRecord
from app.perceptual import fingerprint_media
from app.services.certificate import CertificateService
from app.storage import InMemoryStore


class OwnershipProofError(ValueError):
    """Raised when a registration fails (or is missing) its ownership proof."""


class RegistrationService:
    def __init__(
        self,
        store: InMemoryStore,
        chain_client: HikmalayerClient,
        certificate_service: CertificateService,
        auto_signer=None,
    ) -> None:
        self.store = store
        self.chain_client = chain_client
        self.certificate_service = certificate_service
        # auto_signer(owner_id, content_hash) -> signature_b64 | None. Lets the
        # account system sign registrations with the owner's key automatically.
        self.auto_signer = auto_signer
        self.require_ownership_proof = os.environ.get("HIKMAON_REQUIRE_OWNERSHIP_PROOF", "0") == "1"

    def register_media(self, payload: RegistrationCreate) -> RegistrationRecord:
        raw_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        if not payload.ownership_signature_b64 and self.auto_signer is not None:
            payload.ownership_signature_b64 = self.auto_signer(payload.owner_id, content_hash)

        ownership_proven = self._verify_ownership_proof(payload, content_hash)
        if self.require_ownership_proof and not ownership_proven:
            raise OwnershipProofError(
                "Ownership proof required: sign the SHA-256 content hash with the owner key "
                "and pass ownership_signature_b64"
            )

        fingerprint = fingerprint_media(raw_bytes)

        tx = self.chain_client.submit_media_registration(
            content_hash=content_hash,
            fingerprint_commitment=fingerprint.commitment,
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
            fingerprint_commitment=fingerprint.commitment,
            media_kind=fingerprint.media_kind,
            phash_hex=fingerprint.phash_hex,
            dhash_hex=fingerprint.dhash_hex,
            chunk_fingerprints=fingerprint.chunks,
            embedding=fingerprint.embedding,
            frame_phashes=fingerprint.frame_phashes,
            audio_bits=fingerprint.audio_bits,
            metadata=payload.metadata,
            blockchain_txid=tx.txid,
            chain_mode=tx.chain_mode,
            ownership_proven=ownership_proven,
            created_at=datetime.now(tz=timezone.utc),
        )

        certificate = self.certificate_service.issue(record)
        record.certificate_id = certificate.certificate_id

        self.store.registrations[record.media_id] = record
        self.store.persist()
        return record

    def _verify_ownership_proof(self, payload: RegistrationCreate, content_hash: str) -> bool:
        """Verify an Ed25519 signature over the content hash, if supplied.

        `owner_public_key` must be the base64-encoded 32-byte Ed25519 public
        key for the proof to validate.
        """
        if not payload.ownership_signature_b64:
            return False
        try:
            public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(payload.owner_public_key))
            signature = base64.b64decode(payload.ownership_signature_b64)
            public_key.verify(signature, content_hash.encode("utf-8"))
            return True
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise OwnershipProofError(f"Ownership signature invalid: {exc}") from exc
