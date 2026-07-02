"""Certificate of Ownership issuance and verification.

Each registration receives a portable, independently verifiable certificate:
the certificate body (media hash, fingerprint commitment, owner, Hikmalayer
txid, timestamp) is canonically serialized and signed with Hikmaon's Ed25519
issuing key. Anyone holding Hikmaon's public key can verify the certificate
offline; the embedded txid lets them confirm the anchor on Hikmalayer.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from app.models import OwnershipCertificate, RegistrationRecord
from app.storage import InMemoryStore

_SIGNED_FIELDS = (
    "certificate_id",
    "media_id",
    "content_hash",
    "fingerprint_commitment",
    "owner_id",
    "owner_public_key",
    "blockchain_txid",
    "chain_mode",
    "issued_at",
    "issuer",
)


def _canonical_payload(cert_fields: dict) -> bytes:
    body = {name: str(cert_fields[name]) for name in _SIGNED_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


class CertificateService:
    def __init__(self, store: InMemoryStore, key_path: Path | None = None) -> None:
        self.store = store
        self._private_key = self._load_or_create_key(key_path)
        public_raw = self._private_key.public_key().public_bytes_raw()
        self.public_key_b64 = base64.b64encode(public_raw).decode()
        self.signing_key_id = hashlib.sha256(public_raw).hexdigest()[:16]

    def _load_or_create_key(self, key_path: Path | None) -> Ed25519PrivateKey:
        seed_hex = os.environ.get("HIKMAON_SIGNING_KEY")
        if seed_hex:
            return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
        if key_path is not None:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            if key_path.exists():
                return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(key_path.read_text().strip()))
            key = Ed25519PrivateKey.generate()
            key_path.write_text(key.private_bytes_raw().hex())
            os.chmod(key_path, 0o600)
            return key
        return Ed25519PrivateKey.generate()

    def issue(self, registration: RegistrationRecord) -> OwnershipCertificate:
        fields = {
            "certificate_id": f"cert_{uuid.uuid4().hex[:16]}",
            "media_id": registration.media_id,
            "content_hash": registration.content_hash,
            "fingerprint_commitment": registration.fingerprint_commitment,
            "owner_id": registration.owner_id,
            "owner_public_key": registration.owner_public_key,
            "blockchain_txid": registration.blockchain_txid,
            "chain_mode": registration.chain_mode,
            "issued_at": datetime.now(tz=timezone.utc).isoformat(),
            "issuer": "Hikmaon",
        }
        signature = self._private_key.sign(_canonical_payload(fields))
        certificate = OwnershipCertificate(
            **fields,
            signature_b64=base64.b64encode(signature).decode(),
            signing_key_id=self.signing_key_id,
        )
        self.store.certificates[certificate.certificate_id] = certificate
        return certificate

    def verify(self, certificate_fields: dict) -> dict:
        try:
            payload = _canonical_payload(certificate_fields)
            signature = base64.b64decode(certificate_fields["signature_b64"])
        except (KeyError, ValueError, TypeError) as exc:
            return {"valid": False, "reason": f"malformed_certificate: {exc}"}

        public_key: Ed25519PublicKey = self._private_key.public_key()
        try:
            public_key.verify(signature, payload)
        except InvalidSignature:
            return {"valid": False, "reason": "signature_mismatch"}

        return {
            "valid": True,
            "signing_key_id": self.signing_key_id,
            "issuer_public_key_b64": self.public_key_b64,
            "chain_mode": certificate_fields.get("chain_mode"),
            "note": (
                "Signature verified against this Hikmaon instance's issuing key. "
                "Confirm the blockchain_txid on Hikmalayer for full chain-of-custody."
            ),
        }
