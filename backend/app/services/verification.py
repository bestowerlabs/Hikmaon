from __future__ import annotations

from app.hikmalayer import HikmalayerClient
from app.models import RegistrationRecord
from app.storage import InMemoryStore


class VerificationService:
    """Confirms a registration's ownership proof against Hikmalayer."""

    def __init__(self, store: InMemoryStore, chain_client: HikmalayerClient) -> None:
        self.store = store
        self.chain_client = chain_client

    def verify_registration(self, registration: RegistrationRecord) -> dict:
        chain_record = self.chain_client.get_transaction(registration.blockchain_txid)
        if not chain_record:
            return {
                "status": "not_verified",
                "reason": "tx_not_found",
                "txid": registration.blockchain_txid,
                "chain_mode": self.chain_client.chain_mode,
            }

        payload = chain_record.get("payload", {})
        hash_ok = payload.get("content_hash") == registration.content_hash
        fingerprint_ok = payload.get("fingerprint_commitment") == registration.fingerprint_commitment
        owner_ok = payload.get("owner_pubkey") == registration.owner_public_key
        timestamp_ok = bool(payload.get("timestamp"))

        verified = hash_ok and fingerprint_ok and owner_ok and timestamp_ok
        return {
            "status": "verified" if verified else "not_verified",
            "hash_matches": hash_ok,
            "fingerprint_matches": fingerprint_ok,
            "owner_matches": owner_ok,
            "timestamp_exists": timestamp_ok,
            "txid": registration.blockchain_txid,
            "chain_mode": self.chain_client.chain_mode,
            "simulated": bool(chain_record.get("simulated", False)),
            "ownership_proven_at_registration": registration.ownership_proven,
        }
