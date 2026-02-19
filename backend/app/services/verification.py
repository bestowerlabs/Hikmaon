from __future__ import annotations

from app.models import RegistrationRecord
from app.storage import InMemoryStore


class VerificationService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def verify_registration(self, registration: RegistrationRecord) -> dict:
        chain_record = self.store.blockchain_records.get(registration.blockchain_txid)
        if not chain_record:
            return {"status": "not_verified", "reason": "tx_not_found"}

        payload = chain_record["payload"]
        hash_ok = payload["content_hash"] == registration.content_hash
        owner_ok = payload["owner_pubkey"] == registration.owner_public_key
        timestamp_ok = bool(payload.get("timestamp"))

        verified = hash_ok and owner_ok and timestamp_ok
        return {
            "status": "verified" if verified else "not_verified",
            "hash_matches": hash_ok,
            "owner_matches": owner_ok,
            "timestamp_exists": timestamp_ok,
            "txid": registration.blockchain_txid,
        }
