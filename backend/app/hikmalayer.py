"""Hikmalayer blockchain client.

Hikmalayer (the hybrid PoW+PoS chain) is a separate, independently developed
project. This module is Hikmaon's *client* to it — it never implements
consensus or ledger logic itself.

Modes:
- **rpc** — when ``HIKMALAYER_RPC_URL`` is set, transactions are submitted to
  the real node over HTTP with retry/backoff, and verification queries the
  node. Expected node API (adjust ``TX_SUBMIT_PATH``/``TX_QUERY_PATH`` to the
  node's actual routes):
    POST {url}/transactions          -> {"txid": "..."}
    GET  {url}/transactions/{txid}   -> {"type": ..., "payload": {...}}
- **dev-simulated** — without the env var, transactions land in a local
  persistent dev ledger. Every response is labelled with its mode so a
  simulated proof can never be mistaken for a chain-anchored one.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

TX_SUBMIT_PATH = "/transactions"
TX_QUERY_PATH = "/transactions/{txid}"
RETRY_DELAYS = (1.0, 2.0, 4.0)


@dataclass
class TxResult:
    txid: str
    chain_mode: str  # "rpc" | "dev-simulated"
    confirmed: bool


class HikmalayerClient:
    def __init__(self, dev_ledger: dict[str, dict], rpc_url: str | None = None) -> None:
        self.rpc_url = (rpc_url or os.environ.get("HIKMALAYER_RPC_URL") or "").rstrip("/") or None
        self.dev_ledger = dev_ledger

    @property
    def chain_mode(self) -> str:
        return "rpc" if self.rpc_url else "dev-simulated"

    def submit_media_registration(
        self,
        content_hash: str,
        fingerprint_commitment: str,
        owner_public_key: str,
        metadata_pointer: str,
    ) -> TxResult:
        payload = {
            "type": "MEDIA_REGISTRATION",
            "payload": {
                "content_hash": content_hash,
                "fingerprint_commitment": fingerprint_commitment,
                "owner_pubkey": owner_public_key,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "metadata_pointer": metadata_pointer,
            },
        }
        if self.rpc_url:
            return self._submit_rpc(payload)
        return self._submit_dev(payload)

    def get_transaction(self, txid: str) -> dict | None:
        if self.rpc_url:
            try:
                response = httpx.get(f"{self.rpc_url}{TX_QUERY_PATH.format(txid=txid)}", timeout=10.0)
                if response.status_code == 200:
                    return response.json()
                return None
            except httpx.HTTPError:
                return None
        return self.dev_ledger.get(txid)

    def _submit_rpc(self, payload: dict) -> TxResult:
        last_error: Exception | None = None
        for attempt, delay in enumerate((0.0, *RETRY_DELAYS)):
            if delay:
                time.sleep(delay)
            try:
                response = httpx.post(f"{self.rpc_url}{TX_SUBMIT_PATH}", json=payload, timeout=15.0)
                response.raise_for_status()
                txid = response.json()["txid"]
                return TxResult(txid=txid, chain_mode="rpc", confirmed=True)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"Hikmalayer RPC submission failed after retries: {last_error}")

    def _submit_dev(self, payload: dict) -> TxResult:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]
        txid = f"hkml_dev_{digest}{uuid.uuid4().hex[:8]}"
        self.dev_ledger[txid] = {**payload, "simulated": True}
        return TxResult(txid=txid, chain_mode="dev-simulated", confirmed=True)
