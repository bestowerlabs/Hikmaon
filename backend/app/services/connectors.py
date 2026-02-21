from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

from app.models import ConnectorAccount, ConnectorAccountCreate
from app.storage import InMemoryStore


class ConnectorService:
    """Optional social/cloud ingestion layer.

    In production this service should use provider OAuth and official APIs/webhooks.
    """

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def connect_account(self, payload: ConnectorAccountCreate) -> ConnectorAccount:
        # Placeholder encryption representation; replace with KMS-backed encryption.
        token_ciphertext = base64.b64encode(f"token::{payload.provider}::{payload.account_handle}".encode()).decode()
        account = ConnectorAccount(
            connector_id=f"conn_{uuid.uuid4().hex[:12]}",
            owner_id=payload.owner_id,
            owner_public_key=payload.owner_public_key,
            provider=payload.provider,
            account_handle=payload.account_handle,
            token_ciphertext=token_ciphertext,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.connectors[account.connector_id] = account
        return account

    def disconnect_account(self, connector_id: str) -> bool:
        return bool(self.store.connectors.pop(connector_id, None))

    def list_accounts(self) -> list[ConnectorAccount]:
        return list(self.store.connectors.values())
