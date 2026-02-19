from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import NotificationRecord
from app.storage import InMemoryStore


class NotificationService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def notify(self, channel: str, recipient: str, message: str) -> NotificationRecord:
        record = NotificationRecord(
            notification_id=f"notif_{uuid.uuid4().hex[:12]}",
            channel=channel,
            recipient=recipient,
            message=message,
            sent_at=datetime.now(tz=timezone.utc),
        )
        self.store.notifications.append(record)
        return record
