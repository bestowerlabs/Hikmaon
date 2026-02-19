from __future__ import annotations

from dataclasses import dataclass, field

from app.models import EvidenceReport, NotificationRecord, RegistrationRecord


@dataclass
class InMemoryStore:
    registrations: dict[str, RegistrationRecord] = field(default_factory=dict)
    blockchain_records: dict[str, dict] = field(default_factory=dict)
    notifications: list[NotificationRecord] = field(default_factory=list)
    evidence_reports: dict[str, EvidenceReport] = field(default_factory=dict)
    crawler_index: dict[str, list[dict]] = field(default_factory=dict)
