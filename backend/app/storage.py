from __future__ import annotations

from dataclasses import dataclass, field

from app.models import ConnectorAccount, EvidenceReport, IncidentRecord, NotificationRecord, RegistrationRecord


@dataclass
class InMemoryStore:
    registrations: dict[str, RegistrationRecord] = field(default_factory=dict)
    blockchain_records: dict[str, dict] = field(default_factory=dict)
    notifications: list[NotificationRecord] = field(default_factory=list)
    evidence_reports: dict[str, EvidenceReport] = field(default_factory=dict)
    crawler_index: dict[str, list[dict]] = field(default_factory=dict)
    connectors: dict[str, ConnectorAccount] = field(default_factory=dict)
    incidents: dict[str, IncidentRecord] = field(default_factory=dict)
