from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from app.models import (
    ConnectorAccount,
    CrawlJob,
    EvidenceReport,
    IncidentRecord,
    NotificationRecord,
    OwnershipCertificate,
    RegistrationRecord,
    TakedownCase,
    UserAccount,
)

DEFAULT_DATA_DIR = Path(os.environ.get("HIKMAON_DATA_DIR", "data"))


@dataclass
class InMemoryStore:
    """Application store with optional JSON snapshot persistence.

    Dev-grade persistence: the full state is written atomically after each
    mutating request so registrations, incidents, and dev-ledger entries
    survive restarts. Production replaces this with Postgres + object storage
    + a vector database behind the same attribute interface.
    """

    registrations: dict[str, RegistrationRecord] = field(default_factory=dict)
    blockchain_records: dict[str, dict] = field(default_factory=dict)
    notifications: list[NotificationRecord] = field(default_factory=list)
    evidence_reports: dict[str, EvidenceReport] = field(default_factory=dict)
    crawler_index: list[dict] = field(default_factory=list)
    connectors: dict[str, ConnectorAccount] = field(default_factory=dict)
    incidents: dict[str, IncidentRecord] = field(default_factory=dict)
    takedown_cases: dict[str, TakedownCase] = field(default_factory=dict)
    certificates: dict[str, OwnershipCertificate] = field(default_factory=dict)
    users: dict[str, UserAccount] = field(default_factory=dict)
    refresh_tokens: dict[str, dict] = field(default_factory=dict)
    crawl_jobs: dict[str, CrawlJob] = field(default_factory=dict)
    data_dir: Path | None = None

    def persist(self) -> None:
        if self.data_dir is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "registrations": {k: v.model_dump(mode="json") for k, v in self.registrations.items()},
            "blockchain_records": self.blockchain_records,
            "notifications": [n.model_dump(mode="json") for n in self.notifications],
            "evidence_reports": {k: v.model_dump(mode="json") for k, v in self.evidence_reports.items()},
            "crawler_index": self.crawler_index,
            "connectors": {k: v.model_dump(mode="json") for k, v in self.connectors.items()},
            "incidents": {k: v.model_dump(mode="json") for k, v in self.incidents.items()},
            "takedown_cases": {k: v.model_dump(mode="json") for k, v in self.takedown_cases.items()},
            "certificates": {k: v.model_dump(mode="json") for k, v in self.certificates.items()},
            "users": {k: v.model_dump(mode="json") for k, v in self.users.items()},
            "refresh_tokens": self.refresh_tokens,
            "crawl_jobs": {k: v.model_dump(mode="json") for k, v in self.crawl_jobs.items()},
        }
        target = self.data_dir / "hikmaon_store.json"
        fd, tmp_path = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(snapshot, handle)
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "InMemoryStore":
        store = cls(data_dir=data_dir)
        if data_dir is None:
            return store
        snapshot_path = data_dir / "hikmaon_store.json"
        if not snapshot_path.exists():
            return store
        raw = json.loads(snapshot_path.read_text())
        store.registrations = {k: RegistrationRecord(**v) for k, v in raw.get("registrations", {}).items()}
        store.blockchain_records = raw.get("blockchain_records", {})
        store.notifications = [NotificationRecord(**n) for n in raw.get("notifications", [])]
        store.evidence_reports = {k: EvidenceReport(**v) for k, v in raw.get("evidence_reports", {}).items()}
        store.crawler_index = raw.get("crawler_index", [])
        store.connectors = {k: ConnectorAccount(**v) for k, v in raw.get("connectors", {}).items()}
        store.incidents = {k: IncidentRecord(**v) for k, v in raw.get("incidents", {}).items()}
        store.takedown_cases = {k: TakedownCase(**v) for k, v in raw.get("takedown_cases", {}).items()}
        store.certificates = {k: OwnershipCertificate(**v) for k, v in raw.get("certificates", {}).items()}
        store.users = {k: UserAccount(**v) for k, v in raw.get("users", {}).items()}
        store.refresh_tokens = raw.get("refresh_tokens", {})
        store.crawl_jobs = {k: CrawlJob(**v) for k, v in raw.get("crawl_jobs", {}).items()}
        return store
