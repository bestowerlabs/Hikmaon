from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import EvidenceReport
from app.services.ai import AnalysisResult
from app.storage import InMemoryStore


class EvidenceService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def generate_report(self, result: AnalysisResult) -> EvidenceReport:
        if not result.matched_registration:
            raise ValueError("Cannot generate evidence without matched registration")

        report = EvidenceReport(
            report_id=f"evidence_{uuid.uuid4().hex[:12]}",
            suspicious_media_id=result.suspicious_media_id,
            registered_txid=result.matched_registration.blockchain_txid,
            owner_public_key=result.matched_registration.owner_public_key,
            timestamp=datetime.now(tz=timezone.utc),
            similarity_score=result.similarity_score,
            deepfake_probability=result.deepfake_probability,
            matched_urls=result.matched_urls,
            analysis_metadata={
                "confidence": result.confidence,
                "blockchain_verified": result.blockchain_verified,
            },
            model_versions=result.model_versions,
        )
        self.store.evidence_reports[report.report_id] = report
        return report
