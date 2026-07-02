from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import AnalysisReport, EvidenceReport
from app.storage import InMemoryStore


class EvidenceService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def generate_report(self, analysis: AnalysisReport, verification: dict | None = None) -> EvidenceReport:
        if not analysis.match.matched_media_id:
            raise ValueError("Cannot generate evidence without a matched registration")

        registration = self.store.registrations[analysis.match.matched_media_id]
        report = EvidenceReport(
            report_id=f"evidence_{uuid.uuid4().hex[:12]}",
            suspicious_media_id=analysis.suspicious_media_id,
            registered_txid=registration.blockchain_txid,
            owner_public_key=registration.owner_public_key,
            timestamp=datetime.now(tz=timezone.utc),
            match_percentage=analysis.match.match_percentage,
            manipulation_risk_score=float(analysis.manipulation.get("risk_score", 0.0)),
            manipulation_verdict=str(analysis.manipulation.get("verdict", "not_analyzable")),
            matched_urls=analysis.matched_urls,
            analysis_metadata={
                "match_outcome": analysis.match.outcome,
                "component_scores": analysis.match.component_scores,
                "manipulation_signals": analysis.manipulation.get("signals", []),
                "chain_verification": verification or {},
                "certificate_id": registration.certificate_id,
            },
            model_versions=analysis.model_versions,
        )
        self.store.evidence_reports[report.report_id] = report
        self.store.persist()
        return report
