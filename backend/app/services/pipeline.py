from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import (
    AnalysisReport,
    ConnectorIngestEvent,
    IncidentRecord,
    RegistrationCreate,
)
from app.services.ai import AIService
from app.services.evidence import EvidenceService
from app.services.notification import NotificationService
from app.services.registration import RegistrationService
from app.services.verification import VerificationService
from app.storage import InMemoryStore


class AutomationPipelineService:
    """Coordinates auto-ingest -> chain anchoring -> AI analysis -> verification -> alerting.

    Incidents are only opened for confirmed perceptual matches (>= match
    threshold); possible matches in the review band are surfaced in the
    analysis response without alerting, and non-matches end the cycle.
    """

    def __init__(
        self,
        store: InMemoryStore,
        registration_service: RegistrationService,
        ai_service: AIService,
        verification_service: VerificationService,
        evidence_service: EvidenceService,
        notification_service: NotificationService,
    ) -> None:
        self.store = store
        self.registration_service = registration_service
        self.ai_service = ai_service
        self.verification_service = verification_service
        self.evidence_service = evidence_service
        self.notification_service = notification_service

    def ingest_from_connector(self, event: ConnectorIngestEvent) -> dict:
        connector = self.store.connectors.get(event.connector_id)
        if not connector:
            raise ValueError("connector_not_found")

        registration = self.registration_service.register_media(
            RegistrationCreate(
                owner_id=connector.owner_id,
                owner_public_key=connector.owner_public_key,
                media_type=event.media_type,
                filename=event.filename,
                content_b64=event.content_b64,
                metadata={"source": connector.provider, "source_url": event.source_url},
            )
        )

        return {
            "event": "connector_ingest_registered",
            "connector_id": connector.connector_id,
            "provider": connector.provider,
            "media_id": registration.media_id,
            "blockchain_txid": registration.blockchain_txid,
            "chain_mode": registration.chain_mode,
            "certificate_id": registration.certificate_id,
        }

    def run_detection_cycle(self, media_type: str, filename: str, content_b64: str) -> dict:
        suspicious_media_id = f"sus_{uuid.uuid4().hex[:12]}"
        analysis = self.ai_service.analyze(suspicious_media_id, content_b64)

        if analysis.match.outcome in ("no_match", "no_registrations"):
            return {
                "event": "no_match",
                "suspicious_media_id": suspicious_media_id,
                "analysis": analysis.model_dump(),
            }

        if analysis.match.outcome == "possible_match":
            return {
                "event": "possible_match_review",
                "suspicious_media_id": suspicious_media_id,
                "analysis": analysis.model_dump(),
                "note": "Below confirmation threshold; queued for human review, owner not alerted",
            }

        incident = self._open_incident(suspicious_media_id, analysis)
        return {
            "event": "incident_created",
            "incident": incident.model_dump(),
            "analysis": analysis.model_dump(),
        }

    def _open_incident(self, suspicious_media_id: str, analysis: AnalysisReport) -> IncidentRecord:
        registration = self.store.registrations[analysis.match.matched_media_id]
        verification = self.verification_service.verify_registration(registration)
        analysis.ownership.verified = verification.get("status") == "verified"
        analysis.ownership.detail = f"chain check: {verification.get('status')}"
        report = self.evidence_service.generate_report(analysis, verification)

        incident = IncidentRecord(
            incident_id=f"inc_{uuid.uuid4().hex[:12]}",
            suspicious_media_id=suspicious_media_id,
            matched_media_id=registration.media_id,
            match_percentage=analysis.match.match_percentage,
            manipulation_risk_score=float(analysis.manipulation.get("risk_score", 0.0)),
            manipulation_verdict=str(analysis.manipulation.get("verdict", "not_analyzable")),
            blockchain_verified=analysis.ownership.verified,
            matched_urls=analysis.matched_urls,
            evidence_report_id=report.report_id,
            notified_owner=registration.owner_id,
            status="pending_owner_review",
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.incidents[incident.incident_id] = incident

        self.notification_service.notify(
            channel="dashboard",
            recipient=registration.owner_id,
            message=(
                f"Hikmaon detected a {analysis.match.match_percentage}% match to your registered media "
                f"{registration.media_id} (manipulation analysis: {incident.manipulation_verdict}). "
                f"Review incident {incident.incident_id} and choose Allow or Remove."
            ),
        )
        self.store.persist()
        return incident
