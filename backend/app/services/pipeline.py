from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models import AnalyzeRequest, ConnectorIngestEvent, IncidentRecord, RegistrationCreate
from app.services.ai import AIService
from app.services.evidence import EvidenceService
from app.services.notification import NotificationService
from app.services.registration import RegistrationService
from app.services.verification import VerificationService
from app.storage import InMemoryStore


class AutomationPipelineService:
    """Coordinates auto-ingest -> chain anchoring -> AI monitoring -> verification -> alerting."""

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
        }

    def run_detection_cycle(self, media_type: str, filename: str, content_b64: str) -> dict:
        suspicious_media_id = f"sus_{uuid.uuid4().hex[:12]}"
        result = self.ai_service.analyze(
            suspicious_media_id=suspicious_media_id,
            payload=AnalyzeRequest(media_type=media_type, filename=filename, content_b64=content_b64),
        )
        if not result.matched_registration:
            return {"event": "no_match", "suspicious_media_id": suspicious_media_id}

        verification = self.verification_service.verify_registration(result.matched_registration)
        report = self.evidence_service.generate_report(result)

        message = (
            f"Hikmaon detected potential misuse for media {result.matched_registration.media_id}. "
            f"Confidence={result.confidence:.3f}, deepfake_probability={result.deepfake_probability:.3f}."
        )
        self.notification_service.notify(
            channel="dashboard",
            recipient=result.matched_registration.owner_id,
            message=message,
        )

        incident = IncidentRecord(
            incident_id=f"inc_{uuid.uuid4().hex[:12]}",
            suspicious_media_id=suspicious_media_id,
            matched_media_id=result.matched_registration.media_id,
            similarity_score=result.similarity_score,
            deepfake_probability=result.deepfake_probability,
            confidence=result.confidence,
            blockchain_verified=verification.get("status") == "verified",
            matched_urls=result.matched_urls,
            evidence_report_id=report.report_id,
            notified_owner=result.matched_registration.owner_id,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.incidents[incident.incident_id] = incident

        return {
            "event": "incident_created",
            "incident": incident.model_dump(),
            "verification": verification,
            "evidence": report.model_dump(),
        }
