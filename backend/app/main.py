from __future__ import annotations

import base64
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models import AnalyzeRequest, RegistrationCreate, VerifyRequest
from app.services.ai import AIService
from app.services.evidence import EvidenceService
from app.services.monitoring import MonitoringService
from app.services.notification import NotificationService
from app.services.registration import RegistrationService
from app.services.verification import VerificationService
from app.storage import InMemoryStore

app = FastAPI(title="Hikmaon API", version="0.1.0")
store = InMemoryStore()

registration_service = RegistrationService(store)
ai_service = AIService(store)
verification_service = VerificationService(store)
monitoring_service = MonitoringService(store)
evidence_service = EvidenceService(store)
notification_service = NotificationService(store)

analysis_cache: dict[str, object] = {}


class IndexRequest(BaseModel):
    media_url: str
    content_b64: str


class NotifyRequest(BaseModel):
    channel: str
    recipient: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/registrations")
def register(payload: RegistrationCreate) -> dict:
    record = registration_service.register_media(payload)
    return record.model_dump()


@app.post("/api/monitor/index")
def index_media(payload: IndexRequest) -> dict:
    media_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
    return monitoring_service.index_public_media(payload.media_url, media_bytes)


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    suspicious_media_id = f"sus_{uuid.uuid4().hex[:12]}"
    result = ai_service.analyze(suspicious_media_id, payload)
    if not result.matched_registration:
        raise HTTPException(status_code=404, detail="No registrations found to compare")

    analysis_cache[suspicious_media_id] = result
    return {
        "suspicious_media_id": suspicious_media_id,
        "match_media_id": result.matched_registration.media_id,
        "similarity_score": result.similarity_score,
        "deepfake_probability": result.deepfake_probability,
        "blockchain_verified": result.blockchain_verified,
        "confidence": result.confidence,
        "matched_urls": result.matched_urls,
        "model_versions": result.model_versions,
    }


@app.post("/api/verify")
def verify(payload: VerifyRequest) -> dict:
    result = analysis_cache.get(payload.suspicious_media_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    return verification_service.verify_registration(result.matched_registration)


@app.post("/api/evidence/{suspicious_media_id}")
def evidence(suspicious_media_id: str) -> dict:
    result = analysis_cache.get(suspicious_media_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    report = evidence_service.generate_report(result)
    return report.model_dump()


@app.post("/api/notifications")
def notify(payload: NotifyRequest) -> dict:
    record = notification_service.notify(payload.channel, payload.recipient, payload.message)
    return record.model_dump()
