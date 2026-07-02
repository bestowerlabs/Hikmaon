from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.hikmalayer import HikmalayerClient
from app.integrations.oauth import OAuthManager, ProviderNotConfigured
from app.integrations.providers import provider_status
from app.integrations.sync import MediaSyncService
from app.integrations.webhooks import WebhookError, WebhookService
from app.models import (
    AnalyzeRequest,
    CertificateVerifyRequest,
    ConnectorAccountCreate,
    ConnectorIngestEvent,
    IncidentDecision,
    RegistrationCreate,
    TakedownStatusUpdate,
    VerifyRequest,
)
from app.services.ai import AIService
from app.services.certificate import CertificateService
from app.services.connectors import ConnectorService
from app.services.evidence import EvidenceService
from app.services.monitoring import MonitoringService
from app.services.notification import NotificationService
from app.services.pipeline import AutomationPipelineService
from app.services.model_serving import DeepfakeModelServer
from app.services.registration import OwnershipProofError, RegistrationService
from app.services.takedown import TakedownService
from app.services.verification import VerificationService
from app.storage import InMemoryStore

app = FastAPI(title="Hikmaon API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("HIKMAON_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_data_dir_env = os.environ.get("HIKMAON_DATA_DIR", "data")
DATA_DIR: Path | None = Path(_data_dir_env) if _data_dir_env else None

store = InMemoryStore.load(DATA_DIR)
chain_client = HikmalayerClient(dev_ledger=store.blockchain_records)
certificate_service = CertificateService(store, key_path=(DATA_DIR / "signing_key.hex") if DATA_DIR else None)
registration_service = RegistrationService(store, chain_client, certificate_service)
model_server = DeepfakeModelServer()
ai_service = AIService(store, model_server=model_server)
verification_service = VerificationService(store, chain_client)
monitoring_service = MonitoringService(store)
evidence_service = EvidenceService(store)
notification_service = NotificationService(store)
connector_service = ConnectorService(store)
takedown_service = TakedownService(store, notification_service)
automation_pipeline = AutomationPipelineService(
    store=store,
    registration_service=registration_service,
    ai_service=ai_service,
    verification_service=verification_service,
    evidence_service=evidence_service,
    notification_service=notification_service,
)

oauth_manager = OAuthManager(store)
media_sync = MediaSyncService(store, oauth_manager, automation_pipeline)
webhook_service = WebhookService(store, automation_pipeline)

analysis_cache: dict[str, object] = {}
ANALYSIS_CACHE_LIMIT = 1000


class IndexRequest(BaseModel):
    media_url: str
    content_b64: str


class NotifyRequest(BaseModel):
    channel: str
    recipient: str
    message: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "chain_mode": chain_client.chain_mode}


# ---------------------------------------------------------------- registration


@app.post("/api/registrations")
def register(payload: RegistrationCreate) -> dict:
    try:
        record = registration_service.register_media(payload)
    except OwnershipProofError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.model_dump(exclude={"embedding", "chunk_fingerprints"})


@app.get("/api/registrations")
def list_registrations() -> list[dict]:
    return [
        record.model_dump(exclude={"embedding", "chunk_fingerprints"})
        for record in store.registrations.values()
    ]


# ---------------------------------------------------------------- certificates


@app.get("/api/certificates/{media_id}")
def get_certificate(media_id: str) -> dict:
    registration = store.registrations.get(media_id)
    if not registration or not registration.certificate_id:
        raise HTTPException(status_code=404, detail="Certificate not found")
    certificate = store.certificates.get(registration.certificate_id)
    if not certificate:
        raise HTTPException(status_code=404, detail="Certificate not found")
    return certificate.model_dump()


@app.post("/api/certificates/verify")
def verify_certificate(payload: CertificateVerifyRequest) -> dict:
    return certificate_service.verify(payload.certificate)


# ------------------------------------------------------------------ connectors


@app.post("/api/connectors")
def connect_account(payload: ConnectorAccountCreate) -> dict:
    return connector_service.connect_account(payload).model_dump()


@app.get("/api/connectors")
def list_connectors() -> list[dict]:
    return [connector.model_dump() for connector in connector_service.list_accounts()]


@app.delete("/api/connectors/{connector_id}")
def disconnect_connector(connector_id: str) -> dict:
    ok = connector_service.disconnect_account(connector_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Connector not found")
    return {"disconnected": True, "connector_id": connector_id}


@app.post("/api/connectors/ingest")
def ingest_connector_event(payload: ConnectorIngestEvent) -> dict:
    try:
        return automation_pipeline.ingest_from_connector(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------- platform API integrations


@app.get("/api/integrations/status")
def integrations_status() -> list[dict]:
    return provider_status()


@app.get("/api/model/status")
def model_status() -> dict:
    return model_server.status()


@app.get("/api/connectors/oauth/{provider}/start")
def oauth_start(provider: str, owner_id: str, owner_public_key: str) -> dict:
    try:
        return oauth_manager.start(provider, owner_id, owner_public_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.get("/api/connectors/oauth/{provider}/callback")
def oauth_callback(provider: str, code: str, state: str) -> dict:
    try:
        account = oauth_manager.callback(provider, code, state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return {"connected": True, **account.model_dump(exclude={"token_ciphertext"})}


@app.post("/api/connectors/{connector_id}/sync")
def sync_connector(connector_id: str) -> dict:
    account = store.connectors.get(connector_id)
    if not account:
        raise HTTPException(status_code=404, detail="Connector not found")
    try:
        return media_sync.sync(account)
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@app.get("/api/webhooks/{provider}")
def webhook_verify(provider: str, request: Request) -> PlainTextResponse:
    try:
        challenge = webhook_service.verify_subscription(provider, dict(request.query_params))
    except WebhookError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return PlainTextResponse(challenge)


@app.post("/api/webhooks/{provider}")
async def webhook_event(provider: str, request: Request) -> dict:
    body = await request.body()
    try:
        return webhook_service.handle_event(provider, body, dict(request.headers))
    except WebhookError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


# ------------------------------------------------------------------ monitoring


@app.post("/api/monitor/index")
def index_media(payload: IndexRequest) -> dict:
    media_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
    return monitoring_service.index_public_media(payload.media_url, media_bytes)


# -------------------------------------------------------------------- analysis


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    suspicious_media_id = f"sus_{uuid.uuid4().hex[:12]}"
    report = ai_service.analyze(suspicious_media_id, payload.content_b64)
    if len(analysis_cache) >= ANALYSIS_CACHE_LIMIT:
        analysis_cache.pop(next(iter(analysis_cache)))
    analysis_cache[suspicious_media_id] = report
    return report.model_dump()


@app.post("/api/realtime/detect")
def run_realtime_detection(payload: AnalyzeRequest) -> dict:
    return automation_pipeline.run_detection_cycle(payload.media_type, payload.filename, payload.content_b64)


# ------------------------------------------------------- incidents & takedowns


@app.get("/api/incidents")
def list_incidents() -> list[dict]:
    return [incident.model_dump() for incident in store.incidents.values()]


@app.post("/api/incidents/{incident_id}/decision")
def decide_incident(incident_id: str, payload: IncidentDecision) -> dict:
    incident = store.incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    try:
        return takedown_service.apply_owner_decision(incident, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/takedowns")
def list_takedowns() -> list[dict]:
    return [case.model_dump() for case in store.takedown_cases.values()]


@app.post("/api/takedowns/{case_id}/status")
def update_takedown(case_id: str, payload: TakedownStatusUpdate) -> dict:
    try:
        case = takedown_service.update_case_status(case_id, payload.status, payload.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Takedown case not found") from exc
    return case.model_dump()


# ------------------------------------------------------ verification & evidence


@app.post("/api/verify")
def verify(payload: VerifyRequest) -> dict:
    report = analysis_cache.get(payload.suspicious_media_id)
    if not report:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    if not report.match.matched_media_id:
        raise HTTPException(status_code=409, detail="Analysis found no matched registration to verify")
    registration = store.registrations.get(report.match.matched_media_id)
    if not registration:
        raise HTTPException(status_code=404, detail="Matched registration no longer exists")
    return verification_service.verify_registration(registration)


@app.post("/api/evidence/{suspicious_media_id}")
def evidence(suspicious_media_id: str) -> dict:
    report = analysis_cache.get(suspicious_media_id)
    if not report:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    try:
        evidence_report = evidence_service.generate_report(report)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return evidence_report.model_dump()


# --------------------------------------------------------------- notifications


@app.get("/api/notifications")
def list_notifications() -> list[dict]:
    return [record.model_dump() for record in store.notifications]


@app.post("/api/notifications")
def notify(payload: NotifyRequest) -> dict:
    record = notification_service.notify(payload.channel, payload.recipient, payload.message)
    return record.model_dump()
