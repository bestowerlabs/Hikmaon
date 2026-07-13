from __future__ import annotations

import asyncio
import base64
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.auth import AuthError, AuthService
from app.billing import BillingError, BillingService, plan_catalog
from app.hikmalayer import HikmalayerClient
from app.integrations.oauth import OAuthManager, ProviderNotConfigured
from app.integrations.providers import provider_status
from app.integrations.sync import MediaSyncService
from app.integrations.webhooks import WebhookError, WebhookService
from app.models import (
    AnalyzeRequest,
    ApiKeyCreate,
    CertificateVerifyRequest,
    CheckoutRequest,
    ConnectorAccountCreate,
    ConnectorIngestEvent,
    CrawlJobCreate,
    IncidentDecision,
    LicenseActivate,
    LicenseIssue,
    PlanSet,
    RegistrationCreate,
    TakedownStatusUpdate,
    TokenRefresh,
    UserAccount,
    UserLogin,
    UserRegister,
    VerifyRequest,
)
from app.services.ai import AIService
from app.services.certificate import CertificateService
from app.services.connectors import ConnectorService
from app.services.crawler import CrawlerService, autonomous_schedule
from app.services.evidence import EvidenceService
from app.services.model_serving import DeepfakeModelServer
from app.services.monitoring import MonitoringService
from app.services.notification import NotificationService
from app.services.pipeline import AutomationPipelineService
from app.services.registration import OwnershipProofError, RegistrationService
from app.services.takedown import TakedownService
from app.services.verification import VerificationService
from app.storage import InMemoryStore

app = FastAPI(title="Hikmaon API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("HIKMAON_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_data_dir_env = os.environ.get("HIKMAON_DATA_DIR", "data")
DATA_DIR: Path | None = Path(_data_dir_env) if _data_dir_env else None

store = InMemoryStore.load(DATA_DIR)
auth_service = AuthService(store, data_dir=DATA_DIR)
billing_service = BillingService(store, key_path=(DATA_DIR / "license_key.hex") if DATA_DIR else None)
chain_client = HikmalayerClient(dev_ledger=store.blockchain_records)
certificate_service = CertificateService(store, key_path=(DATA_DIR / "signing_key.hex") if DATA_DIR else None)


def _auto_signer(owner_id: str, content_hash: str) -> str | None:
    """Sign registrations with the owning account's key automatically."""
    user = store.users.get(owner_id)
    if user is None:
        return None
    return auth_service.sign_content_hash(user, content_hash)


registration_service = RegistrationService(store, chain_client, certificate_service, auto_signer=_auto_signer)
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
crawler_service = CrawlerService(
    store, monitoring_service, automation_pipeline, match_threshold=ai_service.match_threshold
)

analysis_cache: dict[str, object] = {}
ANALYSIS_CACHE_LIMIT = 1000


class IndexRequest(BaseModel):
    media_url: str
    content_b64: str


class NotifyRequest(BaseModel):
    channel: str
    recipient: str
    message: str


# ----------------------------------------------------------- authentication


def current_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> UserAccount:
    # Programmatic clients authenticate with an API key; interactive clients
    # with a JWT bearer token.
    if x_api_key:
        user = billing_service.resolve_api_key(x_api_key)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return user
    try:
        return auth_service.authenticate(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _is_admin(user: UserAccount) -> bool:
    return user.role == "admin"


def _bill(call):
    try:
        return call()
    except BillingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/api/auth/register", status_code=201)
def auth_register(payload: UserRegister) -> dict:
    try:
        user = auth_service.register(payload.email, payload.password, payload.display_name)
        return auth_service.login(payload.email, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/api/auth/login")
def auth_login(payload: UserLogin) -> dict:
    try:
        return auth_service.login(payload.email, payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/api/auth/refresh")
def auth_refresh(payload: TokenRefresh) -> dict:
    try:
        return auth_service.refresh(payload.refresh_token)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/api/auth/logout")
def auth_logout(payload: TokenRefresh) -> dict:
    auth_service.logout(payload.refresh_token)
    return {"logged_out": True}


@app.get("/api/auth/me")
def auth_me(user: UserAccount = Depends(current_user)) -> dict:
    return user.model_dump(exclude={"password_hash", "signing_key_ciphertext"})


# ------------------------------------------------------------------- billing


@app.get("/api/billing/plans")
def billing_plans() -> list[dict]:
    return plan_catalog()


@app.get("/api/billing/me")
def billing_me(user: UserAccount = Depends(current_user)) -> dict:
    return billing_service.account_summary(user)


@app.post("/api/billing/checkout")
def billing_checkout(payload: CheckoutRequest, user: UserAccount = Depends(current_user)) -> dict:
    return _bill(lambda: billing_service.create_checkout(user, payload.plan))


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request) -> dict:
    body = await request.body()
    return _bill(lambda: billing_service.handle_stripe_webhook(body, request.headers.get("stripe-signature")))


@app.post("/api/billing/license/activate")
def billing_license_activate(payload: LicenseActivate, user: UserAccount = Depends(current_user)) -> dict:
    return _bill(lambda: billing_service.activate_license(user, payload.license))


@app.post("/api/billing/license/issue")
def billing_license_issue(payload: LicenseIssue, user: UserAccount = Depends(current_user)) -> dict:
    if not _is_admin(user):
        raise HTTPException(status_code=403, detail="Admin only")
    return {"license": billing_service.issue_license(payload.email, payload.seats, payload.days)}


@app.post("/api/billing/dev/set-plan")
def billing_dev_set_plan(payload: PlanSet, user: UserAccount = Depends(current_user)) -> dict:
    return _bill(lambda: billing_service.dev_set_plan(user, payload.plan, payload.days))


# --------------------------------------------------------------------- api keys


@app.post("/api/apikeys", status_code=201)
def create_api_key(payload: ApiKeyCreate, user: UserAccount = Depends(current_user)) -> dict:
    token, record = _bill(lambda: billing_service.mint_api_key(user, payload.name))
    return {"api_key": token, "note": "Store this now — it is not shown again.", **record}


@app.get("/api/apikeys")
def list_api_keys(user: UserAccount = Depends(current_user)) -> list[dict]:
    return billing_service.list_api_keys(user)


@app.delete("/api/apikeys/{key_id}")
def revoke_api_key(key_id: str, user: UserAccount = Depends(current_user)) -> dict:
    if not billing_service.revoke_api_key(user, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"revoked": True, "key_id": key_id}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "chain_mode": chain_client.chain_mode}


# ---------------------------------------------------------------- registration


@app.post("/api/registrations")
def register(payload: RegistrationCreate, user: UserAccount = Depends(current_user)) -> dict:
    _bill(lambda: billing_service.require_registration_slot(user))
    payload.owner_id = user.user_id
    # Advanced callers may register under their own key with an explicit
    # signature; otherwise the account key signs automatically.
    if not (payload.owner_public_key and payload.ownership_signature_b64):
        payload.owner_public_key = user.owner_public_key
        payload.ownership_signature_b64 = None
    try:
        record = registration_service.register_media(payload)
    except OwnershipProofError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.model_dump(exclude={"embedding", "chunk_fingerprints", "frame_phashes", "audio_bits"})


@app.get("/api/registrations")
def list_registrations(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        record.model_dump(exclude={"embedding", "chunk_fingerprints", "frame_phashes", "audio_bits"})
        for record in store.registrations.values()
        if _is_admin(user) or record.owner_id == user.user_id
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
def connect_account(payload: ConnectorAccountCreate, user: UserAccount = Depends(current_user)) -> dict:
    payload.owner_id = user.user_id
    payload.owner_public_key = user.owner_public_key
    return connector_service.connect_account(payload).model_dump()


@app.get("/api/connectors")
def list_connectors(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        connector.model_dump()
        for connector in connector_service.list_accounts()
        if _is_admin(user) or connector.owner_id == user.user_id
    ]


@app.delete("/api/connectors/{connector_id}")
def disconnect_connector(connector_id: str, user: UserAccount = Depends(current_user)) -> dict:
    connector = store.connectors.get(connector_id)
    if not connector:
        raise HTTPException(status_code=404, detail="Connector not found")
    if connector.owner_id != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not your connector")
    connector_service.disconnect_account(connector_id)
    return {"disconnected": True, "connector_id": connector_id}


@app.post("/api/connectors/ingest")
def ingest_connector_event(payload: ConnectorIngestEvent, user: UserAccount = Depends(current_user)) -> dict:
    connector = store.connectors.get(payload.connector_id)
    if connector and connector.owner_id != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not your connector")
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
def oauth_start(provider: str, user: UserAccount = Depends(current_user)) -> dict:
    try:
        return oauth_manager.start(provider, user.user_id, user.owner_public_key)
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
def sync_connector(connector_id: str, user: UserAccount = Depends(current_user)) -> dict:
    account = store.connectors.get(connector_id)
    if not account:
        raise HTTPException(status_code=404, detail="Connector not found")
    if account.owner_id != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not your connector")
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


# --------------------------------------------------------- autonomous crawler


@app.post("/api/crawler/jobs", status_code=201)
async def create_crawl_job(payload: CrawlJobCreate, user: UserAccount = Depends(current_user)) -> dict:
    _bill(lambda: billing_service.require_feature(user, "crawler"))
    try:
        job = crawler_service.create_job(payload, owner_user_id=user.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    crawler_service.start_job(job)
    return job.model_dump()


@app.get("/api/crawler/jobs")
def list_crawl_jobs(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        job.model_dump()
        for job in store.crawl_jobs.values()
        if _is_admin(user) or job.owner_user_id == user.user_id
    ]


@app.get("/api/crawler/jobs/{job_id}")
def get_crawl_job(job_id: str, user: UserAccount = Depends(current_user)) -> dict:
    job = store.crawl_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Crawl job not found")
    if job.owner_user_id != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not your crawl job")
    return job.model_dump()


@app.on_event("startup")
async def start_autonomous_crawler() -> None:
    seeds = [s.strip() for s in os.environ.get("HIKMAON_CRAWLER_SEEDS", "").split(",") if s.strip()]
    interval = float(os.environ.get("HIKMAON_CRAWLER_INTERVAL_MINUTES", "0") or 0)
    if seeds and interval > 0:
        asyncio.get_event_loop().create_task(autonomous_schedule(crawler_service, seeds, interval))


# ------------------------------------------------------------------ monitoring


@app.post("/api/monitor/index")
def index_media(payload: IndexRequest, user: UserAccount = Depends(current_user)) -> dict:
    media_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
    return monitoring_service.index_public_media(payload.media_url, media_bytes)


# -------------------------------------------------------------------- analysis


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest, user: UserAccount = Depends(current_user)) -> dict:
    _bill(lambda: billing_service.check_and_count_analysis(user))
    suspicious_media_id = f"sus_{uuid.uuid4().hex[:12]}"
    report = ai_service.analyze(suspicious_media_id, payload.content_b64)
    if len(analysis_cache) >= ANALYSIS_CACHE_LIMIT:
        analysis_cache.pop(next(iter(analysis_cache)))
    analysis_cache[suspicious_media_id] = report
    return report.model_dump()


@app.post("/api/realtime/detect")
def run_realtime_detection(payload: AnalyzeRequest, user: UserAccount = Depends(current_user)) -> dict:
    _bill(lambda: billing_service.check_and_count_analysis(user))
    return automation_pipeline.run_detection_cycle(payload.media_type, payload.filename, payload.content_b64)


# ------------------------------------------------------- incidents & takedowns


@app.get("/api/incidents")
def list_incidents(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        incident.model_dump()
        for incident in store.incidents.values()
        if _is_admin(user) or incident.notified_owner == user.user_id
    ]


@app.post("/api/incidents/{incident_id}/decision")
def decide_incident(
    incident_id: str, payload: IncidentDecision, user: UserAccount = Depends(current_user)
) -> dict:
    incident = store.incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    if incident.notified_owner != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Only the media owner can decide this incident")
    try:
        return takedown_service.apply_owner_decision(incident, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/takedowns")
def list_takedowns(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        case.model_dump()
        for case in store.takedown_cases.values()
        if _is_admin(user) or case.owner_id == user.user_id
    ]


@app.post("/api/takedowns/{case_id}/status")
def update_takedown(
    case_id: str, payload: TakedownStatusUpdate, user: UserAccount = Depends(current_user)
) -> dict:
    case = store.takedown_cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Takedown case not found")
    if case.owner_id != user.user_id and not _is_admin(user):
        raise HTTPException(status_code=403, detail="Not your takedown case")
    case = takedown_service.update_case_status(case_id, payload.status, payload.note)
    return case.model_dump()


# ------------------------------------------------------ verification & evidence


@app.post("/api/verify")
def verify(payload: VerifyRequest, user: UserAccount = Depends(current_user)) -> dict:
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
def evidence(suspicious_media_id: str, user: UserAccount = Depends(current_user)) -> dict:
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
def list_notifications(user: UserAccount = Depends(current_user)) -> list[dict]:
    return [
        record.model_dump()
        for record in store.notifications
        if _is_admin(user) or record.recipient == user.user_id
    ]


@app.post("/api/notifications")
def notify(payload: NotifyRequest, user: UserAccount = Depends(current_user)) -> dict:
    record = notification_service.notify(payload.channel, payload.recipient, payload.message)
    return record.model_dump()
