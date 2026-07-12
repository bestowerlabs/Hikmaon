from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MediaType = Literal["image", "video", "audio"]
ProviderType = Literal[
    "x",
    "instagram",
    "facebook",
    "youtube",
    "tiktok",
    "snapchat",
    "linkedin",
    "reddit",
    "google_drive",
    "dropbox",
    "onedrive",
]

IncidentStatus = Literal[
    "pending_owner_review",
    "allowed_by_owner",
    "removal_requested",
    "closed",
]

TakedownStatus = Literal["open", "reported", "removed", "rejected"]


class RegistrationCreate(BaseModel):
    # Owner identity is normally derived from the authenticated account;
    # explicit values are accepted only together with an ownership signature.
    owner_id: str = Field(default="", max_length=64)
    owner_public_key: str = Field(default="", max_length=128)
    media_type: MediaType
    filename: str
    content_b64: str = Field(description="Base64-encoded media bytes")
    metadata: dict[str, Any] = Field(default_factory=dict)
    ownership_signature_b64: str | None = Field(
        default=None,
        description="Ed25519 signature over the SHA-256 content hash, proving control of owner_public_key",
    )


class RegistrationRecord(BaseModel):
    media_id: str
    owner_id: str
    owner_public_key: str
    media_type: MediaType
    filename: str
    content_hash: str
    fingerprint_commitment: str
    media_kind: str  # "image" | "video" | "audio" | "binary" — what the engine decoded
    phash_hex: str | None = None
    dhash_hex: str | None = None
    chunk_fingerprints: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    frame_phashes: list[str] = Field(default_factory=list)
    audio_bits: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    blockchain_txid: str
    chain_mode: str
    ownership_proven: bool = False
    certificate_id: str | None = None
    created_at: datetime


class AnalyzeRequest(BaseModel):
    media_type: MediaType
    filename: str
    content_b64: str


class MatchResult(BaseModel):
    matched: bool
    outcome: Literal["match", "possible_match", "no_match", "no_registrations"]
    match_percentage: float = 0.0
    matched_media_id: str | None = None
    matched_owner_id: str | None = None
    component_scores: dict[str, Any] = Field(default_factory=dict)
    match_threshold: float
    review_threshold: float


class OwnershipResult(BaseModel):
    verified: bool
    txid: str | None = None
    chain_mode: str | None = None
    detail: str = ""


class AnalysisReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    suspicious_media_id: str
    match: MatchResult
    manipulation: dict[str, Any]
    ownership: OwnershipResult
    matched_urls: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)


class VerifyRequest(BaseModel):
    suspicious_media_id: str


class NotificationRecord(BaseModel):
    notification_id: str
    channel: Literal["email", "dashboard", "webhook", "api_callback"]
    recipient: str
    message: str
    sent_at: datetime


class EvidenceReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    report_id: str
    suspicious_media_id: str
    registered_txid: str
    owner_public_key: str
    timestamp: datetime
    match_percentage: float
    manipulation_risk_score: float
    manipulation_verdict: str
    matched_urls: list[str]
    analysis_metadata: dict[str, Any]
    model_versions: dict[str, str]


class ConnectorAccountCreate(BaseModel):
    # Owner identity is derived from the authenticated account.
    owner_id: str = ""
    owner_public_key: str = ""
    provider: ProviderType
    account_handle: str


class ConnectorAccount(BaseModel):
    connector_id: str
    owner_id: str
    owner_public_key: str
    provider: ProviderType
    account_handle: str
    token_ciphertext: str
    created_at: datetime


class ConnectorIngestEvent(BaseModel):
    connector_id: str
    media_type: MediaType
    filename: str
    content_b64: str
    source_url: str


class IncidentRecord(BaseModel):
    incident_id: str
    suspicious_media_id: str
    matched_media_id: str
    match_percentage: float
    manipulation_risk_score: float
    manipulation_verdict: str
    blockchain_verified: bool
    matched_urls: list[str]
    evidence_report_id: str
    notified_owner: str
    status: IncidentStatus = "pending_owner_review"
    owner_decision_at: datetime | None = None
    takedown_case_id: str | None = None
    created_at: datetime


class IncidentDecision(BaseModel):
    decision: Literal["allow", "remove"]
    reason: str | None = None


class TakedownCase(BaseModel):
    case_id: str
    incident_id: str
    matched_media_id: str
    owner_id: str
    target_urls: list[str]
    notice_text: str
    certificate_id: str | None = None
    status: TakedownStatus = "open"
    status_history: list[dict[str, str]] = Field(default_factory=list)
    created_at: datetime


class TakedownStatusUpdate(BaseModel):
    status: TakedownStatus
    note: str | None = None


class OwnershipCertificate(BaseModel):
    certificate_id: str
    media_id: str
    content_hash: str
    fingerprint_commitment: str
    owner_id: str
    owner_public_key: str
    blockchain_txid: str
    chain_mode: str
    # Kept as the exact signed string — re-serialization must never alter it.
    issued_at: str
    issuer: str = "Hikmaon"
    signature_b64: str
    signing_key_id: str


class CertificateVerifyRequest(BaseModel):
    certificate: dict[str, Any]


class UserAccount(BaseModel):
    user_id: str
    email: str
    display_name: str
    password_hash: str
    owner_public_key: str
    signing_key_ciphertext: str
    role: Literal["admin", "owner"] = "owner"
    created_at: datetime


class UserRegister(BaseModel):
    email: str
    password: str
    display_name: str = ""


class UserLogin(BaseModel):
    email: str
    password: str


class TokenRefresh(BaseModel):
    refresh_token: str


class CrawlJobCreate(BaseModel):
    seed_urls: list[str] = Field(min_length=1, max_length=20)
    allowed_domains: list[str] = Field(default_factory=list)
    max_pages: int = Field(default=50, ge=1, le=500)
    max_depth: int = Field(default=2, ge=0, le=4)


class CrawlJob(BaseModel):
    job_id: str
    owner_user_id: str
    seed_urls: list[str]
    allowed_domains: list[str]
    max_pages: int
    max_depth: int
    status: Literal["queued", "running", "completed", "failed"] = "queued"
    pages_crawled: int = 0
    media_indexed: int = 0
    matches_found: int = 0
    incidents: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime
    finished_at: datetime | None = None
