from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RegistrationCreate(BaseModel):
    owner_id: str = Field(min_length=3)
    owner_public_key: str = Field(min_length=8)
    media_type: Literal["image", "video", "audio"]
    filename: str
    content_b64: str = Field(description="Base64-encoded media bytes")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistrationRecord(BaseModel):
    media_id: str
    owner_id: str
    owner_public_key: str
    media_type: str
    filename: str
    content_hash: str
    fingerprint_commitment: str
    embedding: list[float]
    metadata: dict[str, Any]
    blockchain_txid: str
    created_at: datetime


class AnalyzeRequest(BaseModel):
    media_type: Literal["image", "video", "audio"]
    filename: str
    content_b64: str


class VerifyRequest(BaseModel):
    suspicious_media_id: str


class NotificationRecord(BaseModel):
    notification_id: str
    channel: Literal["email", "dashboard", "webhook", "api_callback"]
    recipient: str
    message: str
    sent_at: datetime


class EvidenceReport(BaseModel):
    report_id: str
    suspicious_media_id: str
    registered_txid: str
    owner_public_key: str
    timestamp: datetime
    similarity_score: float
    deepfake_probability: float
    matched_urls: list[str]
    analysis_metadata: dict[str, Any]
    model_versions: dict[str, str]
