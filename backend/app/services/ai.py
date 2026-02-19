from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

import numpy as np

from app.models import AnalyzeRequest, RegistrationRecord
from app.storage import InMemoryStore


@dataclass
class AnalysisResult:
    suspicious_media_id: str
    similarity_score: float
    deepfake_probability: float
    blockchain_verified: bool
    confidence: float
    matched_registration: RegistrationRecord | None
    matched_urls: list[str]
    model_versions: dict[str, str]


class AIService:
    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def analyze(self, suspicious_media_id: str, payload: AnalyzeRequest) -> AnalysisResult:
        raw_bytes = base64.b64decode(payload.content_b64.encode("utf-8"))
        probe_embedding = self._embedding(raw_bytes)

        best_match = None
        best_score = -1.0
        for registration in self.store.registrations.values():
            score = self._cosine_similarity(probe_embedding, registration.embedding)
            if score > best_score:
                best_score = score
                best_match = registration

        deepfake_probability = self._deepfake_probability(raw_bytes)
        blockchain_verified = bool(best_match and best_match.blockchain_txid in self.store.blockchain_records)

        confidence = (best_score * 0.5) + (deepfake_probability * 0.3) + (0.2 if blockchain_verified else 0)
        matches = self.store.crawler_index.get(best_match.content_hash, []) if best_match else []

        return AnalysisResult(
            suspicious_media_id=suspicious_media_id,
            similarity_score=max(best_score, 0.0),
            deepfake_probability=deepfake_probability,
            blockchain_verified=blockchain_verified,
            confidence=confidence,
            matched_registration=best_match,
            matched_urls=[item["url"] for item in matches],
            model_versions={
                "embedding_model": "sim-encoder-v1",
                "deepfake_model": "deepfake-cnn-v1",
                "fusion_model": "decision-fusion-v1",
            },
        )

    def _embedding(self, raw_bytes: bytes, dim: int = 512) -> list[float]:
        seed = int(hashlib.sha256(raw_bytes).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        vector = rng.normal(0, 1, dim)
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).astype(float).tolist()

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        l = np.array(left)
        r = np.array(right)
        denom = np.linalg.norm(l) * np.linalg.norm(r)
        if denom == 0:
            return 0.0
        return float(np.dot(l, r) / denom)

    def _deepfake_probability(self, raw_bytes: bytes) -> float:
        digest = hashlib.sha256(raw_bytes).hexdigest()
        bucket = int(digest[-6:], 16) % 1000
        return round(bucket / 1000, 3)
