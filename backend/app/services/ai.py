"""AI analysis service: perceptual matching + manipulation forensics.

Produces two deliberately separate verdicts:

1. **match** — is this media derived from a registered original?
   (perceptual hash + embedding similarity, expressed as a 0-100%)
2. **manipulation** — does this media carry manipulation/AI-generation
   indicators? (forensic signal analysis)

Blockchain verification answers a third, orthogonal question — *who owns the
matched original* — and is reported under `ownership`, never blended into
the match or manipulation scores.
"""
from __future__ import annotations

import base64
import os

from app.forensics import FORENSICS_VERSION, analyze_image_bytes
from app.models import AnalysisReport, MatchResult, OwnershipResult
from app.services.model_serving import DeepfakeModelServer
from app.perceptual import (
    EMBEDDING_VERSION,
    PHASH_VERSION,
    fingerprint_media,
    hash_similarity,
    match_percentage,
)
from app.storage import InMemoryStore

# Calibrated on transformation experiments (see tests/test_perceptual.py):
# edited copies (re-encode/resize/blur/brightness/moderate crop) score 62-100%,
# unrelated media 0-28%.
DEFAULT_MATCH_THRESHOLD = 55.0
DEFAULT_REVIEW_THRESHOLD = 35.0


class AIService:
    def __init__(self, store: InMemoryStore, model_server: DeepfakeModelServer | None = None) -> None:
        self.store = store
        self.model_server = model_server or DeepfakeModelServer()
        self.match_threshold = float(os.environ.get("HIKMAON_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD))
        self.review_threshold = float(os.environ.get("HIKMAON_REVIEW_THRESHOLD", DEFAULT_REVIEW_THRESHOLD))

    def analyze(self, suspicious_media_id: str, media_bytes_b64: str) -> AnalysisReport:
        raw_bytes = base64.b64decode(media_bytes_b64.encode("utf-8"))
        probe = fingerprint_media(raw_bytes)

        match = self._find_best_match(probe)
        neural_probability = self.model_server.predict_probability(raw_bytes)
        manipulation = analyze_image_bytes(raw_bytes, neural_probability=neural_probability).to_dict()
        ownership = self._ownership(match)
        matched_urls = self._matched_urls(probe, match)

        return AnalysisReport(
            suspicious_media_id=suspicious_media_id,
            match=match,
            manipulation=manipulation,
            ownership=ownership,
            matched_urls=matched_urls,
            model_versions={
                "perceptual_hash": PHASH_VERSION,
                "embedding": EMBEDDING_VERSION,
                "forensics": FORENSICS_VERSION,
                "neural_detector": "hikmaonnet-v1" if neural_probability is not None else "not_deployed",
            },
        )

    def _find_best_match(self, probe) -> MatchResult:
        if not self.store.registrations:
            return MatchResult(
                matched=False,
                outcome="no_registrations",
                match_threshold=self.match_threshold,
                review_threshold=self.review_threshold,
            )

        best_scores: dict | None = None
        best_registration = None
        for registration in self.store.registrations.values():
            reg_fp = _registration_fingerprint(registration)
            scores = match_percentage(probe, reg_fp)
            if best_scores is None or scores["match_percentage"] > best_scores["match_percentage"]:
                best_scores = scores
                best_registration = registration

        percentage = best_scores["match_percentage"]
        if percentage >= self.match_threshold:
            outcome = "match"
        elif percentage >= self.review_threshold:
            outcome = "possible_match"
        else:
            outcome = "no_match"

        include_match = outcome in ("match", "possible_match")
        return MatchResult(
            matched=outcome == "match",
            outcome=outcome,
            match_percentage=percentage,
            matched_media_id=best_registration.media_id if include_match else None,
            matched_owner_id=best_registration.owner_id if include_match else None,
            component_scores=best_scores,
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
        )

    def _ownership(self, match: MatchResult) -> OwnershipResult:
        if not match.matched_media_id:
            return OwnershipResult(verified=False, detail="no matched registration to verify")
        registration = self.store.registrations[match.matched_media_id]
        return OwnershipResult(
            verified=False,  # verification service confirms against the chain
            txid=registration.blockchain_txid,
            chain_mode=registration.chain_mode,
            detail="pending chain verification",
        )

    def _matched_urls(self, probe, match: MatchResult) -> list[str]:
        """Public-index URLs whose fingerprints are perceptually close to the probe."""
        from app.perceptual import MediaFingerprint

        urls = []
        for item in self.store.crawler_index:
            item_fp = MediaFingerprint(
                media_kind=item.get("media_kind", "binary"),
                phash_hex=item.get("phash_hex"),
                dhash_hex=item.get("dhash_hex"),
                embedding=item.get("embedding", []),
                chunks=item.get("chunks", []),
                frame_phashes=item.get("frame_phashes", []),
                audio_bits=item.get("audio_bits", []),
            )
            try:
                scores = match_percentage(probe, item_fp)
            except (TypeError, ValueError):
                continue
            if scores["match_percentage"] >= 65.0:
                urls.append(item["url"])
        return urls


def _registration_fingerprint(registration):
    from app.perceptual import MediaFingerprint

    return MediaFingerprint(
        media_kind=registration.media_kind,
        phash_hex=registration.phash_hex,
        dhash_hex=registration.dhash_hex,
        embedding=registration.embedding,
        chunks=registration.chunk_fingerprints,
        frame_phashes=registration.frame_phashes,
        audio_bits=registration.audio_bits,
    )
