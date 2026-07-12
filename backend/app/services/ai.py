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

import numpy as np

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
        self._img_index: tuple | None = None
        self._img_index_key: int = -1

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

    def _image_index(self) -> tuple:
        """Cached numpy arrays of image registrations for vectorized scoring.
        Rebuilt when the registration count changes (registrations are
        append-only, so the count is a sufficient cache key)."""
        key = len(self.store.registrations)
        if self._img_index_key == key and self._img_index is not None:
            return self._img_index
        regs = [r for r in self.store.registrations.values() if r.media_kind == "image" and r.phash_hex]
        if regs:
            phash = np.array([int(r.phash_hex, 16) for r in regs], dtype=np.uint64)
            dhash = np.array([int(r.dhash_hex, 16) for r in regs], dtype=np.uint64)
            emb = np.array([r.embedding for r in regs], dtype=np.float64)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            index = (regs, phash, dhash, emb)
        else:
            index = ([], None, None, None)
        self._img_index, self._img_index_key = index, key
        return index

    def _best_image_registration(self, probe):
        """Exact image-vs-image best match, vectorized (same formula as
        perceptual._match_images). Returns (registration, percentage) or None."""
        regs, phash, dhash, emb = self._image_index()
        if not regs:
            return None
        probe_p = np.uint64(int(probe.phash_hex, 16))
        probe_d = np.uint64(int(probe.dhash_hex, 16))
        probe_e = np.asarray(probe.embedding, dtype=np.float64)
        probe_e /= np.linalg.norm(probe_e) + 1e-8

        p_sim = 1.0 - np.bitwise_count(phash ^ probe_p) / 64.0
        d_sim = 1.0 - np.bitwise_count(dhash ^ probe_d) / 64.0
        e_sim = emb @ probe_e
        combined = (
            0.45 * np.clip((p_sim - 0.5) / 0.5, 0, 1)
            + 0.20 * np.clip((d_sim - 0.5) / 0.5, 0, 1)
            + 0.35 * np.clip((e_sim - 0.4) / 0.6, 0, 1)
        )
        best = int(np.argmax(combined))
        return regs[best], round(100.0 * float(combined[best]), 1)

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

        # Fast path: image probe vs image registrations, scored in one
        # vectorized pass instead of a per-registration Python loop.
        if probe.media_kind == "image":
            fast = self._best_image_registration(probe)
            if fast is not None:
                best_registration, best_pct = fast
                best_scores = match_percentage(probe, _registration_fingerprint(best_registration))

        # Remaining pairs (video/audio/binary, and cross-kind) — typically few.
        for registration in self.store.registrations.values():
            if probe.media_kind == "image" and registration.media_kind == "image":
                continue  # handled by the vectorized fast path
            scores = match_percentage(probe, _registration_fingerprint(registration))
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
