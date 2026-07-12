from __future__ import annotations

from datetime import datetime, timezone

from app.models import RegistrationRecord
from app.perceptual import fingerprint_media, match_percentage
from app.services.ai import AIService, _registration_fingerprint
from app.storage import InMemoryStore


def _register(store, media_id, raw):
    fp = fingerprint_media(raw)
    store.registrations[media_id] = RegistrationRecord(
        media_id=media_id, owner_id="o", owner_public_key="k", media_type="image",
        filename="f", content_hash=media_id, fingerprint_commitment="c",
        media_kind=fp.media_kind, phash_hex=fp.phash_hex, dhash_hex=fp.dhash_hex,
        chunk_fingerprints=fp.chunks, embedding=fp.embedding, frame_phashes=fp.frame_phashes,
        audio_bits=fp.audio_bits, blockchain_txid="t", chain_mode="dev-simulated",
        created_at=datetime.now(timezone.utc),
    )


def test_vectorized_image_match_equals_bruteforce(make_photo, to_bytes):
    """The vectorized image fast-path must return exactly what the
    per-registration loop would for the winning match + percentage."""
    store = InMemoryStore()
    for i in range(20):
        _register(store, f"m{i}", to_bytes(make_photo(i)))
    ai = AIService(store)

    for target in (3, 11, 17):
        probe = fingerprint_media(to_bytes(make_photo(target)))
        # brute-force reference
        ref_id, ref = None, None
        for reg in store.registrations.values():
            s = match_percentage(probe, _registration_fingerprint(reg))
            if ref is None or s["match_percentage"] > ref["match_percentage"]:
                ref, ref_id = s, reg.media_id
        result = ai._find_best_match(probe)
        assert result.matched_media_id == ref_id == f"m{target}"
        assert result.match_percentage == ref["match_percentage"] == 100.0


def test_index_rebuilds_when_registrations_added(make_photo, to_bytes):
    store = InMemoryStore()
    _register(store, "a", to_bytes(make_photo(1)))
    ai = AIService(store)
    ai._find_best_match(fingerprint_media(to_bytes(make_photo(1))))  # builds index (1 reg)
    _register(store, "b", to_bytes(make_photo(2)))
    # New registration must be found (cache invalidated by count change).
    result = ai._find_best_match(fingerprint_media(to_bytes(make_photo(2))))
    assert result.matched_media_id == "b"
