from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

from app.perceptual import chunk_fingerprints, chunk_similarity, fingerprint_media, match_percentage


def test_edited_copies_still_match(make_photo, to_bytes):
    original = make_photo(1)
    registered = fingerprint_media(to_bytes(original))

    edited_variants = {
        "jpeg_q75": to_bytes(original, "JPEG", quality=75),
        "jpeg_q40": to_bytes(original, "JPEG", quality=40),
        "resized_60pct": to_bytes(original.resize((153, 153), Image.LANCZOS)),
        "brightness_+25pct": to_bytes(ImageEnhance.Brightness(original).enhance(1.25)),
        "blurred": to_bytes(original.filter(ImageFilter.GaussianBlur(1.5))),
        "cropped_90pct": to_bytes(original.crop((12, 12, 244, 244))),
    }
    for name, variant_bytes in edited_variants.items():
        scores = match_percentage(fingerprint_media(variant_bytes), registered)
        assert scores["match_percentage"] >= 55.0, f"{name} fell below match threshold: {scores}"


def test_unrelated_images_do_not_match(make_photo, to_bytes):
    registered = fingerprint_media(to_bytes(make_photo(1)))
    for seed in (99, 7, 1234):
        scores = match_percentage(fingerprint_media(to_bytes(make_photo(seed))), registered)
        assert scores["match_percentage"] < 35.0, f"unrelated seed {seed} matched: {scores}"


def test_identical_image_scores_100(make_photo, to_bytes):
    raw = to_bytes(make_photo(2))
    scores = match_percentage(fingerprint_media(raw), fingerprint_media(raw))
    assert scores["match_percentage"] == 100.0


def test_binary_media_chunk_matching():
    video_like = bytes(range(256)) * 2048  # 512 KB pseudo-video
    prints = chunk_fingerprints(video_like)
    assert chunk_similarity(prints, chunk_fingerprints(video_like)) == 1.0
    # A trimmed copy still shares most chunks.
    assert chunk_similarity(prints, chunk_fingerprints(video_like[65536:])) > 0.5
    # Unrelated bytes share none.
    other = bytes(reversed(range(256))) * 2048
    assert chunk_similarity(prints, chunk_fingerprints(other)) == 0.0


def test_fingerprint_commitment_is_stable(make_photo, to_bytes):
    raw = to_bytes(make_photo(3))
    assert fingerprint_media(raw).commitment == fingerprint_media(raw).commitment
