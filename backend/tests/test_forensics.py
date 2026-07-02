from __future__ import annotations

import io

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.forensics import analyze_image_bytes


def test_natural_photo_reports_no_artifacts(make_photo, to_bytes):
    result = analyze_image_bytes(to_bytes(make_photo(11), "JPEG", quality=85))
    assert result.analyzable
    assert result.verdict == "no_artifacts_detected"
    assert result.risk_score < 0.2


def test_ai_generator_metadata_is_flagged(make_photo, to_bytes):
    metadata = PngInfo()
    metadata.add_text("parameters", "portrait, Steps: 30, Model: stable diffusion xl")
    result = analyze_image_bytes(to_bytes(make_photo(12), "PNG", pnginfo=metadata))
    assert result.verdict == "manipulation_indicators"
    assert any(s.name == "generator_metadata" and s.score >= 0.8 for s in result.signals)


def test_spliced_image_raises_risk_above_natural(make_photo, to_bytes):
    natural = analyze_image_bytes(to_bytes(make_photo(50, noise=0.01), "JPEG", quality=95))

    base = make_photo(50, noise=0.01)
    donor = make_photo(100, noise=0.002)
    buffer = io.BytesIO()
    donor.save(buffer, "JPEG", quality=35)
    donor = Image.open(buffer).convert("RGB")
    base.paste(donor.crop((60, 60, 180, 180)), (68, 68))
    spliced = analyze_image_bytes(to_bytes(base, "JPEG", quality=95))

    assert spliced.risk_score > natural.risk_score
    assert spliced.verdict in ("inconclusive", "manipulation_indicators")


def test_undecodable_media_abstains():
    result = analyze_image_bytes(b"\x00\x01binary-video-bytes" * 100)
    assert not result.analyzable
    assert result.verdict == "not_analyzable"


def test_every_signal_carries_explanation(make_photo, to_bytes):
    result = analyze_image_bytes(to_bytes(make_photo(13)))
    assert result.signals
    for signal in result.signals:
        assert signal.explanation
