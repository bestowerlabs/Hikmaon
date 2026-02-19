from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def test_registration_analysis_verify_evidence_flow() -> None:
    content = b"sample-media-content"

    reg = client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-01",
            "owner_public_key": "pubkey_owner_123456",
            "media_type": "image",
            "filename": "original.png",
            "content_b64": _b64(content),
            "metadata": {"title": "Original"},
        },
    )
    assert reg.status_code == 200
    reg_data = reg.json()
    assert reg_data["blockchain_txid"].startswith("hkml_")

    idx = client.post(
        "/api/monitor/index",
        json={"media_url": "https://example.com/a.png", "content_b64": _b64(content)},
    )
    assert idx.status_code == 200

    analysis = client.post(
        "/api/analyze",
        json={
            "media_type": "image",
            "filename": "suspicious.png",
            "content_b64": _b64(content),
        },
    )
    assert analysis.status_code == 200
    analysis_data = analysis.json()
    assert analysis_data["similarity_score"] > 0.99

    verify = client.post(
        "/api/verify",
        json={"suspicious_media_id": analysis_data["suspicious_media_id"]},
    )
    assert verify.status_code == 200
    assert verify.json()["status"] == "verified"

    evidence = client.post(f"/api/evidence/{analysis_data['suspicious_media_id']}")
    assert evidence.status_code == 200
    evidence_data = evidence.json()
    assert evidence_data["registered_txid"] == reg_data["blockchain_txid"]
