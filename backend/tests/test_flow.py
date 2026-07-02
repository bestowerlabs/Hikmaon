from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, store

client = TestClient(app)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _edited_copy(raw: bytes) -> bytes:
    """Re-encode + resize: a realistic 'stolen and reposted' transformation."""
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    image = image.resize((180, 180), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=70)
    return buffer.getvalue()


def test_full_lifecycle_connector_to_takedown(make_photo, to_bytes):
    original = to_bytes(make_photo(21))

    # Step 1: owner connects a social account.
    connected = client.post(
        "/api/connectors",
        json={
            "owner_id": "owner-01",
            "owner_public_key": "pubkey_owner_123456",
            "provider": "instagram",
            "account_handle": "@owner",
        },
    )
    assert connected.status_code == 200
    connector_id = connected.json()["connector_id"]

    # Step 2: upload event auto-registers + anchors + issues certificate.
    ingest = client.post(
        "/api/connectors/ingest",
        json={
            "connector_id": connector_id,
            "media_type": "image",
            "filename": "original.png",
            "content_b64": _b64(original),
            "source_url": "https://instagram.com/p/x",
        },
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["event"] == "connector_ingest_registered"
    media_id = body["media_id"]
    assert body["certificate_id"]
    assert body["chain_mode"] == "dev-simulated"

    certificate = client.get(f"/api/certificates/{media_id}")
    assert certificate.status_code == 200
    cert = certificate.json()
    verified = client.post("/api/certificates/verify", json={"certificate": cert})
    assert verified.json()["valid"] is True

    # Tampered certificate must fail verification.
    tampered = dict(cert, owner_id="attacker")
    assert client.post("/api/certificates/verify", json={"certificate": tampered}).json()["valid"] is False

    # Step 3a: an *edited* copy appears on the public internet and is indexed.
    stolen = _edited_copy(original)
    indexed = client.post(
        "/api/monitor/index",
        json={"media_url": "https://malicious.site/fake.jpg", "content_b64": _b64(stolen)},
    )
    assert indexed.status_code == 200

    # Step 3b: realtime detection matches the edited copy and opens an incident.
    detected = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "fake.jpg", "content_b64": _b64(stolen)},
    )
    assert detected.status_code == 200
    data = detected.json()
    assert data["event"] == "incident_created"
    incident = data["incident"]
    assert incident["matched_media_id"] == media_id
    assert incident["match_percentage"] >= 55.0
    assert incident["status"] == "pending_owner_review"
    assert incident["blockchain_verified"] is True
    assert "https://malicious.site/fake.jpg" in incident["matched_urls"]

    # Step 3c: owner refuses consent -> takedown case is opened automatically.
    decision = client.post(
        f"/api/incidents/{incident['incident_id']}/decision",
        json={"decision": "remove", "reason": "not authorized"},
    )
    assert decision.status_code == 200
    outcome = decision.json()
    assert outcome["status"] == "removal_requested"
    case = outcome["takedown_case"]
    assert case["status"] == "reported"
    assert "https://malicious.site/fake.jpg" in case["target_urls"]
    assert incident["match_percentage"] >= 55.0
    assert str(incident["match_percentage"]) in case["notice_text"]

    # Platform confirms removal; incident closes.
    closed = client.post(
        f"/api/takedowns/{case['case_id']}/status",
        json={"status": "removed", "note": "platform confirmed"},
    )
    assert closed.status_code == 200
    assert client.get("/api/incidents").json()[-1]["status"] == "closed" or (
        store.incidents[incident["incident_id"]].status == "closed"
    )

    # A second decision on the same incident is rejected.
    again = client.post(
        f"/api/incidents/{incident['incident_id']}/decision",
        json={"decision": "allow"},
    )
    assert again.status_code == 409


def test_unrelated_media_creates_no_incident(make_photo, to_bytes):
    registered = to_bytes(make_photo(31))
    client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-02",
            "owner_public_key": "pubkey_owner_222222",
            "media_type": "image",
            "filename": "mine.png",
            "content_b64": _b64(registered),
        },
    )

    incidents_before = len(client.get("/api/incidents").json())
    unrelated = to_bytes(make_photo(777))
    result = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "random.png", "content_b64": _b64(unrelated)},
    )
    assert result.status_code == 200
    assert result.json()["event"] == "no_match"
    assert len(client.get("/api/incidents").json()) == incidents_before


def test_owner_allow_closes_without_takedown(make_photo, to_bytes):
    original = to_bytes(make_photo(41))
    client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-03",
            "owner_public_key": "pubkey_owner_333333",
            "media_type": "image",
            "filename": "mine.png",
            "content_b64": _b64(original),
        },
    )
    detected = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "copy.jpg", "content_b64": _b64(_edited_copy(original))},
    )
    incident = detected.json()["incident"]
    cases_before = len(client.get("/api/takedowns").json())

    decision = client.post(
        f"/api/incidents/{incident['incident_id']}/decision",
        json={"decision": "allow"},
    )
    assert decision.json()["status"] == "allowed_by_owner"
    assert len(client.get("/api/takedowns").json()) == cases_before


def test_analyze_reports_match_percentage_and_forensics(make_photo, to_bytes):
    original = to_bytes(make_photo(51))
    client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-04",
            "owner_public_key": "pubkey_owner_444444",
            "media_type": "image",
            "filename": "mine.png",
            "content_b64": _b64(original),
        },
    )
    analysis = client.post(
        "/api/analyze",
        json={"media_type": "image", "filename": "probe.jpg", "content_b64": _b64(_edited_copy(original))},
    ).json()
    assert analysis["match"]["outcome"] == "match"
    assert analysis["match"]["match_percentage"] >= 55.0
    assert analysis["manipulation"]["verdict"] in (
        "no_artifacts_detected",
        "inconclusive",
        "manipulation_indicators",
    )
    assert analysis["manipulation"]["signals"]
    assert analysis["model_versions"]["perceptual_hash"] == "phash-dct-v1"


def test_ownership_signature_proof(make_photo, to_bytes):
    import hashlib

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw = to_bytes(make_photo(61))
    key = Ed25519PrivateKey.generate()
    public_b64 = base64.b64encode(key.public_key().public_bytes_raw()).decode()
    content_hash = hashlib.sha256(raw).hexdigest()
    signature_b64 = base64.b64encode(key.sign(content_hash.encode())).decode()

    response = client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-05",
            "owner_public_key": public_b64,
            "media_type": "image",
            "filename": "proved.png",
            "content_b64": _b64(raw),
            "ownership_signature_b64": signature_b64,
        },
    )
    assert response.status_code == 200
    assert response.json()["ownership_proven"] is True

    # A bad signature is rejected outright.
    bad = client.post(
        "/api/registrations",
        json={
            "owner_id": "owner-05",
            "owner_public_key": public_b64,
            "media_type": "image",
            "filename": "forged.png",
            "content_b64": _b64(to_bytes(make_photo(62))),
            "ownership_signature_b64": signature_b64,
        },
    )
    assert bad.status_code == 400
