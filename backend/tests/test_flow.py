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


def test_full_lifecycle_connector_to_takedown(make_user, make_photo, to_bytes):
    headers, user, _ = make_user()
    original = to_bytes(make_photo(21))

    # Step 1: owner connects a social account.
    connected = client.post(
        "/api/connectors",
        json={"provider": "instagram", "account_handle": "@owner"},
        headers=headers,
    )
    assert connected.status_code == 200
    assert connected.json()["owner_id"] == user["user_id"]
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
        headers=headers,
    )
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["event"] == "connector_ingest_registered"
    media_id = body["media_id"]
    assert body["certificate_id"]
    assert body["chain_mode"] == "dev-simulated"
    # Connector-ingested media is auto-signed with the account key.
    assert store.registrations[media_id].ownership_proven is True

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
        headers=headers,
    )
    assert indexed.status_code == 200

    # Step 3b: realtime detection matches the edited copy and opens an incident.
    detected = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "fake.jpg", "content_b64": _b64(stolen)},
        headers=headers,
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
        headers=headers,
    )
    assert decision.status_code == 200
    outcome = decision.json()
    assert outcome["status"] == "removal_requested"
    case = outcome["takedown_case"]
    assert case["status"] == "reported"
    assert "https://malicious.site/fake.jpg" in case["target_urls"]
    assert str(incident["match_percentage"]) in case["notice_text"]

    # Platform confirms removal; incident closes.
    closed = client.post(
        f"/api/takedowns/{case['case_id']}/status",
        json={"status": "removed", "note": "platform confirmed"},
        headers=headers,
    )
    assert closed.status_code == 200
    assert store.incidents[incident["incident_id"]].status == "closed"

    # A second decision on the same incident is rejected.
    again = client.post(
        f"/api/incidents/{incident['incident_id']}/decision",
        json={"decision": "allow"},
        headers=headers,
    )
    assert again.status_code == 409


def test_other_users_cannot_touch_my_incident(make_user, make_photo, to_bytes):
    headers_owner, _, _ = make_user()
    headers_intruder, _, _ = make_user()

    original = to_bytes(make_photo(35))
    client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "mine.png", "content_b64": _b64(original)},
        headers=headers_owner,
    )
    detected = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "c.jpg", "content_b64": _b64(_edited_copy(original))},
        headers=headers_owner,
    ).json()
    incident_id = detected["incident"]["incident_id"]

    forbidden = client.post(
        f"/api/incidents/{incident_id}/decision",
        json={"decision": "allow"},
        headers=headers_intruder,
    )
    assert forbidden.status_code == 403

    intruder_view = client.get("/api/incidents", headers=headers_intruder).json()
    assert not any(i["incident_id"] == incident_id for i in intruder_view)


def test_unrelated_media_creates_no_incident(make_user, make_photo, to_bytes):
    headers, _, _ = make_user()
    registered = to_bytes(make_photo(31))
    client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "mine.png", "content_b64": _b64(registered)},
        headers=headers,
    )

    incidents_before = len(client.get("/api/incidents", headers=headers).json())
    unrelated = to_bytes(make_photo(777))
    result = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "random.png", "content_b64": _b64(unrelated)},
        headers=headers,
    )
    assert result.status_code == 200
    assert result.json()["event"] == "no_match"
    assert len(client.get("/api/incidents", headers=headers).json()) == incidents_before


def test_owner_allow_closes_without_takedown(make_user, make_photo, to_bytes):
    headers, _, _ = make_user()
    original = to_bytes(make_photo(41))
    client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "mine.png", "content_b64": _b64(original)},
        headers=headers,
    )
    detected = client.post(
        "/api/realtime/detect",
        json={"media_type": "image", "filename": "copy.jpg", "content_b64": _b64(_edited_copy(original))},
        headers=headers,
    )
    incident = detected.json()["incident"]
    cases_before = len(client.get("/api/takedowns", headers=headers).json())

    decision = client.post(
        f"/api/incidents/{incident['incident_id']}/decision",
        json={"decision": "allow"},
        headers=headers,
    )
    assert decision.json()["status"] == "allowed_by_owner"
    assert len(client.get("/api/takedowns", headers=headers).json()) == cases_before


def test_analyze_reports_match_percentage_and_forensics(make_user, make_photo, to_bytes):
    headers, _, _ = make_user()
    original = to_bytes(make_photo(51))
    client.post(
        "/api/registrations",
        json={"media_type": "image", "filename": "mine.png", "content_b64": _b64(original)},
        headers=headers,
    )
    analysis = client.post(
        "/api/analyze",
        json={"media_type": "image", "filename": "probe.jpg", "content_b64": _b64(_edited_copy(original))},
        headers=headers,
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


def test_explicit_ownership_signature_still_supported(make_user, make_photo, to_bytes):
    import hashlib

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    headers, _, _ = make_user()
    raw = to_bytes(make_photo(61))
    key = Ed25519PrivateKey.generate()
    public_b64 = base64.b64encode(key.public_key().public_bytes_raw()).decode()
    content_hash = hashlib.sha256(raw).hexdigest()
    signature_b64 = base64.b64encode(key.sign(content_hash.encode())).decode()

    response = client.post(
        "/api/registrations",
        json={
            "owner_public_key": public_b64,
            "media_type": "image",
            "filename": "proved.png",
            "content_b64": _b64(raw),
            "ownership_signature_b64": signature_b64,
        },
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()["ownership_proven"] is True
    assert response.json()["owner_public_key"] == public_b64

    # A signature that does not match the content is rejected outright.
    bad = client.post(
        "/api/registrations",
        json={
            "owner_public_key": public_b64,
            "media_type": "image",
            "filename": "forged.png",
            "content_b64": _b64(to_bytes(make_photo(62))),
            "ownership_signature_b64": signature_b64,
        },
        headers=headers,
    )
    assert bad.status_code == 400
