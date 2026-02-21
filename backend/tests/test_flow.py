from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def test_connector_to_incident_automation_flow() -> None:
    media = b"owner-media-content"

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

    ingest = client.post(
        "/api/connectors/ingest",
        json={
            "connector_id": connector_id,
            "media_type": "image",
            "filename": "original.png",
            "content_b64": _b64(media),
            "source_url": "https://instagram.com/p/x",
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["event"] == "connector_ingest_registered"

    indexed = client.post(
        "/api/monitor/index",
        json={"media_url": "https://malicious.site/fake.png", "content_b64": _b64(media)},
    )
    assert indexed.status_code == 200

    detected = client.post(
        "/api/realtime/detect",
        json={
            "media_type": "image",
            "filename": "fake.png",
            "content_b64": _b64(media),
        },
    )
    assert detected.status_code == 200
    data = detected.json()
    assert data["event"] == "incident_created"
    assert data["verification"]["status"] == "verified"

    incidents = client.get("/api/incidents")
    assert incidents.status_code == 200
    assert len(incidents.json()) >= 1
