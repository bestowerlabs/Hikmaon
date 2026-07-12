"""Webhook receivers: platforms push upload events to Hikmaon in realtime.

Endpoint: ``/api/webhooks/{provider}``

- **GET** — subscription verification handshakes:
  Meta (Instagram/Facebook) sends ``hub.mode=subscribe&hub.verify_token=...&
  hub.challenge=...``; we echo the challenge when the verify token matches
  ``HIKMAON_WEBHOOK_VERIFY_TOKEN``.
- **POST** — event delivery. Authenticity checks, in order of preference:
  1. ``X-Hub-Signature-256`` (Meta-style HMAC of the raw body with the app
     secret) — verified when the provider app secret is configured.
  2. Shared secret header ``X-Hikmaon-Webhook-Secret`` matching
     ``HIKMAON_WEBHOOK_SHARED_SECRET`` — for platforms without signatures
     and for internal/collector agents.

Event bodies vary per platform; the generic payload accepted from any
verified source is:

    {"connector_id": "...", "media_type": "image",
     "filename": "...", "media_url": "https://...", "source_url": "..."}

The referenced media is downloaded and pushed through the standard ingest
pipeline (register -> anchor -> certificate). Provider-native payloads
(Meta change notifications, Dropbox cursors) should be translated into this
shape in `_extract_events` as each platform app goes live.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

import httpx

from app import net_guard
from app.integrations.providers import PROVIDERS
from app.models import ConnectorIngestEvent
from app.storage import InMemoryStore


class WebhookError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class WebhookService:
    def __init__(self, store: InMemoryStore, pipeline) -> None:
        self.store = store
        self.pipeline = pipeline

    # ----------------------------------------------------- GET handshake
    def verify_subscription(self, provider: str, params: dict) -> str:
        expected = os.environ.get("HIKMAON_WEBHOOK_VERIFY_TOKEN")
        if not expected:
            raise WebhookError(501, "Set HIKMAON_WEBHOOK_VERIFY_TOKEN to enable webhook subscriptions")
        if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == expected:
            return params.get("hub.challenge", "")
        raise WebhookError(403, "verify token mismatch")

    # ------------------------------------------------------ POST events
    def handle_event(self, provider: str, raw_body: bytes, headers: dict) -> dict:
        if provider not in PROVIDERS:
            raise WebhookError(404, f"unknown provider {provider}")
        self._authenticate(provider, raw_body, headers)

        import json

        try:
            payload = json.loads(raw_body)
        except ValueError as exc:
            raise WebhookError(400, f"invalid JSON body: {exc}") from exc

        results = []
        for event in self._extract_events(provider, payload):
            connector = self.store.connectors.get(event.get("connector_id", ""))
            if connector is None:
                results.append({"status": "skipped", "reason": "connector_not_found", **event})
                continue
            media_bytes = self._download(event["media_url"])
            if media_bytes is None:
                results.append({"status": "skipped", "reason": "download_failed", **event})
                continue
            ingest = self.pipeline.ingest_from_connector(
                ConnectorIngestEvent(
                    connector_id=connector.connector_id,
                    media_type=event.get("media_type", "image"),
                    filename=event.get("filename", "webhook-media"),
                    content_b64=base64.b64encode(media_bytes).decode(),
                    source_url=event.get("source_url", event["media_url"]),
                )
            )
            results.append({"status": "registered", "media_id": ingest["media_id"]})
        return {"provider": provider, "processed": len(results), "results": results}

    # ----------------------------------------------------------- helpers
    def _authenticate(self, provider: str, raw_body: bytes, headers: dict) -> None:
        headers = {k.lower(): v for k, v in headers.items()}

        signature = headers.get("x-hub-signature-256")
        app_secret = PROVIDERS[provider].client_secret
        if signature and app_secret:
            digest = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
            if hmac.compare_digest(signature, f"sha256={digest}"):
                return
            raise WebhookError(403, "HMAC signature mismatch")

        shared = os.environ.get("HIKMAON_WEBHOOK_SHARED_SECRET")
        if shared and hmac.compare_digest(headers.get("x-hikmaon-webhook-secret", ""), shared):
            return

        raise WebhookError(
            403,
            "unauthenticated webhook: provide X-Hub-Signature-256 (with provider app secret configured) "
            "or X-Hikmaon-Webhook-Secret matching HIKMAON_WEBHOOK_SHARED_SECRET",
        )

    def _extract_events(self, provider: str, payload: dict) -> list[dict]:
        # Generic Hikmaon shape (single event or list under "events").
        if "media_url" in payload:
            return [payload]
        if isinstance(payload.get("events"), list):
            return [e for e in payload["events"] if "media_url" in e]
        # Provider-native payload translation lands here as each platform
        # integration goes live (Meta "entry/changes", Dropbox cursors, ...).
        return []

    def _download(self, url: str) -> bytes | None:
        try:
            response = net_guard.safe_get(url)
            response.raise_for_status()
            return response.content
        except (httpx.HTTPError, net_guard.UnsafeURLError):
            return None
