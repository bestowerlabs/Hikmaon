"""OAuth2 authorization-code flow (with PKCE) and encrypted token vault.

Flow:
1. ``GET /api/connectors/oauth/{provider}/start`` -> authorization URL.
   A one-time ``state`` (CSRF token) and PKCE verifier are stored server-side.
2. User authorizes on the platform; the platform redirects to
   ``/api/connectors/oauth/{provider}/callback?code=...&state=...``.
3. The code is exchanged for tokens over HTTPS and stored **Fernet-encrypted**
   (key = ``HIKMAON_TOKEN_KEY`` or derived from the instance signing key).
4. A ConnectorAccount is created; sync.py can now pull media with the token.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet

from app.integrations.providers import PROVIDERS, ProviderConfig
from app.models import ConnectorAccount
from app.storage import InMemoryStore

STATE_TTL_SECONDS = 600


class ProviderNotConfigured(Exception):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        super().__init__(
            f"{config.display_name} is not configured. Create an OAuth app at "
            f"{config.docs_url} and set {config.env_prefix}_CLIENT_ID / "
            f"{config.env_prefix}_CLIENT_SECRET, plus HIKMAON_OAUTH_REDIRECT_BASE."
        )


def _token_key(signing_seed: bytes | None = None) -> bytes:
    env_key = os.environ.get("HIKMAON_TOKEN_KEY")
    if env_key:
        return env_key.encode()
    seed = signing_seed or b"hikmaon-dev-token-key"
    return base64.urlsafe_b64encode(hashlib.sha256(seed).digest())


class OAuthManager:
    def __init__(self, store: InMemoryStore, signing_seed: bytes | None = None) -> None:
        self.store = store
        self.fernet = Fernet(_token_key(signing_seed))
        self.redirect_base = os.environ.get("HIKMAON_OAUTH_REDIRECT_BASE", "http://localhost:8000").rstrip("/")
        self._states: dict[str, dict] = {}

    # ------------------------------------------------------------ start
    def start(self, provider: str, owner_id: str, owner_public_key: str) -> dict:
        config = self._config(provider)
        state = secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

        self._states[state] = {
            "provider": provider,
            "owner_id": owner_id,
            "owner_public_key": owner_public_key,
            "verifier": verifier,
            "created": time.time(),
        }
        self._prune_states()

        params = {
            "client_id": config.client_id,
            "redirect_uri": self._redirect_uri(provider),
            "response_type": "code",
            "scope": " ".join(config.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return {
            "provider": provider,
            "authorization_url": f"{config.auth_url}?{urlencode(params)}",
            "state": state,
            "expires_in": STATE_TTL_SECONDS,
        }

    # --------------------------------------------------------- callback
    def callback(self, provider: str, code: str, state: str) -> ConnectorAccount:
        pending = self._states.pop(state, None)
        if not pending or pending["provider"] != provider:
            raise ValueError("invalid_or_expired_state")
        if time.time() - pending["created"] > STATE_TTL_SECONDS:
            raise ValueError("state_expired")

        config = self._config(provider)
        response = httpx.post(
            config.token_url,
            data={
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri(provider),
                "code_verifier": pending["verifier"],
            },
            headers={"Accept": "application/json"},
            timeout=20.0,
        )
        response.raise_for_status()
        tokens = response.json()

        vault_entry = {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": time.time() + float(tokens.get("expires_in", 3600)),
            "token_type": tokens.get("token_type", "Bearer"),
            "scope": tokens.get("scope"),
        }
        ciphertext = self.fernet.encrypt(json.dumps(vault_entry).encode()).decode()

        account = ConnectorAccount(
            connector_id=f"conn_{uuid.uuid4().hex[:12]}",
            owner_id=pending["owner_id"],
            owner_public_key=pending["owner_public_key"],
            provider=provider,
            account_handle=f"oauth:{provider}",
            token_ciphertext=ciphertext,
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.connectors[account.connector_id] = account
        self.store.persist()
        return account

    # ------------------------------------------------------------ tokens
    def access_token(self, account: ConnectorAccount) -> str:
        """Decrypt and return a live access token, refreshing if expired."""
        entry = json.loads(self.fernet.decrypt(account.token_ciphertext.encode()))
        if entry.get("expires_at", 0) > time.time() + 60 or not entry.get("refresh_token"):
            return entry["access_token"]

        config = self._config(account.provider)
        response = httpx.post(
            config.token_url,
            data={
                "client_id": config.client_id,
                "client_secret": config.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": entry["refresh_token"],
            },
            headers={"Accept": "application/json"},
            timeout=20.0,
        )
        response.raise_for_status()
        refreshed = response.json()
        entry["access_token"] = refreshed["access_token"]
        entry["expires_at"] = time.time() + float(refreshed.get("expires_in", 3600))
        if refreshed.get("refresh_token"):
            entry["refresh_token"] = refreshed["refresh_token"]
        account.token_ciphertext = self.fernet.encrypt(json.dumps(entry).encode()).decode()
        self.store.persist()
        return entry["access_token"]

    # ----------------------------------------------------------- helpers
    def _config(self, provider: str) -> ProviderConfig:
        config = PROVIDERS.get(provider)
        if config is None:
            raise KeyError(f"unknown provider: {provider}")
        if not config.configured:
            raise ProviderNotConfigured(config)
        return config

    def _redirect_uri(self, provider: str) -> str:
        return f"{self.redirect_base}/api/connectors/oauth/{provider}/callback"

    def _prune_states(self) -> None:
        cutoff = time.time() - STATE_TTL_SECONDS
        for key in [k for k, v in self._states.items() if v["created"] < cutoff]:
            self._states.pop(key, None)
