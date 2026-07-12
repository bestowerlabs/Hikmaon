"""Authentication and account system.

Security design (modern best practice, no shortcuts):

- **Argon2id** password hashing (argon2-cffi defaults: 64 MiB memory cost,
  3 iterations) with transparent rehash-on-login when parameters change.
- **Password policy**: >= 10 chars, must mix letters and digits; the top of
  the worst-password list is rejected outright.
- **JWT access tokens** (HS256, 30 min) + **rotating refresh tokens**
  (random 256-bit, stored only as SHA-256 hashes, 14 days). Every refresh
  rotates the token; reuse of a rotated token is treated as theft and
  revokes the whole session family.
- **Login throttling**: 5 failed attempts per account -> 15 minute lockout.
  Failures are answered identically for unknown emails (no user enumeration).
- **Per-user Ed25519 keypair**, generated at registration. The public key is
  the user's on-chain ownership identity; the private key is stored
  Fernet-encrypted and used server-side to sign content hashes, so every
  registration carries a cryptographic ownership proof automatically.
- Secrets: ``HIKMAON_JWT_SECRET`` env, or an auto-generated secret persisted
  next to the data snapshot.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.models import UserAccount
from app.storage import InMemoryStore

ACCESS_TOKEN_MINUTES = 30
REFRESH_TOKEN_DAYS = 14
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60

_COMMON_PASSWORDS = {
    "password12", "password123", "qwerty12345", "1234567890", "letmein123",
    "iloveyou12", "admin12345", "welcome123", "abc1234567", "password1!",
}


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _password_policy(password: str) -> None:
    if len(password) < 10:
        raise AuthError(400, "Password must be at least 10 characters")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise AuthError(400, "Password must contain both letters and digits")
    if password.lower() in _COMMON_PASSWORDS:
        raise AuthError(400, "Password is too common")


def _valid_email(email: str) -> str:
    email = email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise AuthError(400, "Invalid email address")
    return email


class AuthService:
    def __init__(self, store: InMemoryStore, data_dir: Path | None = None) -> None:
        self.store = store
        self.hasher = PasswordHasher()
        self.jwt_secret = self._load_jwt_secret(data_dir)
        self.fernet = Fernet(self._vault_key())
        # In-memory throttling state (per-process; production: Redis).
        self._failures: dict[str, list[float]] = {}

    # ------------------------------------------------------------ secrets
    def _load_jwt_secret(self, data_dir: Path | None) -> str:
        env = os.environ.get("HIKMAON_JWT_SECRET")
        if env:
            return env
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)
            secret_path = data_dir / "auth_secret.hex"
            if secret_path.exists():
                return secret_path.read_text().strip()
            secret = secrets.token_hex(32)
            secret_path.write_text(secret)
            os.chmod(secret_path, 0o600)
            return secret
        return secrets.token_hex(32)

    def _vault_key(self) -> bytes:
        env = os.environ.get("HIKMAON_TOKEN_KEY")
        if env:
            return env.encode()
        return base64.urlsafe_b64encode(hashlib.sha256(self.jwt_secret.encode()).digest())

    # ----------------------------------------------------------- register
    def register(self, email: str, password: str, display_name: str) -> UserAccount:
        email = _valid_email(email)
        _password_policy(password)
        if any(u.email == email for u in self.store.users.values()):
            raise AuthError(409, "An account with this email already exists")

        signing_key = Ed25519PrivateKey.generate()
        public_key_b64 = base64.b64encode(signing_key.public_key().public_bytes_raw()).decode()
        key_ciphertext = self.fernet.encrypt(signing_key.private_bytes_raw()).decode()

        user = UserAccount(
            user_id=f"user_{uuid.uuid4().hex[:12]}",
            email=email,
            display_name=display_name.strip()[:80] or email.split("@")[0],
            password_hash=self.hasher.hash(password),
            owner_public_key=public_key_b64,
            signing_key_ciphertext=key_ciphertext,
            role="admin" if not self.store.users else "owner",
            created_at=datetime.now(tz=timezone.utc),
        )
        self.store.users[user.user_id] = user
        self.store.persist()
        return user

    # -------------------------------------------------------------- login
    def login(self, email: str, password: str) -> dict:
        email = _valid_email(email)
        self._check_lockout(email)

        user = next((u for u in self.store.users.values() if u.email == email), None)
        if user is None:
            # Burn comparable time to a real verification; identical error.
            self.hasher.hash(password)
            self._record_failure(email)
            raise AuthError(401, "Invalid email or password")

        try:
            self.hasher.verify(user.password_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            self._record_failure(email)
            raise AuthError(401, "Invalid email or password") from None

        if self.hasher.check_needs_rehash(user.password_hash):
            user.password_hash = self.hasher.hash(password)

        self._failures.pop(email, None)
        tokens = self._issue_tokens(user)
        self.store.persist()
        return tokens

    def _check_lockout(self, email: str) -> None:
        recent = [t for t in self._failures.get(email, []) if t > time.time() - LOCKOUT_SECONDS]
        self._failures[email] = recent
        if len(recent) >= MAX_FAILED_ATTEMPTS:
            raise AuthError(429, "Too many failed attempts; account temporarily locked")

    def _record_failure(self, email: str) -> None:
        self._failures.setdefault(email, []).append(time.time())

    # ------------------------------------------------------------- tokens
    def _issue_tokens(self, user: UserAccount, family: str | None = None) -> dict:
        now = int(time.time())
        access = jwt.encode(
            {
                "sub": user.user_id,
                "email": user.email,
                "role": user.role,
                "iat": now,
                "exp": now + ACCESS_TOKEN_MINUTES * 60,
                "iss": "hikmaon",
            },
            self.jwt_secret,
            algorithm="HS256",
        )
        refresh_plain = secrets.token_urlsafe(48)
        family = family or uuid.uuid4().hex
        self.store.refresh_tokens[hashlib.sha256(refresh_plain.encode()).hexdigest()] = {
            "user_id": user.user_id,
            "family": family,
            "expires": now + REFRESH_TOKEN_DAYS * 86400,
            "rotated": False,
        }
        return {
            "access_token": access,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_MINUTES * 60,
            "refresh_token": refresh_plain,
            "user": {
                "user_id": user.user_id,
                "email": user.email,
                "display_name": user.display_name,
                "owner_public_key": user.owner_public_key,
                "role": user.role,
            },
        }

    def refresh(self, refresh_token: str) -> dict:
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        record = self.store.refresh_tokens.get(token_hash)
        if record is None:
            raise AuthError(401, "Invalid refresh token")
        if record["rotated"]:
            # Reuse of a rotated token => theft signal: kill the whole family.
            for key, other in list(self.store.refresh_tokens.items()):
                if other["family"] == record["family"]:
                    self.store.refresh_tokens.pop(key, None)
            self.store.persist()
            raise AuthError(401, "Refresh token reuse detected; session revoked")
        if record["expires"] < time.time():
            self.store.refresh_tokens.pop(token_hash, None)
            raise AuthError(401, "Refresh token expired")

        user = self.store.users.get(record["user_id"])
        if user is None:
            raise AuthError(401, "Account no longer exists")
        record["rotated"] = True
        tokens = self._issue_tokens(user, family=record["family"])
        self.store.persist()
        return tokens

    def logout(self, refresh_token: str) -> None:
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        self.store.refresh_tokens.pop(token_hash, None)
        self.store.persist()

    # ------------------------------------------------------- verification
    def authenticate(self, authorization_header: str | None) -> UserAccount:
        if not authorization_header or not authorization_header.lower().startswith("bearer "):
            raise AuthError(401, "Missing bearer token")
        token = authorization_header.split(" ", 1)[1].strip()
        try:
            claims = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], issuer="hikmaon")
        except jwt.PyJWTError as exc:
            raise AuthError(401, f"Invalid token: {exc}") from exc
        user = self.store.users.get(claims.get("sub", ""))
        if user is None:
            raise AuthError(401, "Account no longer exists")
        return user

    # -------------------------------------------------- ownership signing
    def sign_content_hash(self, user: UserAccount, content_hash: str) -> str:
        """Sign a content hash with the user's Ed25519 key (ownership proof)."""
        seed = self.fernet.decrypt(user.signing_key_ciphertext.encode())
        key = Ed25519PrivateKey.from_private_bytes(seed)
        return base64.b64encode(key.sign(content_hash.encode("utf-8"))).decode()


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())
