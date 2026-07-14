"""Subscriptions, enterprise licensing, API keys, and usage metering.

Plan tiers gate features and monthly usage:

- **free**      — evaluation: light quotas, no crawler, no API access.
- **pro**       — monthly subscription (Stripe): higher quotas, crawler, API keys.
- **enterprise**— licensed: an Ed25519-signed license the customer activates
  offline; high quotas and all features.

Metering is calendar-month based (resets on month rollover). Programmatic
access uses API keys (``hik_live_...``) stored only as SHA-256 hashes.

Payment provider (Stripe) is an integration point: ``/api/billing/checkout``
and the webhook activate when ``HIKMAON_STRIPE_KEY`` /
``HIKMAON_STRIPE_WEBHOOK_SECRET`` are set, and return precise setup guidance
otherwise. A dev path (``HIKMAON_BILLING_DEV=1``) lets you set plans without a
processor for testing.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from app.models import UserAccount
from app.storage import InMemoryStore


@dataclass(frozen=True)
class Plan:
    name: str
    price_usd_month: float | None  # None = custom / contact sales
    monthly_analyses: int
    max_registrations: int
    max_api_keys: int
    features: frozenset[str]


PLANS: dict[str, Plan] = {
    "free": Plan("free", 0.0, 100, 25, 0, frozenset()),
    "pro": Plan("pro", 49.0, 5_000, 1_000, 3, frozenset({"crawler", "api"})),
    "enterprise": Plan("enterprise", None, 1_000_000, 1_000_000, 50, frozenset({"crawler", "api", "priority"})),
}

_LICENSE_SIGNED_FIELDS = ("license_id", "email", "plan", "seats", "issued", "expires")


class BillingError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _month(dt: datetime | None = None) -> str:
    return (dt or _now()).strftime("%Y-%m")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class BillingService:
    def __init__(self, store: InMemoryStore, key_path: Path | None = None) -> None:
        self.store = store
        self._license_key = self._load_license_key(key_path)
        self.license_public_key_b64 = base64.b64encode(
            self._license_key.public_key().public_bytes_raw()
        ).decode()

    def _load_license_key(self, key_path: Path | None) -> Ed25519PrivateKey:
        seed = os.environ.get("HIKMAON_LICENSE_KEY")
        if seed:
            return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed))
        if key_path is not None:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            if key_path.exists():
                return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(key_path.read_text().strip()))
            key = Ed25519PrivateKey.generate()
            key_path.write_text(key.private_bytes_raw().hex())
            os.chmod(key_path, 0o600)
            return key
        return Ed25519PrivateKey.generate()

    # -------------------------------------------------------------- plans
    def plan_for(self, user: UserAccount) -> Plan:
        """Effective plan — a lapsed paid plan falls back to free."""
        if user.plan != "free":
            if user.subscription_status != "active" or (
                user.current_period_end and user.current_period_end < _now()
            ):
                return PLANS["free"]
        return PLANS.get(user.plan, PLANS["free"])

    # -------------------------------------------------------------- usage
    def _usage(self, user_id: str) -> dict:
        entry = self.store.usage.get(user_id)
        if not entry or entry.get("month") != _month():
            entry = {"month": _month(), "analyses": 0}
            self.store.usage[user_id] = entry
        return entry

    def check_and_count_analysis(self, user: UserAccount) -> None:
        plan = self.plan_for(user)
        usage = self._usage(user.user_id)
        if usage["analyses"] >= plan.monthly_analyses:
            raise BillingError(
                402,
                f"Monthly analysis quota reached ({plan.monthly_analyses} on the '{plan.name}' plan). "
                "Upgrade to continue.",
            )
        usage["analyses"] += 1
        self.store.persist()

    def require_feature(self, user: UserAccount, feature: str) -> None:
        if feature not in self.plan_for(user).features:
            raise BillingError(403, f"The '{feature}' feature requires a paid plan.")

    def require_registration_slot(self, user: UserAccount) -> None:
        plan = self.plan_for(user)
        count = sum(1 for r in self.store.registrations.values() if r.owner_id == user.user_id)
        if count >= plan.max_registrations:
            raise BillingError(
                402, f"Registration limit reached ({plan.max_registrations} on '{plan.name}'). Upgrade to add more."
            )

    def account_summary(self, user: UserAccount) -> dict:
        plan = self.plan_for(user)
        usage = self._usage(user.user_id)
        registrations = sum(1 for r in self.store.registrations.values() if r.owner_id == user.user_id)
        return {
            "plan": plan.name,
            "subscription_status": user.subscription_status,
            "current_period_end": user.current_period_end.isoformat() if user.current_period_end else None,
            "usage": {
                "month": usage["month"],
                "analyses_used": usage["analyses"],
                "analyses_limit": plan.monthly_analyses,
                "registrations_used": registrations,
                "registrations_limit": plan.max_registrations,
            },
            "features": sorted(plan.features),
            "api_keys_limit": plan.max_api_keys,
        }

    # ------------------------------------------------------------ api keys
    def mint_api_key(self, user: UserAccount, name: str) -> tuple[str, dict]:
        plan = self.plan_for(user)
        if "api" not in plan.features:
            raise BillingError(403, "API access requires a paid plan.")
        owned = [k for k in self.store.api_keys.values() if k["user_id"] == user.user_id]
        if len(owned) >= plan.max_api_keys:
            raise BillingError(402, f"API key limit reached ({plan.max_api_keys} on '{plan.name}').")
        token = "hik_live_" + secrets.token_urlsafe(24)
        record = {
            "key_id": f"ak_{uuid.uuid4().hex[:12]}",
            "user_id": user.user_id,
            "name": name[:60],
            "prefix": token[:16],
            "created_at": _now().isoformat(),
            "last_used": None,
        }
        self.store.api_keys[_hash(token)] = record
        self.store.persist()
        return token, record

    def list_api_keys(self, user: UserAccount) -> list[dict]:
        return [
            {k: v for k, v in rec.items() if k != "user_id"}
            for rec in self.store.api_keys.values()
            if rec["user_id"] == user.user_id
        ]

    def revoke_api_key(self, user: UserAccount, key_id: str) -> bool:
        for token_hash, rec in list(self.store.api_keys.items()):
            if rec["key_id"] == key_id and rec["user_id"] == user.user_id:
                self.store.api_keys.pop(token_hash)
                self.store.persist()
                return True
        return False

    def resolve_api_key(self, token: str) -> UserAccount | None:
        rec = self.store.api_keys.get(_hash(token))
        if not rec:
            return None
        rec["last_used"] = _now().isoformat()
        return self.store.users.get(rec["user_id"])

    # --------------------------------------------------- enterprise license
    def _license_payload(self, fields: dict) -> bytes:
        body = {name: str(fields[name]) for name in _LICENSE_SIGNED_FIELDS}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()

    def issue_license(self, email: str, seats: int, days: int) -> str:
        fields = {
            "license_id": f"lic_{uuid.uuid4().hex[:16]}",
            "email": email.strip().lower(),
            "plan": "enterprise",
            "seats": int(seats),
            "issued": _now().isoformat(),
            "expires": (_now() + timedelta(days=days)).isoformat(),
        }
        signature = self._license_key.sign(self._license_payload(fields))
        blob = {**fields, "signature": base64.b64encode(signature).decode()}
        return base64.urlsafe_b64encode(json.dumps(blob).encode()).decode()

    def activate_license(self, user: UserAccount, license_str: str) -> dict:
        try:
            blob = json.loads(base64.urlsafe_b64decode(license_str.encode()))
            signature = base64.b64decode(blob["signature"])
        except (ValueError, KeyError, TypeError) as exc:
            raise BillingError(400, f"Malformed license: {exc}") from exc

        public: Ed25519PublicKey = self._license_key.public_key()
        try:
            public.verify(signature, self._license_payload(blob))
        except InvalidSignature:
            raise BillingError(400, "Invalid license signature") from None

        if blob["email"].lower() not in (user.email.lower(), "*"):
            raise BillingError(403, "License is issued to a different account.")
        expires = datetime.fromisoformat(blob["expires"])
        if expires < _now():
            raise BillingError(400, "License has expired.")

        user.plan = "enterprise"
        user.subscription_status = "active"
        user.current_period_end = expires
        self.store.persist()
        return {"plan": "enterprise", "expires": blob["expires"], "seats": blob["seats"]}

    # ---------------------------------------------------------- stripe / dev
    def create_checkout(self, user: UserAccount, plan: str) -> dict:
        key = os.environ.get("HIKMAON_STRIPE_KEY")
        price = os.environ.get(f"HIKMAON_STRIPE_PRICE_{plan.upper()}")
        if not key or not price:
            raise BillingError(
                501,
                f"Payments not configured. Set HIKMAON_STRIPE_KEY and HIKMAON_STRIPE_PRICE_{plan.upper()} "
                "(a Stripe Price id), then this endpoint returns a Checkout URL.",
            )
        base = os.environ.get("HIKMAON_PUBLIC_URL", "http://localhost:8000").rstrip("/")
        response = httpx.post(
            "https://api.stripe.com/v1/checkout/sessions",
            auth=(key, ""),
            data={
                "mode": "subscription",
                "line_items[0][price]": price,
                "line_items[0][quantity]": 1,
                "client_reference_id": user.user_id,
                "customer_email": user.email,
                "success_url": f"{base}/billing/success",
                "cancel_url": f"{base}/billing/cancel",
            },
            timeout=20.0,
        )
        response.raise_for_status()
        return {"checkout_url": response.json()["url"]}

    def handle_stripe_webhook(self, raw_body: bytes, signature_header: str | None) -> dict:
        secret = os.environ.get("HIKMAON_STRIPE_WEBHOOK_SECRET")
        if not secret:
            raise BillingError(501, "Set HIKMAON_STRIPE_WEBHOOK_SECRET to enable billing webhooks.")
        if not self._verify_stripe_signature(raw_body, signature_header, secret):
            raise BillingError(403, "Invalid Stripe webhook signature.")

        event = json.loads(raw_body)
        obj = event.get("data", {}).get("object", {})
        event_type = event.get("type", "")

        user = self._user_from_stripe(obj)
        if user is None:
            return {"handled": False, "reason": "no matching account"}

        if event_type in ("checkout.session.completed", "customer.subscription.updated"):
            user.plan = "pro"
            user.subscription_status = "active"
            user.stripe_customer_id = obj.get("customer") or user.stripe_customer_id
            period_end = obj.get("current_period_end")
            user.current_period_end = (
                datetime.fromtimestamp(period_end, tz=timezone.utc) if period_end else _now() + timedelta(days=31)
            )
        elif event_type == "customer.subscription.deleted":
            user.plan = "free"
            user.subscription_status = "canceled"
        self.store.persist()
        return {"handled": True, "type": event_type, "user_id": user.user_id}

    def _verify_stripe_signature(self, raw_body: bytes, header: str | None, secret: str) -> bool:
        if not header:
            return False
        parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
        timestamp, provided = parts.get("t"), parts.get("v1")
        if not timestamp or not provided:
            return False
        expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, provided)

    def _user_from_stripe(self, obj: dict) -> UserAccount | None:
        user_id = obj.get("client_reference_id")
        if user_id and user_id in self.store.users:
            return self.store.users[user_id]
        customer = obj.get("customer")
        email = (obj.get("customer_email") or obj.get("customer_details", {}).get("email") or "").lower()
        for user in self.store.users.values():
            if (customer and user.stripe_customer_id == customer) or (email and user.email.lower() == email):
                return user
        return None

    def dev_set_plan(self, user: UserAccount, plan: str, days: int) -> dict:
        if os.environ.get("HIKMAON_BILLING_DEV") != "1":
            raise BillingError(403, "Manual plan changes are disabled (set HIKMAON_BILLING_DEV=1 for dev/testing).")
        user.plan = plan
        user.subscription_status = "active" if plan != "free" else "none"
        user.current_period_end = (_now() + timedelta(days=days)) if plan != "free" else None
        self.store.persist()
        return self.account_summary(user)


def plan_catalog() -> list[dict]:
    return [
        {
            "plan": p.name,
            "price_usd_month": p.price_usd_month,
            "monthly_analyses": p.monthly_analyses,
            "max_registrations": p.max_registrations,
            "max_api_keys": p.max_api_keys,
            "features": sorted(p.features),
        }
        for p in PLANS.values()
    ]
