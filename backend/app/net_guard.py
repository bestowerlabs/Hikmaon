"""Network egress guard: SSRF-safe outbound HTTP.

Outbound fetches driven by user- or remote-supplied URLs (crawler targets,
webhook media URLs, connector media URLs) must never be steerable at internal
services or cloud-metadata endpoints. Two protections live here:

1. ``url_is_allowed`` — permits only http/https whose host resolves *entirely*
   to globally-routable addresses (rejects loopback, private, link-local,
   reserved, and the 169.254.169.254 metadata range).
2. ``safe_get`` — follows redirects **manually**, re-validating every hop.
   This closes the classic bypass where a public URL 302-redirects to an
   internal host: with the stdlib/httpx auto-follow, only the first URL is
   checked; here each Location is validated before it is fetched.

Legitimate cross-host redirects to public CDNs (e.g. Microsoft Graph → a
storage host) still work — only non-public hops are refused.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx
from httpcore._backends.anyio import AnyIOBackend
from httpcore._backends.sync import SyncBackend

MAX_REDIRECTS = 5
_REDIRECT_CODES = {301, 302, 303, 307, 308}


class UnsafeURLError(Exception):
    """Raised when a URL (or a redirect hop) targets a non-public host."""


def host_is_public(hostname: str) -> bool:
    """True only if every resolved address for the host is globally routable."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return False
    return True


# --------------------------------------------------------------------------- #
# DNS-rebinding defense: pin the socket to an address we validated.
#
# host_is_public/url_is_allowed run at *check* time; the OS re-resolves at
# *connect* time, so a low-TTL attacker domain could rebind to an internal IP
# in between. These backends resolve once, reject if any address is non-public,
# and connect to that exact validated IP — closing the rebind window. TLS is
# untouched: httpcore still passes the original hostname as SNI / for cert
# verification, so certificates validate normally.
# --------------------------------------------------------------------------- #
def _validated_ip(host: str, port: int) -> str:
    try:
        infos = socket.getaddrinfo(host, port)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"cannot resolve {host}") from exc
    addresses = [info[4][0] for info in infos]
    if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
        raise UnsafeURLError(f"non-public host {host}")
    return addresses[0]


class _PinnedSyncBackend(SyncBackend):
    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return super().connect_tcp(
            _validated_ip(host, port), port,
            timeout=timeout, local_address=local_address, socket_options=socket_options,
        )


class _PinnedAsyncBackend(AnyIOBackend):
    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        return await super().connect_tcp(
            _validated_ip(host, port), port,
            timeout=timeout, local_address=local_address, socket_options=socket_options,
        )


def _pinned_sync_transport(**kwargs) -> httpx.HTTPTransport:
    transport = httpx.HTTPTransport(**kwargs)
    transport._pool._network_backend = _PinnedSyncBackend()
    return transport


def pinned_async_transport(**kwargs) -> httpx.AsyncHTTPTransport:
    """An httpx async transport that pins connections to validated public IPs.
    Used by the crawler's AsyncClient."""
    transport = httpx.AsyncHTTPTransport(**kwargs)
    transport._pool._network_backend = _PinnedAsyncBackend()
    return transport


def url_is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return host_is_public(parsed.hostname)


def _next_redirect(response: httpx.Response, current: str) -> str | None:
    if response.status_code in _REDIRECT_CODES and "location" in response.headers:
        return str(httpx.URL(current).join(response.headers["location"]))
    return None


def safe_get(url: str, *, headers: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    """Synchronous GET that validates the URL and every redirect hop.

    Raises ``UnsafeURLError`` if the URL or any redirect targets a non-public
    host; propagates ``httpx.HTTPError`` for transport failures.
    """
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, transport=_pinned_sync_transport()) as client:
        for _ in range(MAX_REDIRECTS + 1):
            if not url_is_allowed(current):
                raise UnsafeURLError(current)
            response = client.get(current, headers=headers)
            nxt = _next_redirect(response, current)
            if nxt is None:
                return response
            current = nxt
    raise UnsafeURLError(f"too many redirects from {url}")


async def safe_get_async(
    client: httpx.AsyncClient, url: str, *, url_validator=url_is_allowed
) -> httpx.Response:
    """Async GET over an existing client, validating the URL and each redirect.

    ``client`` must be created with ``follow_redirects=False``. ``url_validator``
    is injectable so the crawler can reuse its own (test-patchable) guard.
    Raises ``UnsafeURLError`` on a non-public URL or redirect hop.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not url_validator(current):
            raise UnsafeURLError(current)
        response = await client.get(current)
        nxt = _next_redirect(response, current)
        if nxt is None:
            return response
        current = nxt
    raise UnsafeURLError(f"too many redirects from {url}")
