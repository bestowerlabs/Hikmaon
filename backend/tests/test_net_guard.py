from __future__ import annotations

import asyncio

import httpx
import pytest

from app import net_guard


def test_url_is_allowed_rejects_private_and_nonhttp(monkeypatch):
    monkeypatch.setattr(net_guard, "host_is_public", lambda host: host == "public.example")
    assert net_guard.url_is_allowed("https://public.example/x") is True
    assert net_guard.url_is_allowed("http://internal.local/x") is False
    assert net_guard.url_is_allowed("ftp://public.example/x") is False
    assert net_guard.url_is_allowed("https:///nohost") is False


def test_host_is_public_blocks_loopback_and_metadata():
    assert net_guard.host_is_public("127.0.0.1") is False
    assert net_guard.host_is_public("169.254.169.254") is False  # cloud metadata
    assert net_guard.host_is_public("10.0.0.5") is False
    assert net_guard.host_is_public("::1") is False


def _patch_client(monkeypatch, handler) -> None:
    """Make net_guard.safe_get build a mock-transport client."""
    real_client = httpx.Client

    def fake_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), follow_redirects=False)

    monkeypatch.setattr(net_guard.httpx, "Client", fake_client)


def test_safe_get_blocks_redirect_to_internal_host(monkeypatch):
    """A public URL that 302-redirects to the metadata endpoint must be refused."""
    monkeypatch.setattr(net_guard, "host_is_public", lambda host: host == "public.example")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, content=b"SECRET-CREDENTIALS")

    _patch_client(monkeypatch, handler)
    with pytest.raises(net_guard.UnsafeURLError):
        net_guard.safe_get("https://public.example/redirector")


def test_safe_get_follows_redirect_to_public_host(monkeypatch):
    """Cross-host redirects to other PUBLIC hosts are still followed (no breakage)."""
    monkeypatch.setattr(net_guard, "host_is_public", lambda host: host in ("a.example", "cdn.example"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"location": "https://cdn.example/file.jpg"})
        return httpx.Response(200, content=b"IMAGE-BYTES")

    _patch_client(monkeypatch, handler)
    response = net_guard.safe_get("https://a.example/start")
    assert response.status_code == 200
    assert response.content == b"IMAGE-BYTES"


def test_safe_get_async_blocks_internal_redirect():
    """The crawler's async path also refuses a redirect to an internal host."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/admin"})
        return httpx.Response(200, content=b"internal")

    def validator(url: str) -> bool:
        return "127.0.0.1" not in url and "://public.example" in url

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=False
        ) as client:
            with pytest.raises(net_guard.UnsafeURLError):
                await net_guard.safe_get_async(client, "https://public.example/x", url_validator=validator)

    asyncio.new_event_loop().run_until_complete(run())
