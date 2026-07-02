"""Provider registry: OAuth2 endpoints, scopes, and media APIs per platform.

Each provider becomes ACTIVE when its app credentials are set:

    HIKMAON_<PROVIDER>_CLIENT_ID
    HIKMAON_<PROVIDER>_CLIENT_SECRET
    HIKMAON_OAUTH_REDIRECT_BASE   (e.g. https://app.hikmaon.com)

Register the OAuth app in each platform's developer console (see docs_url),
add `<redirect_base>/api/connectors/oauth/<provider>/callback` as the
authorized redirect URI, and export the credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    display_name: str
    auth_url: str
    token_url: str
    scopes: list[str] = field(default_factory=list)
    media_api: str = ""  # notes/endpoint used by sync.py
    docs_url: str = ""
    supports_webhooks: bool = False
    notes: str = ""

    @property
    def env_prefix(self) -> str:
        return f"HIKMAON_{self.name.upper()}"

    @property
    def client_id(self) -> str | None:
        return os.environ.get(f"{self.env_prefix}_CLIENT_ID")

    @property
    def client_secret(self) -> str | None:
        return os.environ.get(f"{self.env_prefix}_CLIENT_SECRET")

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


PROVIDERS: dict[str, ProviderConfig] = {
    "instagram": ProviderConfig(
        name="instagram",
        display_name="Instagram (Meta Graph API)",
        auth_url="https://www.facebook.com/v19.0/dialog/oauth",
        token_url="https://graph.facebook.com/v19.0/oauth/access_token",
        scopes=["instagram_basic", "pages_show_list"],
        media_api="https://graph.facebook.com/v19.0/me/media?fields=id,media_type,media_url,timestamp,permalink",
        docs_url="https://developers.facebook.com/docs/instagram-platform",
        supports_webhooks=True,
    ),
    "facebook": ProviderConfig(
        name="facebook",
        display_name="Facebook (Meta Graph API)",
        auth_url="https://www.facebook.com/v19.0/dialog/oauth",
        token_url="https://graph.facebook.com/v19.0/oauth/access_token",
        scopes=["user_photos", "user_videos"],
        media_api="https://graph.facebook.com/v19.0/me/photos/uploaded?fields=id,images,created_time",
        docs_url="https://developers.facebook.com/docs/graph-api",
        supports_webhooks=True,
    ),
    "google_drive": ProviderConfig(
        name="google_drive",
        display_name="Google Drive",
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        media_api="https://www.googleapis.com/drive/v3/files",
        docs_url="https://developers.google.com/drive/api",
        supports_webhooks=True,
        notes="Webhook = Drive push notifications (files.watch channels)",
    ),
    "youtube": ProviderConfig(
        name="youtube",
        display_name="YouTube (Google)",
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/youtube.readonly"],
        media_api="https://www.googleapis.com/youtube/v3/search?part=snippet&forMine=true&type=video",
        docs_url="https://developers.google.com/youtube/v3",
        supports_webhooks=True,
        notes="Webhook = PubSubHubbub push on channel uploads",
    ),
    "dropbox": ProviderConfig(
        name="dropbox",
        display_name="Dropbox",
        auth_url="https://www.dropbox.com/oauth2/authorize",
        token_url="https://api.dropboxapi.com/oauth2/token",
        scopes=["files.content.read"],
        media_api="https://api.dropboxapi.com/2/files/list_folder",
        docs_url="https://www.dropbox.com/developers/documentation/http/documentation",
        supports_webhooks=True,
    ),
    "onedrive": ProviderConfig(
        name="onedrive",
        display_name="OneDrive (Microsoft Graph)",
        auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
        scopes=["Files.Read", "offline_access"],
        media_api="https://graph.microsoft.com/v1.0/me/drive/recent",
        docs_url="https://learn.microsoft.com/graph/api/resources/onedrive",
        supports_webhooks=True,
    ),
    "x": ProviderConfig(
        name="x",
        display_name="X (Twitter API v2)",
        auth_url="https://x.com/i/oauth2/authorize",
        token_url="https://api.x.com/2/oauth2/token",
        scopes=["tweet.read", "users.read", "offline.access"],
        media_api="https://api.x.com/2/users/me/tweets?expansions=attachments.media_keys&media.fields=url,type",
        docs_url="https://developer.x.com/en/docs/x-api",
        notes="OAuth2 with PKCE required (code_challenge included automatically)",
    ),
    "tiktok": ProviderConfig(
        name="tiktok",
        display_name="TikTok",
        auth_url="https://www.tiktok.com/v2/auth/authorize/",
        token_url="https://open.tiktokapis.com/v2/oauth/token/",
        scopes=["user.info.basic", "video.list"],
        media_api="https://open.tiktokapis.com/v2/video/list/",
        docs_url="https://developers.tiktok.com/doc/login-kit-web",
    ),
    "linkedin": ProviderConfig(
        name="linkedin",
        display_name="LinkedIn",
        auth_url="https://www.linkedin.com/oauth/v2/authorization",
        token_url="https://www.linkedin.com/oauth/v2/accessToken",
        scopes=["r_basicprofile"],
        media_api="https://api.linkedin.com/v2/posts",
        docs_url="https://learn.microsoft.com/linkedin/",
    ),
    "reddit": ProviderConfig(
        name="reddit",
        display_name="Reddit",
        auth_url="https://www.reddit.com/api/v1/authorize",
        token_url="https://www.reddit.com/api/v1/access_token",
        scopes=["identity", "history"],
        media_api="https://oauth.reddit.com/user/{username}/submitted",
        docs_url="https://www.reddit.com/dev/api/",
    ),
    "snapchat": ProviderConfig(
        name="snapchat",
        display_name="Snapchat",
        auth_url="https://accounts.snapchat.com/accounts/oauth2/auth",
        token_url="https://accounts.snapchat.com/accounts/oauth2/token",
        scopes=["https://auth.snapchat.com/oauth2/api/user.display_name"],
        media_api="",
        docs_url="https://developers.snap.com/",
        notes=(
            "Snapchat exposes no consumer content-read API; media ingestion "
            "requires the on-device attestation plug-in path from the patent"
        ),
    ),
}


def provider_status() -> list[dict]:
    return [
        {
            "provider": config.name,
            "display_name": config.display_name,
            "configured": config.configured,
            "supports_webhooks": config.supports_webhooks,
            "docs_url": config.docs_url,
            "required_env": [f"{config.env_prefix}_CLIENT_ID", f"{config.env_prefix}_CLIENT_SECRET"],
            "notes": config.notes,
        }
        for config in PROVIDERS.values()
    ]
