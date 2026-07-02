"""Platform API access: OAuth2 flows, webhooks, and media sync.

This package turns the connector module into real platform integrations.
Each provider activates when its OAuth app credentials are configured via
environment variables (see providers.py); until then the API returns precise
setup instructions instead of pretending.
"""
