"""API authentication middleware."""

from __future__ import annotations

import secrets

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from trade.core.secrets import get_api_key

logger_instance = __import__("logging").getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Verify API key on incoming requests."""

    def __init__(self, app, skip_paths: list[str] | None = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/health", "/docs", "/openapi.json", "/redoc"]

    async def dispatch(self, request: Request, call_next):
        """Check API key for protected routes."""
        # Skip auth for certain paths
        if any(request.url.path.startswith(path) for path in self.skip_paths):
            return await call_next(request)

        api_key = get_api_key()

        # If no API key is configured, allow all requests (dev mode)
        if not api_key:
            logger_instance.warning("No API key configured - allowing all requests (dev mode)")
            return await call_next(request)

        # Extract provided API key
        provided_key = request.headers.get("X-API-Key", "")

        # Compare securely (constant-time comparison)
        if not secrets.compare_digest(provided_key, api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        return await call_next(request)
