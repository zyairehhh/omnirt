"""HTTP API key middleware."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


def load_api_keys(path: str | None) -> Set[str]:
    if not path:
        return set()
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"API key file not found: {file_path}")
    return {line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, api_keys: Iterable[str] | None = None) -> None:
        super().__init__(app)
        self.api_keys = set(api_keys or [])

    async def dispatch(self, request, call_next):
        if not self.api_keys or request.url.path in {"/healthz", "/readyz"}:
            return await call_next(request)

        provided = request.headers.get("x-api-key")
        auth_header = request.headers.get("authorization", "")
        if not provided and auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()

        if provided in self.api_keys:
            return await call_next(request)
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
