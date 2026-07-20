"""Pure-ASGI middleware that binds an X-Request-ID to the request contextvar.

Reuses an inbound ``X-Request-ID`` header (e.g. from a reverse proxy) or
generates a UUID4, binds it for the request's duration, and echoes it back on
the response ``X-Request-ID`` header. Framework-agnostic — register with
``app.add_middleware(RequestIDMiddleware)`` on any Starlette/FastAPI app, or
wrap any ASGI app directly. Depends on nothing outside the stdlib + this package.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agent_runtime.observability.request_context import _request_id_var

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class RequestIDMiddleware:
    """Bind a correlation id per HTTP request; echo it on the response."""

    def __init__(self, app: Callable, header_name: str = "X-Request-ID") -> None:
        self.app = app
        self._header = header_name.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._extract(scope) or str(uuid.uuid4())
        token = _request_id_var.set(request_id)
        encoded = request_id.encode("latin-1")

        async def send_with_header(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((self._header, encoded))
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            _request_id_var.reset(token)

    def _extract(self, scope: Scope) -> str | None:
        for name, value in scope.get("headers", []):
            if name.lower() == self._header:
                return value.decode("latin-1")
        return None
