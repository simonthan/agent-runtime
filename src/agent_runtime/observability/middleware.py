"""Pure-ASGI middleware that binds an X-Request-ID to the request contextvar.

Reuses an inbound ``X-Request-ID`` header (e.g. from a reverse proxy) or
generates a UUID4, binds it for the request's duration, and echoes it back on
the response ``X-Request-ID`` header. Framework-agnostic — register with
``app.add_middleware(RequestIDMiddleware)`` on any Starlette/FastAPI app, or
wrap any ASGI app directly. Depends on nothing outside the stdlib + this package.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agent_runtime.observability.request_context import _request_id_var

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]

# Accept only a conservative id charset (letters, digits, and ``. _ : -``) up to
# 128 chars — covers UUIDs and common proxy/trace ids. An inbound X-Request-ID
# that fails this (control/escape bytes a permissive ASGI server like httptools
# can pass through, or an oversized value) is rejected and a fresh UUID4 is
# generated instead, so attacker-controlled bytes are never reflected into the
# response header, bound to the contextvar, or written into every log line
# (log-injection / amplification defense-in-depth).
_SAFE_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


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
                candidate = value.decode("latin-1")
                return candidate if _SAFE_REQUEST_ID_RE.fullmatch(candidate) else None
        return None
