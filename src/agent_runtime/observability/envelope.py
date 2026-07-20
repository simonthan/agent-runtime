"""Uniform error-envelope helper: {error, error_code, detail, request_id, timestamp}.

Pure data — the consumer wraps the returned dict in its framework's JSON
response type (e.g. FastAPI ``JSONResponse(status_code=..., content=...)``).
``request_id`` defaults to the current contextvar; ``timestamp`` to now (UTC ISO-8601).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_runtime.observability.request_context import get_request_id


def error_envelope(
    *,
    error: str,
    error_code: str,
    detail: Any = None,
    request_id: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the standard error envelope. See module docstring for field semantics."""
    return {
        "error": error,
        "error_code": error_code,
        "detail": detail,
        "request_id": request_id if request_id is not None else get_request_id(),
        "timestamp": timestamp if timestamp is not None else datetime.now(UTC).isoformat(),
    }
