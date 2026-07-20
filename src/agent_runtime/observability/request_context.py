"""Request-scoped correlation-ID storage (contextvar) + audit-inject hook.

Framework-agnostic. Works across async task boundaries for HTTP requests and
background jobs alike. Any knowledge-bot consumer wires this identically; the
lib's own agent-loop / circuit-breaker log lines read the same contextvar so a
single request id threads the whole stack.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Any

# Distinct name from any consumer's own contextvar to avoid collisions.
_request_id_var: ContextVar[str | None] = ContextVar("agent_runtime_request_id", default=None)


def get_request_id() -> str | None:
    """Return the current request id, or ``None`` if unset."""
    return _request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Bind ``request_id`` to the current context."""
    _request_id_var.set(request_id)


def generate_request_id(prefix: str = "") -> str:
    """Generate a UUID4 request id (optional ``prefix``), set it, and return it."""
    request_id = f"{prefix}{uuid.uuid4()}" if prefix else str(uuid.uuid4())
    set_request_id(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request id from the current context."""
    _request_id_var.set(None)


def get_or_create_request_id(prefix: str = "") -> str:
    """Return the existing request id, generating one (with ``prefix``) if unset."""
    request_id = get_request_id()
    if request_id is None:
        request_id = generate_request_id(prefix)
    return request_id


def request_id_log_fields() -> dict[str, Any]:
    """Audit-inject hook: ``{"request_id": <id>}`` when set, else ``{}``.

    Consumers fold this into their audit/log kwargs
    (``log_data |= request_id_log_fields()``) so every line auto-carries the id.
    No-op safe: returns an empty dict when no request id is bound.
    """
    request_id = get_request_id()
    return {"request_id": request_id} if request_id is not None else {}
