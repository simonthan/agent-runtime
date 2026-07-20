"""Correlation-ID + error-envelope observability primitives (T-060a)."""

from agent_runtime.observability.envelope import error_envelope
from agent_runtime.observability.middleware import RequestIDMiddleware
from agent_runtime.observability.request_context import (
    clear_request_id,
    generate_request_id,
    get_or_create_request_id,
    get_request_id,
    request_id_log_fields,
    set_request_id,
)

__all__ = [
    "RequestIDMiddleware",
    "clear_request_id",
    "error_envelope",
    "generate_request_id",
    "get_or_create_request_id",
    "get_request_id",
    "request_id_log_fields",
    "set_request_id",
]
