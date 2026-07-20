"""Tests for the pure-ASGI RequestIDMiddleware.

Drives the middleware with hand-rolled ASGI scope/receive/send — no FastAPI
or Starlette needed (the lib has neither as a core dependency).
"""

from __future__ import annotations

from typing import Any

from agent_runtime.observability.middleware import RequestIDMiddleware
from agent_runtime.observability.request_context import get_request_id


async def _receive() -> dict[str, Any]:
    return {"type": "http.request"}


def _make_send() -> tuple[list[dict[str, Any]], Any]:
    messages: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    return messages, send


def _make_inner_app(captured: dict[str, Any]):
    async def app(_scope: dict[str, Any], _receive, send) -> None:
        captured["request_id"] = get_request_id()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


async def test_generates_when_no_header():
    captured: dict[str, Any] = {}
    inner = _make_inner_app(captured)
    middleware = RequestIDMiddleware(inner)
    messages, send = _make_send()

    await middleware({"type": "http", "headers": []}, _receive, send)

    assert captured["request_id"] is not None
    start = next(m for m in messages if m["type"] == "http.response.start")
    header_names = {name for name, _ in start["headers"]}
    assert b"x-request-id" in header_names
    header_value = next(v for name, v in start["headers"] if name == b"x-request-id")
    assert header_value.decode("latin-1") == captured["request_id"]


async def test_reuses_inbound_header():
    captured: dict[str, Any] = {}
    inner = _make_inner_app(captured)
    middleware = RequestIDMiddleware(inner)
    messages, send = _make_send()

    scope = {"type": "http", "headers": [(b"x-request-id", b"proxy-123")]}
    await middleware(scope, _receive, send)

    assert captured["request_id"] == "proxy-123"
    start = next(m for m in messages if m["type"] == "http.response.start")
    header_value = next(v for name, v in start["headers"] if name == b"x-request-id")
    assert header_value == b"proxy-123"


async def test_contextvar_reset_after_request():
    captured: dict[str, Any] = {}
    inner = _make_inner_app(captured)
    middleware = RequestIDMiddleware(inner)
    _, send = _make_send()

    await middleware({"type": "http", "headers": []}, _receive, send)

    assert get_request_id() is None


async def test_non_http_scope_passthrough():
    called = {"value": False}

    async def inner(_scope, _receive, _send) -> None:
        called["value"] = True

    middleware = RequestIDMiddleware(inner)
    _, send = _make_send()

    await middleware({"type": "lifespan"}, _receive, send)

    assert called["value"] is True
