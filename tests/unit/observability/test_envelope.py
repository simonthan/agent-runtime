"""Tests for the pure-data error_envelope helper."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_runtime.observability.envelope import error_envelope
from agent_runtime.observability.request_context import clear_request_id, set_request_id


def test_shape_has_all_five_keys():
    clear_request_id()
    envelope = error_envelope(error="Something broke", error_code="SVC_004")
    assert set(envelope.keys()) == {"error", "error_code", "detail", "request_id", "timestamp"}


def test_request_id_defaults_to_contextvar():
    set_request_id("rid")
    envelope = error_envelope(error="x", error_code="SVC_004")
    assert envelope["request_id"] == "rid"


def test_request_id_none_when_unset():
    clear_request_id()
    envelope = error_envelope(error="x", error_code="SVC_004")
    assert envelope["request_id"] is None


def test_explicit_request_id_overrides_contextvar():
    set_request_id("rid")
    envelope = error_envelope(error="x", error_code="SVC_004", request_id="explicit")
    assert envelope["request_id"] == "explicit"


def test_timestamp_is_iso8601_utc():
    envelope = error_envelope(error="x", error_code="SVC_004")
    parsed = datetime.fromisoformat(envelope["timestamp"])
    assert parsed.tzinfo == UTC


def test_detail_passthrough():
    detail = [{"field": "x"}]
    envelope = error_envelope(error="x", error_code="SVC_004", detail=detail)
    assert envelope["detail"] == detail
