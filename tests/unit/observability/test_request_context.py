"""Tests for the request-id contextvar accessors + audit-inject hook."""

from agent_runtime.observability.request_context import (
    clear_request_id,
    generate_request_id,
    get_or_create_request_id,
    get_request_id,
    request_id_log_fields,
    set_request_id,
)


def test_get_returns_none_when_unset():
    assert get_request_id() is None


def test_set_then_get_roundtrip():
    set_request_id("abc-123")
    assert get_request_id() == "abc-123"


def test_generate_returns_and_sets():
    request_id = generate_request_id()
    assert get_request_id() == request_id
    assert len(request_id) == 36  # UUID4 canonical string length

    prefixed = generate_request_id(prefix="req-")
    assert prefixed.startswith("req-")
    assert get_request_id() == prefixed


def test_get_or_create_reuses_existing():
    set_request_id("abc")
    assert get_or_create_request_id() == "abc"


def test_get_or_create_generates_when_unset():
    clear_request_id()
    request_id = get_or_create_request_id()
    assert request_id is not None
    assert get_request_id() == request_id


def test_clear_resets_to_none():
    set_request_id("abc")
    clear_request_id()
    assert get_request_id() is None


def test_request_id_log_fields_present_when_set():
    set_request_id("r1")
    assert request_id_log_fields() == {"request_id": "r1"}


def test_request_id_log_fields_empty_when_unset():
    clear_request_id()
    assert request_id_log_fields() == {}
