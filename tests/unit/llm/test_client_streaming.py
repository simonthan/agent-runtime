"""T-316b -- AnthropicClient streaming transport and the `on_stream` listener.

TBP needs (a) the model's text while it is generated, to show the answer as it is written, and
(b) a transport that does not wait on one HTTP read for a 16k-token reply. Default (no flag,
no listener) stays `messages.create` byte-for-byte.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("anthropic")

import anthropic
import httpx

from agent_runtime.llm import AnthropicClient, LLMAPIError, LLMRateLimitError
from agent_runtime.llm.streaming import StreamEvent, TextDelta, ToolUseStarted

from .conftest import RecordingAudit
from .fakes import FakeAsyncAnthropic, make_ok, make_tool_use

SYSTEM = [{"type": "text", "text": "sys"}]
MSGS = [{"role": "user", "content": "hi"}]


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    response = httpx.Response(status, request=request)
    cls = {
        400: anthropic.BadRequestError,
        404: anthropic.NotFoundError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }[status]
    return cls(f"status {status}", response=response, body=None)


class _Collect:
    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    async def __call__(self, event: StreamEvent) -> None:
        self.events.append(event)

    def text(self) -> str:
        return "".join(e.text for e in self.events if isinstance(e, TextDelta))


def _client(fake: FakeAsyncAnthropic, **kw: Any) -> AnthropicClient:
    return AnthropicClient(client=fake, **kw)


async def test_default_uses_create_not_stream() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="plain"))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert resp.content == "plain"
    assert fake.messages.stream_requests == []
    assert len(fake.messages.captured_requests) == 1


async def test_stream_transport_returns_the_same_response() -> None:
    plain_fake, streamed_fake = FakeAsyncAnthropic(), FakeAsyncAnthropic()
    plain_fake.messages.responses.append(make_ok(text="same answer", output_tokens=7))
    streamed_fake.messages.responses.append(make_ok(text="same answer", output_tokens=7))
    plain = await _client(plain_fake).complete_messages(system_blocks=SYSTEM, messages=MSGS)
    streamed = await _client(streamed_fake, stream_transport=True).complete_messages(
        system_blocks=SYSTEM, messages=MSGS
    )
    assert streamed == plain
    assert len(streamed_fake.messages.stream_requests) == 1
    # identical request kwargs: streaming changes the transport, not the request
    assert streamed_fake.messages.stream_requests[0] == plain_fake.messages.captured_requests[0]


async def test_listener_gets_text_deltas_in_order() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="The answer is forty-two."))
    seen = _Collect()
    resp = await _client(fake).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_stream=seen
    )
    assert seen.text() == resp.content == "The answer is forty-two."
    assert len([e for e in seen.events if isinstance(e, TextDelta)]) > 1
    assert len(fake.messages.stream_requests) == 1  # the listener alone turns streaming on


async def test_listener_gets_tool_use_started() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_tool_use(name="search", text="Checking."))
    seen = _Collect()
    resp = await _client(fake).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, tools=[{"name": "search"}], on_stream=seen
    )
    assert resp.tool_use[0].name == "search"
    assert seen.text() == "Checking."
    assert ToolUseStarted(name="search") in seen.events
    # the text came before the tool block started
    first_tool = next(i for i, e in enumerate(seen.events) if isinstance(e, ToolUseStarted))
    assert all(isinstance(e, TextDelta) for e in seen.events[:first_tool])


async def test_raising_listener_is_detached_not_fatal() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="still answered fine"))
    audit = RecordingAudit()
    calls = 0

    async def broken(_event: StreamEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("consumer bug")

    resp = await _client(fake, audit_logger=audit).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_stream=broken
    )
    assert resp.content == "still answered fine"
    assert calls == 1  # detached after the first failure
    failed = [e for e in audit.events if e[1] == "llm_stream_listener_failed"]
    assert len(failed) == 1
    assert failed[0][2]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("status", [400, 404])
async def test_unsupported_stream_falls_back_and_latches(status: int) -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.stream_open_exceptions.append(_status_error(status))
    fake.messages.responses.extend([make_ok(text="via create"), make_ok(text="second")])
    audit = RecordingAudit()
    client = _client(fake, stream_transport=True, audit_logger=audit)
    first = await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert first.content == "via create"
    unsupported = [e for e in audit.events if e[1] == "llm_stream_unsupported"]
    assert unsupported == [
        ("error", "llm_stream_unsupported", {"model": "claude-sonnet-4-6", "status_code": status})
    ]
    # latched: the next call does not try to stream again
    second = await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert second.content == "second"
    assert len(fake.messages.stream_requests) == 1
    assert len(fake.messages.captured_requests) == 3  # stream, create retry, create


async def test_a_stream_with_no_events_falls_back_and_latches() -> None:
    """An endpoint that ignores `stream: true` and answers plain JSON yields no stream event:
    retried once without streaming, then streaming stays off (no SDK AssertionError)."""
    fake = FakeAsyncAnthropic()
    fake.messages.stream_no_events = 1
    fake.messages.responses.extend([make_ok(text="json answer"), make_ok(text="next")])
    audit = RecordingAudit()
    client = _client(fake, stream_transport=True, audit_logger=audit)
    resp = await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert resp.content == "json answer"
    assert (
        "error",
        "llm_stream_unsupported",
        {"model": "claude-sonnet-4-6", "status_code": None},
    ) in (audit.events)
    assert (await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)).content == "next"
    assert len(fake.messages.stream_requests) == 1


async def test_bad_request_that_also_fails_without_streaming_is_not_latched() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.stream_open_exceptions.append(_status_error(400))
    fake.messages.exceptions.append(_status_error(400))
    fake.messages.responses.append(make_ok(text="later"))
    client = _client(fake, stream_transport=True)
    with pytest.raises(LLMAPIError):
        await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert len(fake.messages.stream_requests) == 2  # still streaming


async def test_rate_limit_on_stream_open_maps_without_fallback() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.stream_open_exceptions.append(_status_error(429))
    with pytest.raises(LLMRateLimitError):
        await _client(fake, stream_transport=True).complete_messages(
            system_blocks=SYSTEM, messages=MSGS
        )
    assert len(fake.messages.captured_requests) == 1  # no create retry


async def test_mid_stream_error_maps_and_is_not_retried() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="partial answer here"))
    fake.messages.stream_mid_exceptions.append((3, _status_error(500)))
    seen = _Collect()
    with pytest.raises(LLMAPIError):
        await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, on_stream=seen)
    assert len(fake.messages.captured_requests) == 1
    assert seen.text()  # some text had already arrived; the caller must treat it as void


async def test_complete_forwards_the_listener() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="kb answer"))
    seen = _Collect()
    resp = await _client(fake).complete(
        static_system_prefix="sys", user_message="q", on_stream=seen
    )
    assert seen.text() == resp.content == "kb answer"


async def test_complete_without_listener_does_not_pass_the_keyword() -> None:
    """A subclass / double of complete_messages without `on_stream` keeps working."""

    class _Narrow(AnthropicClient):
        async def complete_messages(self, **kwargs: Any):  # type: ignore[override]
            assert "on_stream" not in kwargs
            return await super().complete_messages(**kwargs)

    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="ok"))
    resp = await _Narrow(client=fake).complete(static_system_prefix="sys", user_message="q")
    assert resp.content == "ok"


# -- R3 (T-316b review) HIGH-1 / HIGH-2 / MEDIUM: failures after the stream started -------------


def _in_stream_error(error_type: str) -> anthropic.APIStatusError:
    """An `event: error` the API sends inside a stream: it rides a 200 response."""
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    return anthropic.APIStatusError(
        "in-stream error",
        response=httpx.Response(200, request=request),
        body={"type": "error", "error": {"type": error_type, "message": "x"}},
    )


async def test_dropped_connection_mid_stream_maps_to_llm_api_error() -> None:
    fake = FakeAsyncAnthropic()
    fake.max_retries = 1
    fake.messages.responses.append(make_ok(text="partial answer here"))
    fake.messages.stream_mid_exceptions.append((3, httpx.RemoteProtocolError("peer closed")))
    seen = _Collect()
    with pytest.raises(LLMAPIError, match="RemoteProtocolError"):
        await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, on_stream=seen)
    assert seen.text()  # text was forwarded: never re-generated
    assert len(fake.messages.stream_requests) == 1


async def test_failure_before_anything_was_shown_is_retried_once() -> None:
    fake = FakeAsyncAnthropic()
    fake.max_retries = 1
    fake.messages.responses.extend([make_ok(text="lost"), make_ok(text="second try")])
    fake.messages.stream_mid_exceptions.append((1, httpx.ReadTimeout("stalled")))
    audit = RecordingAudit()
    resp = await _client(fake, stream_transport=True, audit_logger=audit).complete_messages(
        system_blocks=SYSTEM, messages=MSGS
    )
    assert resp.content == "second try"
    assert len(fake.messages.stream_requests) == 2
    assert [e[1] for e in audit.events].count("llm_stream_retry") == 1


async def test_no_retry_when_the_client_allows_none() -> None:
    fake = FakeAsyncAnthropic()  # max_retries=0
    fake.messages.responses.append(make_ok(text="lost"))
    fake.messages.stream_mid_exceptions.append((1, _in_stream_error("overloaded_error")))
    with pytest.raises(LLMAPIError):
        await _client(fake, stream_transport=True).complete_messages(
            system_blocks=SYSTEM, messages=MSGS
        )
    assert len(fake.messages.stream_requests) == 1


async def test_a_stream_cut_before_message_stop_is_not_an_answer() -> None:
    fake = FakeAsyncAnthropic()
    fake.max_retries = 1
    fake.messages.responses.append(make_ok(text="half an answer"))
    fake.messages.stream_truncated = 1
    audit = RecordingAudit()
    with pytest.raises(LLMAPIError, match="IncompleteStream"):
        await _client(fake, stream_transport=True, audit_logger=audit).complete_messages(
            system_blocks=SYSTEM, messages=MSGS
        )
    assert ("llm_api_error", "IncompleteStream") in [
        (e[1], e[2].get("error_type")) for e in audit.events
    ]


async def test_in_stream_rate_limit_error_maps_to_rate_limit() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="some text first"))
    fake.messages.stream_mid_exceptions.append((3, _in_stream_error("rate_limit_error")))
    with pytest.raises(LLMRateLimitError):
        await _client(fake).complete_messages(
            system_blocks=SYSTEM, messages=MSGS, on_stream=_Collect()
        )


# -- iav review (T-316b) M1 / M2: failures after the stream opened, before the first event --------


async def test_an_error_event_before_the_first_event_is_retried_once() -> None:
    """An `event: error` (overloaded) arriving as the FIRST event is read from the body after
    the SDK's send(), so the SDK never retried it; nothing was shown, so it is retried here."""
    fake = FakeAsyncAnthropic()
    fake.max_retries = 1
    fake.messages.responses.extend([make_ok(text="lost"), make_ok(text="second try")])
    fake.messages.stream_mid_exceptions.append((0, _in_stream_error("overloaded_error")))
    audit = RecordingAudit()
    resp = await _client(fake, stream_transport=True, audit_logger=audit).complete_messages(
        system_blocks=SYSTEM, messages=MSGS
    )
    assert resp.content == "second try"
    assert len(fake.messages.stream_requests) == 2
    assert [e[1] for e in audit.events].count("llm_stream_retry") == 1


async def test_an_empty_event_stream_is_an_error_not_a_latch() -> None:
    """A real event stream cut before its first event (empty SSE body, or pings only) is a
    transient failure, not an endpoint that cannot stream: it maps to LLMAPIError and the
    client keeps streaming."""
    fake = FakeAsyncAnthropic()
    fake.messages.stream_empty_sse = 1
    fake.messages.responses.append(make_ok(text="next"))
    audit = RecordingAudit()
    client = _client(fake, stream_transport=True, audit_logger=audit)
    with pytest.raises(LLMAPIError, match="IncompleteStream"):
        await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert "llm_stream_unsupported" not in [e[1] for e in audit.events]
    assert len(fake.messages.captured_requests) == 1  # no plain-JSON fallback request
    # not latched: the next call streams again
    assert (await client.complete_messages(system_blocks=SYSTEM, messages=MSGS)).content == "next"
    assert len(fake.messages.stream_requests) == 2
