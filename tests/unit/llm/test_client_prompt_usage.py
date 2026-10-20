"""T-7624 (v0.44.0) -- `complete_messages(on_usage=...)`: a streamed call's prompt usage, reported
when the response starts, so a consumer can account for a call it cancels (TBP: a user's Stop)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, PromptUsage
from agent_runtime.llm.streaming import StreamEvent, TextDelta

from .conftest import RecordingAudit
from .fakes import FakeAsyncAnthropic, make_ok

SYSTEM = [{"type": "text", "text": "sys"}]
MSGS = [{"role": "user", "content": "hi"}]


def _ok() -> object:
    return make_ok(text="the answer", input_tokens=120, cache_creation=30, cache_read=900)


async def test_a_streamed_call_reports_its_prompt_usage_before_any_text() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok())
    order: list[object] = []

    async def listener(event: StreamEvent) -> None:
        order.append(event)

    resp = await AnthropicClient(client=fake).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_stream=listener, on_usage=order.append
    )
    assert resp.content == "the answer"
    assert order[0] == PromptUsage(
        input_tokens=120, cache_creation_input_tokens=30, cache_read_input_tokens=900
    )
    assert [type(e) for e in order[1:]] and all(isinstance(e, TextDelta) for e in order[1:])
    assert sum(isinstance(e, PromptUsage) for e in order) == 1


async def test_on_usage_alone_does_not_make_a_call_stream() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok())
    seen: list[PromptUsage] = []
    await AnthropicClient(client=fake).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_usage=seen.append
    )
    assert fake.messages.stream_requests == []  # messages.create, as before
    assert seen == []  # a call that is not streamed never reports


async def test_the_transport_reports_without_a_stream_listener() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok())
    seen: list[PromptUsage] = []
    await AnthropicClient(client=fake, stream_transport=True).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_usage=seen.append
    )
    assert seen == [PromptUsage(120, 30, 900)]
    # the request is unchanged by the callback
    assert "on_usage" not in fake.messages.stream_requests[0]


async def test_a_raising_callback_never_fails_the_call() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok())
    audit = RecordingAudit()

    def broken(_usage: PromptUsage) -> None:
        raise ValueError("consumer bug")

    resp = await AnthropicClient(
        client=fake, stream_transport=True, audit_logger=audit
    ).complete_messages(system_blocks=SYSTEM, messages=MSGS, on_usage=broken)
    assert resp.content == "the answer"
    assert ("llm_usage_listener_failed", "ValueError") in [
        (e[1], e[2].get("error_type")) for e in audit.events
    ]


async def test_reported_usage_does_not_count_as_shown_so_the_retry_stays() -> None:
    """The T-316b retry after the stream started runs only when nothing was handed to the
    stream listener. Usage goes to `on_usage`, so a failure after `message_start` is still
    retried once -- and each attempt reports its own prompt."""
    fake = FakeAsyncAnthropic()
    fake.max_retries = 1
    fake.messages.responses.extend([_ok(), _ok()])
    fake.messages.stream_mid_exceptions.append((1, httpx.ReadTimeout("stalled")))
    seen: list[PromptUsage] = []
    resp = await AnthropicClient(client=fake, stream_transport=True).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, on_usage=seen.append
    )
    assert resp.content == "the answer"
    assert len(fake.messages.stream_requests) == 2
    assert len(seen) == 2


async def test_a_cancelled_call_has_already_reported_its_prompt() -> None:
    """The point of the callback: the consumer cancels a call mid-generation (a user's Stop)
    and still knows what its prompt cost."""
    fake = FakeAsyncAnthropic()
    hold = asyncio.Event()
    fake.messages.responses.append(_ok())
    fake.messages.stream_pause = (2, hold)  # after message_start + content_block_start
    seen: list[PromptUsage] = []
    task = asyncio.create_task(
        AnthropicClient(client=fake, stream_transport=True).complete_messages(
            system_blocks=SYSTEM, messages=MSGS, on_usage=seen.append
        )
    )
    for _ in range(20):
        await asyncio.sleep(0)
        if seen:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [PromptUsage(120, 30, 900)]
