"""Unit tests for ``agent_runtime.llm.AnthropicClient``.

16 cases covering: request-shape correctness for both breakpoints, history
pass-through, per-call overrides, defaults, token-usage parsing, cache-write
detection (incl. unknown-model branch), exception wrapping (rate-limit, API
error, malformed response — both empty content and non-text first block).
"""

from __future__ import annotations

import pytest
from anthropic import APIError, RateLimitError

from agent_runtime.llm import (
    AnthropicClient,
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
    Message,
)

from .conftest import RecordingAudit
from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeNonTextBlock,
    FakeUsage,
    make_ok,
)


def _make_rate_limit() -> RateLimitError:
    # Bypass __init__ via __new__ — the SDK's RateLimitError.__init__ takes
    # message + response + body kwargs that change across minor versions.
    # We only need an instance that satisfies ``isinstance(_, RateLimitError)``
    # so the wrapper's ``except RateLimitError`` catches it.
    return RateLimitError.__new__(RateLimitError)


def _make_api_error() -> APIError:
    # Same __new__ bypass rationale as _make_rate_limit.
    return APIError.__new__(APIError)


@pytest.mark.asyncio
async def test_happy_path_two_breakpoints_sent(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(
        static_system_prefix="STATIC",
        retrieval_block="RETRIEVED",
        user_message="hi",
    )
    req = fake_sdk.messages.captured_requests[0]
    assert req["system"] == [
        {"type": "text", "text": "STATIC", "cache_control": {"type": "ephemeral"}}
    ]
    user_msg = req["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == [
        {"type": "text", "text": "RETRIEVED", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "hi"},
    ]


@pytest.mark.asyncio
async def test_no_retrieval_block_single_user_text(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    user_msg = fake_sdk.messages.captured_requests[0]["messages"][-1]
    assert user_msg["content"] == [{"type": "text", "text": "hi"}]


@pytest.mark.asyncio
async def test_no_dynamic_suffix_single_system_block(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert len(fake_sdk.messages.captured_requests[0]["system"]) == 1


@pytest.mark.asyncio
async def test_dynamic_suffix_two_blocks_only_first_cached(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(
        static_system_prefix="STATIC",
        dynamic_system_suffix="DYN",
        user_message="hi",
    )
    blocks = fake_sdk.messages.captured_requests[0]["system"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1]
    assert blocks[1]["text"] == "DYN"


@pytest.mark.asyncio
async def test_history_passed_through_verbatim(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    history: tuple[Message, ...] = (
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    )
    await client.complete(
        static_system_prefix="STATIC", history=history, user_message="hi"
    )
    msgs = fake_sdk.messages.captured_requests[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "earlier-q"}
    assert msgs[1] == {"role": "assistant", "content": "earlier-a"}


@pytest.mark.asyncio
async def test_per_call_model_overrides_default(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok(model="claude-opus-4-7"))
    await client.complete(
        static_system_prefix="STATIC", user_message="hi", model="claude-opus-4-7"
    )
    assert fake_sdk.messages.captured_requests[0]["model"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_per_call_max_tokens_and_temperature(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(
        static_system_prefix="STATIC",
        user_message="hi",
        max_tokens=512,
        temperature=0.7,
    )
    req = fake_sdk.messages.captured_requests[0]
    assert req["max_tokens"] == 512
    assert req["temperature"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_defaults_used_when_call_omits(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    req = fake_sdk.messages.captured_requests[0]
    assert req["model"] == "claude-sonnet-4-6"
    assert req["max_tokens"] == 4096
    assert req["temperature"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_token_usage_parsed_into_claude_response(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(
        make_ok(
            text="response-text",
            input_tokens=1500,
            output_tokens=200,
            cache_creation=1200,
            cache_read=0,
        )
    )
    response = await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert response.content == "response-text"
    assert response.input_tokens == 1500
    assert response.output_tokens == 200
    assert response.cache_creation_input_tokens == 1200
    assert response.cache_read_input_tokens == 0


@pytest.mark.asyncio
async def test_cache_not_written_warning_emitted(
    client: AnthropicClient,
    fake_sdk: FakeAsyncAnthropic,
    audit: RecordingAudit,
) -> None:
    fake_sdk.messages.responses.append(
        make_ok(input_tokens=500, cache_creation=0, cache_read=0)
    )
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    warnings = [e for e in audit.events if e[0] == "warning"]
    assert any(name == "llm_cache_not_written" for _, name, _ in warnings)
    _, _, kwargs = next(e for e in warnings if e[1] == "llm_cache_not_written")
    assert kwargs["threshold_hint"] == 1024
    assert kwargs["model_unknown"] is False


@pytest.mark.asyncio
async def test_cache_warning_silent_when_input_tokens_zero(
    client: AnthropicClient,
    fake_sdk: FakeAsyncAnthropic,
    audit: RecordingAudit,
) -> None:
    fake_sdk.messages.responses.append(
        make_ok(input_tokens=0, cache_creation=0, cache_read=0)
    )
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    names = [name for _, name, _ in audit.events if name == "llm_cache_not_written"]
    assert names == []


@pytest.mark.asyncio
async def test_rate_limit_error_wrapped(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.exceptions.append(_make_rate_limit())
    with pytest.raises(LLMRateLimitError) as excinfo:
        await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert isinstance(excinfo.value.__cause__, RateLimitError)


@pytest.mark.asyncio
async def test_api_error_wrapped(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.exceptions.append(_make_api_error())
    with pytest.raises(LLMAPIError) as excinfo:
        await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert isinstance(excinfo.value.__cause__, APIError)


@pytest.mark.asyncio
async def test_malformed_response_no_content_blocks_raises(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(
        FakeMessage(
            content=[],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=10, output_tokens=0),
        )
    )
    with pytest.raises(LLMResponseError):
        await client.complete(static_system_prefix="STATIC", user_message="hi")


@pytest.mark.asyncio
async def test_malformed_response_non_text_first_block_raises(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    # Server returns a non-text first block (e.g., image/tool_use); wrapper
    # raises LLMResponseError rather than returning empty content.
    fake_sdk.messages.responses.append(
        FakeMessage(
            content=[FakeNonTextBlock()],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=10, output_tokens=0),
        )
    )
    with pytest.raises(LLMResponseError):
        await client.complete(static_system_prefix="STATIC", user_message="hi")


@pytest.mark.asyncio
async def test_unknown_model_warning_uses_default_hint(
    fake_sdk: FakeAsyncAnthropic,
    audit: RecordingAudit,
) -> None:
    # Caller asks for a model name not in _CACHE_MIN_TOKENS — validator should
    # fall back to default hint (1024) and flag model_unknown=True.
    client = AnthropicClient(
        client=fake_sdk,  # type: ignore[arg-type]
        default_model="experimental-mystery-9",
        audit_logger=audit,
    )
    fake_sdk.messages.responses.append(
        make_ok(model="experimental-mystery-9", input_tokens=500, cache_creation=0, cache_read=0)
    )
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    _, _, kwargs = next(e for e in audit.events if e[1] == "llm_cache_not_written")
    assert kwargs["threshold_hint"] == 1024
    assert kwargs["model_unknown"] is True
