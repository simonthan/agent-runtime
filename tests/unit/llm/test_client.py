"""Unit tests for ``agent_runtime.llm.AnthropicClient``.

16 cases covering: request-shape correctness for both breakpoints, history
pass-through, per-call overrides, defaults, token-usage parsing, cache-write
detection (incl. unknown-model branch), exception wrapping (rate-limit, API
error, malformed response — both empty content and non-text first block).

T-038a additions (tests 17-23): ``assemble_history_messages`` helper +
``_mark_cache_control`` + ``complete()`` cache_history= param.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from anthropic import APIError, RateLimitError

from agent_runtime.llm import (
    AnthropicClient,
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
    Message,
)
from agent_runtime.llm.client import assemble_history_messages, _mark_cache_control

if TYPE_CHECKING:
    from .conftest import RecordingAudit

from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeNonTextBlock,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
    make_tool_use,
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
    await client.complete(static_system_prefix="STATIC", history=history, user_message="hi")
    msgs = fake_sdk.messages.captured_requests[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "earlier-q"}
    assert msgs[1] == {"role": "assistant", "content": "earlier-a"}


@pytest.mark.asyncio
async def test_per_call_model_overrides_default(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    fake_sdk.messages.responses.append(make_ok(model="claude-opus-4-7"))
    await client.complete(static_system_prefix="STATIC", user_message="hi", model="claude-opus-4-7")
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
    fake_sdk.messages.responses.append(make_ok(input_tokens=500, cache_creation=0, cache_read=0))
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
    fake_sdk.messages.responses.append(make_ok(input_tokens=0, cache_creation=0, cache_read=0))
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
async def test_api_error_wrapped(client: AnthropicClient, fake_sdk: FakeAsyncAnthropic) -> None:
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


# --- §6c additions: tools= param, tool_use parsing, unknown-block warning ---


@pytest.mark.asyncio
async def test_tools_param_puts_tools_in_sdk_request(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """complete(tools=[...]) passes tools in the SDK create call."""
    tools = [{"name": "search", "description": "s", "input_schema": {}}]
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(static_system_prefix="STATIC", user_message="hi", tools=tools)
    assert "tools" in fake_sdk.messages.captured_requests[0]
    assert fake_sdk.messages.captured_requests[0]["tools"] == tools


@pytest.mark.asyncio
async def test_no_tools_omits_tools_key(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """complete(tools=None) must NOT include a 'tools' key in the SDK call (D5)."""
    fake_sdk.messages.responses.append(make_ok())
    await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert "tools" not in fake_sdk.messages.captured_requests[0]


@pytest.mark.asyncio
async def test_tool_use_block_parsed_into_claude_response(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """A tool_use response block is parsed into ClaudeResponse.tool_use."""
    fake_sdk.messages.responses.append(
        make_tool_use(tool_id="tu_1", name="search", tool_input={"q": "hello"})
    )
    resp = await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert len(resp.tool_use) == 1
    tu = resp.tool_use[0]
    assert tu.id == "tu_1"
    assert tu.name == "search"
    assert tu.input == {"q": "hello"}


@pytest.mark.asyncio
async def test_text_and_tool_use_response_concatenates_text(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """A text block + tool_use block: text goes to .content, tool_use to .tool_use."""
    fake_sdk.messages.responses.append(
        FakeMessage(
            content=[
                FakeTextBlock(text="thinking..."),
                FakeToolUseBlock(id="tu_2", name="lookup", input={"id": 99}),
            ],
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=100, output_tokens=20),
        )
    )
    resp = await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert resp.content == "thinking..."
    assert len(resp.tool_use) == 1
    assert resp.tool_use[0].id == "tu_2"


@pytest.mark.asyncio
async def test_unknown_block_type_fires_warning_with_count(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic, audit: RecordingAudit
) -> None:
    """A text block + unknown-type block ('thinking') → text parsed + warning count=1.

    This covers the new unknown-block branch in _parse_response introduced in v0.6.0.
    No existing test asserted this warning (Sonnet F1 / Opus H2).
    """

    class FakeThinkingBlock:
        type = "thinking"

    fake_sdk.messages.responses.append(
        FakeMessage(
            content=[FakeTextBlock(text="answer"), FakeThinkingBlock()],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=100, output_tokens=20),
        )
    )
    resp = await client.complete(static_system_prefix="STATIC", user_message="hi")
    assert resp.content == "answer"
    warnings = [
        e for e in audit.events if e[0] == "warning" and e[1] == "llm_unexpected_extra_blocks"
    ]
    assert len(warnings) == 1
    _, _, kwargs = warnings[0]
    assert kwargs["count"] == 1


# ── T-038a: assemble_history_messages + _mark_cache_control + cache_history= ──


def test_assemble_history_messages_byte_identical_when_flag_false() -> None:
    """cache_history=False → output identical to the bare role/content comprehension."""
    history: tuple[Message, ...] = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    )
    result = assemble_history_messages(history, cache_history=False)
    expected = [{"role": m["role"], "content": m["content"]} for m in history]
    assert result == expected
    # No cache_control key must appear anywhere
    for msg in result:
        assert "cache_control" not in msg
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block


def test_assemble_history_messages_marks_last_message_str_content() -> None:
    """cache_history=True + str content: last message content becomes a text block
    with cache_control; first message is unchanged (still bare str)."""
    history: tuple[Message, ...] = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    )
    result = assemble_history_messages(history, cache_history=True)
    # First message unchanged
    assert result[0]["content"] == "q1"
    assert "cache_control" not in result[0]
    # Last message content is a text block list with cache_control
    last_content = result[-1]["content"]
    assert isinstance(last_content, list)
    assert len(last_content) == 1
    block = last_content[0]
    assert block["type"] == "text"
    assert block["text"] == "a1"
    assert block["cache_control"] == {"type": "ephemeral"}


def test_assemble_history_messages_marks_last_message_list_content() -> None:
    """cache_history=True + block-list content: trailing block gains cache_control;
    original input list is NOT mutated."""
    orig_blocks = [
        {"type": "text", "text": "part1"},
        {"type": "text", "text": "part2"},
    ]
    history: tuple[Message, ...] = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": orig_blocks},  # type: ignore[typeddict-item]
    )
    result = assemble_history_messages(history, cache_history=True)
    last_content = result[-1]["content"]
    assert isinstance(last_content, list)
    # Trailing block has cache_control
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}
    assert last_content[-1]["text"] == "part2"
    # Earlier block unchanged
    assert "cache_control" not in last_content[0]
    # Original list NOT mutated
    assert "cache_control" not in orig_blocks[-1]


def test_assemble_history_messages_empty_history_noop() -> None:
    """Empty history with cache_history=True → [] with no IndexError."""
    result = assemble_history_messages((), cache_history=True)
    assert result == []


def test_mark_cache_control_empty_list_returned_unchanged() -> None:
    """_mark_cache_control([]) → [] (nothing to mark, no error)."""
    result = _mark_cache_control([])
    assert result == []


@pytest.mark.asyncio
async def test_complete_marks_history_when_cache_history_true(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """complete(cache_history=True) with 2-msg history: last history message carries
    cache_control on its trailing block; first history message does not."""
    fake_sdk.messages.responses.append(make_ok())
    history: tuple[Message, ...] = (
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    )
    await client.complete(
        static_system_prefix="STATIC",
        history=history,
        user_message="now",
        cache_history=True,
    )
    msgs = fake_sdk.messages.captured_requests[0]["messages"]
    # messages[-1] is the current user turn; messages[-2] is the last history msg
    last_history_msg = msgs[-2]
    last_history_content = last_history_msg["content"]
    assert isinstance(last_history_content, list)
    assert last_history_content[-1]["cache_control"] == {"type": "ephemeral"}
    # First history message has no cache_control
    first_history_msg = msgs[0]
    assert "cache_control" not in first_history_msg
    first_content = first_history_msg["content"]
    if isinstance(first_content, list):
        for block in first_content:
            assert "cache_control" not in block


@pytest.mark.asyncio
async def test_complete_history_uncached_when_flag_false(
    client: AnthropicClient, fake_sdk: FakeAsyncAnthropic
) -> None:
    """complete(cache_history=False) default: NO history message carries cache_control."""
    fake_sdk.messages.responses.append(make_ok())
    history: tuple[Message, ...] = (
        {"role": "user", "content": "earlier-q"},
        {"role": "assistant", "content": "earlier-a"},
    )
    await client.complete(
        static_system_prefix="STATIC",
        history=history,
        user_message="now",
        retrieval_block="RETRIEVED",
        cache_history=False,
    )
    msgs = fake_sdk.messages.captured_requests[0]["messages"]
    # All history messages (all but the last which is the current user turn)
    for msg in msgs[:-1]:
        assert "cache_control" not in msg
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block
        elif isinstance(content, str):
            pass  # bare str — fine
