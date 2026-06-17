"""Unit tests for ``agent_runtime.llm.ToolUseLoop``.

Covers: no-tool answer, single round then answer, multiple tool_use blocks in one
round, cap exhaustion, executor error fed back, and token aggregation.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, ToolUseLoop
from agent_runtime.llm.tool_loop import ToolResult

from .fakes import FakeAsyncAnthropic, FakeMessage, FakeToolUseBlock, FakeUsage, make_ok, make_tool_use


def _make_client(fake_sdk: FakeAsyncAnthropic) -> AnthropicClient:
    return AnthropicClient(client=fake_sdk)  # type: ignore[arg-type]


def _make_loop(fake_sdk: FakeAsyncAnthropic) -> tuple[ToolUseLoop, FakeAsyncAnthropic]:
    client = _make_client(fake_sdk)
    loop = ToolUseLoop(client=client)
    return loop, fake_sdk


async def _never_called(name: str, inp: dict) -> ToolResult:
    """Executor that must not be called — fails the test if it is."""
    msg = f"executor must not be called, got name={name!r}"
    raise AssertionError(msg)


async def _ok_executor(name: str, inp: dict) -> ToolResult:
    return ToolResult(content="hit")


@pytest.mark.asyncio
async def test_no_tool_answer_immediate_return() -> None:
    """Queue a non-tool_use response; executor must not be called."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_ok(text="hi", stop_reason="end_turn"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="hello",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
    )
    assert result.final_text == "hi"
    assert result.steps == ()
    assert result.cap_exhausted is False
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_single_round_then_answer() -> None:
    """One tool_use round then a final text answer."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
    )

    assert result.final_text == "done"
    assert result.cap_exhausted is False
    assert len(result.steps) == 1
    step = result.steps[0]
    assert len(step.tool_calls) == 1
    tc = step.tool_calls[0]
    assert tc.name == "search"
    assert tc.result == "hit"
    assert tc.is_error is False

    # 2nd request must contain a tool_result block with the correct tool_use_id
    second_req = sdk.messages.captured_requests[1]
    user_turn = second_req["messages"][-1]
    assert user_turn["role"] == "user"
    tool_result_blocks = user_turn["content"]
    assert any(
        b.get("type") == "tool_result" and b.get("tool_use_id") == tc.id
        for b in tool_result_blocks
    )


@pytest.mark.asyncio
async def test_multiple_tool_use_blocks_in_one_round() -> None:
    """A response with TWO FakeToolUseBlocks → both executed serially, one step."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)

    # Build a FakeMessage with two tool_use blocks
    two_tool_msg = FakeMessage(
        content=[
            FakeToolUseBlock(id="tu_1", name="search", input={"q": "a"}),
            FakeToolUseBlock(id="tu_2", name="lookup", input={"id": 1}),
        ],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=100, output_tokens=30),
    )
    sdk.messages.responses.append(two_tool_msg)
    sdk.messages.responses.append(make_ok(text="both done"))

    call_log: list[str] = []

    async def tracking_executor(name: str, inp: dict) -> ToolResult:
        call_log.append(name)
        return ToolResult(content=f"result_{name}")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}, {"name": "lookup", "input_schema": {}}],
        executor=tracking_executor,
        max_rounds=3,
    )

    assert len(result.steps) == 1
    step = result.steps[0]
    assert len(step.tool_calls) == 2
    assert call_log == ["search", "lookup"]
    assert step.tool_calls[0].name == "search"
    assert step.tool_calls[1].name == "lookup"
    assert result.final_text == "both done"


@pytest.mark.asyncio
async def test_cap_exhausted() -> None:
    """max_rounds=2: queue 2 tool_use, 1 final ok → cap_exhausted=True, 2 steps, final no tools."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_a"))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_b"))
    sdk.messages.responses.append(make_ok(text="forced"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=2,
    )

    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert len(result.steps) == 2
    assert result.final_text == "forced"

    # The final (3rd) request must have no 'tools' key (forced final call)
    assert "tools" not in sdk.messages.captured_requests[-1]


@pytest.mark.asyncio
async def test_executor_error_fed_back() -> None:
    """Executor returning ToolResult(is_error=True) → tool_result has is_error + ToolCall."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_err", name="search"))
    sdk.messages.responses.append(make_ok(text="recovered"))

    async def error_executor(name: str, inp: dict) -> ToolResult:
        return ToolResult(content="boom", is_error=True)

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=error_executor,
        max_rounds=3,
    )

    assert len(result.steps) == 1
    tc = result.steps[0].tool_calls[0]
    assert tc.is_error is True
    assert tc.result == "boom"

    # The tool_result block in the 2nd request must have is_error=True
    second_req = sdk.messages.captured_requests[1]
    user_turn = second_req["messages"][-1]
    tr_block = next(b for b in user_turn["content"] if b.get("type") == "tool_result")
    assert tr_block["is_error"] is True


@pytest.mark.asyncio
async def test_token_aggregation() -> None:
    """Tokens from all model calls are summed in ToolLoopResult."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(input_tokens=100, output_tokens=20)
    )
    sdk.messages.responses.append(
        make_ok(text="done", input_tokens=150, output_tokens=30)
    )

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
    )

    assert result.input_tokens == 100 + 150
    assert result.output_tokens == 20 + 30
