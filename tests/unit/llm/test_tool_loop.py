"""Unit tests for ``agent_runtime.llm.ToolUseLoop``.

Covers: no-tool answer, single round then answer, multiple tool_use blocks in one
round, cap exhaustion, executor error fed back, token aggregation, and
confirm-before-dispatch suspend/resume (T-025a). T-038a: cache_history= flag
on run() + resume() preserves marker from state["messages"].
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, ToolUseLoop, current_tool_round
from agent_runtime.llm.tool_loop import ExecuteDecision, InjectResultDecision, ToolResult

from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
    make_tool_use,
)


def _make_client(fake_sdk: FakeAsyncAnthropic) -> AnthropicClient:
    return AnthropicClient(client=fake_sdk)  # type: ignore[arg-type]


def _make_loop(fake_sdk: FakeAsyncAnthropic) -> tuple[ToolUseLoop, FakeAsyncAnthropic]:
    client = _make_client(fake_sdk)
    loop = ToolUseLoop(client=client)
    return loop, fake_sdk


async def _never_called(name: str, _inp: dict) -> ToolResult:
    """Executor that must not be called — fails the test if it is."""
    msg = f"executor must not be called, got name={name!r}"
    raise AssertionError(msg)


async def _ok_executor(_name: str, _inp: dict) -> ToolResult:
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
        b.get("type") == "tool_result" and b.get("tool_use_id") == tc.id for b in tool_result_blocks
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

    async def tracking_executor(name: str, _inp: dict) -> ToolResult:
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

    async def error_executor(_name: str, _inp: dict) -> ToolResult:
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
    sdk.messages.responses.append(make_tool_use(input_tokens=100, output_tokens=20))
    sdk.messages.responses.append(make_ok(text="done", input_tokens=150, output_tokens=30))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
    )

    assert result.input_tokens == 100 + 150
    assert result.output_tokens == 20 + 30


_CONFIRM_WRITES = lambda name, _inp: name == "send_email"  # noqa: E731 — test predicate


# ---- T-025a: confirm-before-dispatch ---------------------------------------


@pytest.mark.asyncio
async def test_confirm_none_never_suspends() -> None:
    """confirm omitted (None default): a would-be-flagged tool still executes; no suspend."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="send_email", tool_input={"to": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
    )
    assert result.pending_confirmation is None
    assert result.final_text == "done"


@pytest.mark.asyncio
async def test_confirm_suspends_before_dispatch() -> None:
    """A flagged tool suspends: executor NOT called, tokens recorded, only 1 model call."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )
    pc = result.pending_confirmation
    assert pc is not None
    assert pc.tool_name == "send_email"
    assert pc.tool_input == {"to": "x"}
    assert pc.tool_call_id == "tu_w"
    assert result.stop_reason == "pending_confirmation"
    assert result.cap_exhausted is False
    assert result.input_tokens == 100  # suspending round's tokens recorded (D6)
    assert len(sdk.messages.captured_requests) == 1  # no continuation call
    assert pc.state["v"] == 1  # schema-version tag present
    # Dangling-tool_use invariant: the suspending round's assistant turn is NOT yet in
    # messages (it lives in state["round"] until the round completes on resume).
    assert all(
        not (
            msg["role"] == "assistant"
            and isinstance(msg["content"], list)
            and any(b.get("type") == "tool_use" for b in msg["content"])
        )
        for msg in pc.state["messages"]
    )


@pytest.mark.asyncio
async def test_resume_execute_runs_original_input_and_finishes() -> None:
    """resume(ExecuteDecision()) runs the tool with the ORIGINAL input, then answers."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )

    sdk.messages.responses.append(make_ok(text="sent!"))
    sent: list[dict] = []

    async def exec_capture(_name: str, inp: dict) -> ToolResult:
        sent.append(inp)
        return ToolResult(content="ok-sent")

    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=exec_capture,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.pending_confirmation is None
    assert result.final_text == "sent!"
    assert sent == [{"to": "x"}]
    assert len(result.steps) == 1
    assert result.steps[0].tool_calls[0].result == "ok-sent"


@pytest.mark.asyncio
async def test_resume_execute_with_edited_input() -> None:
    """resume(ExecuteDecision(tool_input=...)) runs the tool with the EDITED input."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )

    sdk.messages.responses.append(make_ok(text="sent edited"))
    sent: list[dict] = []

    async def exec_capture(_name: str, inp: dict) -> ToolResult:
        sent.append(inp)
        return ToolResult(content="ok")

    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(tool_input={"to": "y"}),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=exec_capture,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert sent == [{"to": "y"}]
    assert result.steps[0].tool_calls[0].input == {"to": "y"}
    assert result.final_text == "sent edited"


@pytest.mark.asyncio
async def test_resume_inject_result_skips_executor() -> None:
    """resume(InjectResultDecision(...)) feeds a synthetic tool_result, no executor call,
    model reacts (Discard path, D2)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )

    sdk.messages.responses.append(make_ok(text="Okay, I won't send it."))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=InjectResultDecision(content="User declined to send."),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,  # MUST NOT be called
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "Okay, I won't send it."
    assert result.steps[0].tool_calls[0].result == "User declined to send."
    # The continuation request's user turn carries the injected tool_result.
    last_req = sdk.messages.captured_requests[-1]
    user_turn = last_req["messages"][-1]
    tr = next(b for b in user_turn["content"] if b.get("type") == "tool_result")
    assert tr["content"] == "User declined to send."


@pytest.mark.asyncio
async def test_resume_inject_result_error_flag() -> None:
    """InjectResultDecision(is_error=True) flows through to the ToolCall + wire block."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )

    sdk.messages.responses.append(make_ok(text="noted"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=InjectResultDecision(content="blocked by policy", is_error=True),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_never_called,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.steps[0].tool_calls[0].is_error is True
    last_req = sdk.messages.captured_requests[-1]
    tr = next(b for b in last_req["messages"][-1]["content"] if b.get("type") == "tool_result")
    assert tr["is_error"] is True


@pytest.mark.asyncio
async def test_multi_tool_round_executes_reads_suspends_write() -> None:
    """Round = [read(auto), write(confirm)] → read runs, suspend at write; resume completes."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        FakeMessage(
            content=[
                FakeToolUseBlock(id="tu_r", name="search", input={"q": "a"}),
                FakeToolUseBlock(id="tu_w", name="send_email", input={"to": "x"}),
            ],
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=100, output_tokens=30),
        )
    )
    log: list[str] = []

    async def ex(name: str, _inp: dict) -> ToolResult:
        log.append(name)
        return ToolResult(content=f"r_{name}")

    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )
    assert log == ["search"]  # read executed, write held
    assert suspended.pending_confirmation.tool_name == "send_email"

    sdk.messages.responses.append(make_ok(text="all done"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}],
        executor=ex,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert log == ["search", "send_email"]
    assert result.final_text == "all done"
    assert len(result.steps) == 1
    assert [c.name for c in result.steps[0].tool_calls] == ["search", "send_email"]


@pytest.mark.asyncio
async def test_multi_confirm_round_suspends_twice() -> None:
    """Round = [write1, write2] (both confirm) → suspend, resume, suspend again, resume (D5)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        FakeMessage(
            content=[
                FakeToolUseBlock(id="tu_1", name="send_email", input={"n": 1}),
                FakeToolUseBlock(id="tu_2", name="send_email", input={"n": 2}),
            ],
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=100, output_tokens=30),
        )
    )

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult(content="ok")

    tools = [{"name": "send_email", "input_schema": {}}]
    s1 = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=tools,
        executor=ex,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )
    assert s1.pending_confirmation.tool_input == {"n": 1}

    s2 = await loop.resume(
        state=s1.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=tools,
        executor=ex,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert s2.pending_confirmation is not None
    assert s2.pending_confirmation.tool_input == {"n": 2}

    sdk.messages.responses.append(make_ok(text="both sent"))
    result = await loop.resume(
        state=s2.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=tools,
        executor=ex,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "both sent"
    assert [c.input for c in result.steps[0].tool_calls] == [{"n": 1}, {"n": 2}]


@pytest.mark.asyncio
async def test_state_survives_json_round_trip() -> None:
    """The suspend `state` is JSON-serializable; resume works from the round-tripped dict."""
    import json

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )
    round_tripped = json.loads(json.dumps(suspended.pending_confirmation.state))

    sdk.messages.responses.append(make_ok(text="sent!"))
    result = await loop.resume(
        state=round_tripped,
        decision=ExecuteDecision(),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "sent!"
    assert result.steps[0].tool_calls[0].name == "send_email"


@pytest.mark.asyncio
async def test_resume_into_cap_forces_final_call() -> None:
    """Suspend in the last allowed round → resume commits it, hits cap, forced no-tools call."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_w", name="send_email", tool_input={}))
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=1,
        confirm=_CONFIRM_WRITES,
    )

    sdk.messages.responses.append(make_ok(text="forced final"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=1,
    )
    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert result.final_text == "forced final"
    assert "tools" not in sdk.messages.captured_requests[-1]


# ── T-038a: cache_history flag on run() + resume() marker preservation ──────


@pytest.mark.asyncio
async def test_tool_loop_run_marks_history_when_cache_history_true() -> None:
    """run(cache_history=True) with history: first SDK call has cache_control on the
    last history message; the history message before that does not."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    # No tool use — return immediately
    sdk.messages.responses.append(make_ok(text="done"))
    history: tuple[dict, ...] = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    )
    await loop.run(
        static_system_prefix="SYS",
        user_message="now",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        history=history,
        cache_history=True,
    )
    msgs = sdk.messages.captured_requests[0]["messages"]
    # messages[-1] is the current user turn; messages[-2] is the last history msg
    last_history_msg = msgs[-2]
    last_content = last_history_msg["content"]
    assert isinstance(last_content, list)
    assert last_content[-1]["cache_control"] == {"type": "ephemeral"}
    # First history message has no cache_control
    first_history_msg = msgs[0]
    assert "cache_control" not in first_history_msg
    first_content = first_history_msg["content"]
    if isinstance(first_content, list):
        for block in first_content:
            assert "cache_control" not in block


@pytest.mark.asyncio
async def test_tool_loop_resume_preserves_history_marker() -> None:
    """resume() does NOT strip the cache_control marker placed by run() on the last
    history message — the marker rides state['messages'] unchanged."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    # Suspend on a confirm-required write tool
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    history: tuple[dict, ...] = (
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    )
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
        history=history,
        cache_history=True,
    )
    assert suspended.pending_confirmation is not None

    # Verify the marker is in state["messages"] before resume.
    # state_msgs = [user q1, assistant a1 (marked last history msg), current user turn, ...]
    state_msgs = suspended.pending_confirmation.state["messages"]
    last_history_in_state = state_msgs[1]  # index 1 = last history msg (assistant a1)
    content = last_history_in_state["content"]
    assert isinstance(content, list)
    assert content[-1]["cache_control"] == {"type": "ephemeral"}

    # Now resume and check the marker is still present in the continuation call
    sdk.messages.responses.append(make_ok(text="sent!"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "sent!"
    # The continuation call's messages should still carry the marker
    continuation_msgs = sdk.messages.captured_requests[-1]["messages"]
    last_history_in_cont = continuation_msgs[1]
    cont_content = last_history_in_cont["content"]
    assert isinstance(cont_content, list)
    assert cont_content[-1]["cache_control"] == {"type": "ephemeral"}


# ── T-067d: images kwarg on run() (vision passthrough) ──────────────────────


@pytest.mark.asyncio
async def test_run_images_block_order() -> None:
    """retrieval + 2 images + text: block order is
    [retrieval(cache_control), image, image, text]; only retrieval carries
    cache_control."""
    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_ok(text="done"))
    img1 = LLMImage(media_type="image/png", data_b64="AAAA")
    img2 = LLMImage(media_type="image/jpeg", data_b64="BBBB")

    await loop.run(
        static_system_prefix="SYS",
        user_message="what is this",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        retrieval_block="RETRIEVED",
        images=(img1, img2),
    )
    first_user = sdk.messages.captured_requests[0]["messages"][-1]
    assert first_user["role"] == "user"
    content = first_user["content"]
    assert content == [
        {"type": "text", "text": "RETRIEVED", "cache_control": {"type": "ephemeral"}},
        img1.to_block(),
        img2.to_block(),
        {"type": "text", "text": "what is this"},
    ]
    assert "cache_control" not in content[1]
    assert "cache_control" not in content[2]


@pytest.mark.asyncio
async def test_run_images_default_empty_byte_identical() -> None:
    """images=() (default) produces exactly today's first-user structure."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_ok(text="done"))

    await loop.run(
        static_system_prefix="SYS",
        user_message="hello",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        retrieval_block="RETRIEVED",
    )
    first_user = sdk.messages.captured_requests[0]["messages"][-1]
    assert first_user["content"] == [
        {"type": "text", "text": "RETRIEVED", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "hello"},
    ]


@pytest.mark.asyncio
async def test_run_images_with_empty_text_omits_text_block() -> None:
    """user_message="" + one image: no text block. user_message="" + images=():
    the empty text block is still appended (pre-existing behavior preserved)."""
    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_ok(text="done"))
    img = LLMImage(media_type="image/png", data_b64="AAAA")

    await loop.run(
        static_system_prefix="SYS",
        user_message="",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        images=(img,),
    )
    first_user = sdk.messages.captured_requests[0]["messages"][-1]
    assert first_user["content"] == [img.to_block()]

    fake_sdk2 = FakeAsyncAnthropic()
    loop2, sdk2 = _make_loop(fake_sdk2)
    sdk2.messages.responses.append(make_ok(text="done"))
    await loop2.run(
        static_system_prefix="SYS",
        user_message="",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
    )
    first_user2 = sdk2.messages.captured_requests[0]["messages"][-1]
    assert first_user2["content"] == [{"type": "text", "text": ""}]


@pytest.mark.asyncio
async def test_resume_roundtrips_image_blocks() -> None:
    """Drive a confirm-suspending tool turn with one image; state["messages"]
    JSON-round-trips; resume proceeds and the first user message still
    contains the image block."""
    import json

    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )
    img = LLMImage(media_type="image/png", data_b64="AAAA")

    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="describe this",
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
        images=(img,),
    )
    assert suspended.pending_confirmation is not None
    round_tripped = json.loads(json.dumps(suspended.pending_confirmation.state))
    state_first_user = round_tripped["messages"][0]
    assert img.to_block() in state_first_user["content"]

    sdk.messages.responses.append(make_ok(text="sent!"))
    result = await loop.resume(
        state=round_tripped,
        decision=ExecuteDecision(),
        tools=[{"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        confirm=_CONFIRM_WRITES,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "sent!"
    continuation_first_user = sdk.messages.captured_requests[-1]["messages"][0]
    assert img.to_block() in continuation_first_user["content"]


class _CapturingAudit:
    """Minimal AuditLogger that records warning() calls for truncation assertions."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def debug(self, message: str, **kw) -> None: ...
    def info(self, message: str, **kw) -> None: ...
    def warning(self, message: str, **kw) -> None:
        self.warnings.append((message, kw))

    def error(self, message: str, **kw) -> None: ...
    def security(self, message: str, **kw) -> None: ...
    def action(self, *a, **kw) -> None: ...


def _big_executor(text: str):
    async def _run_tool(_name: str, _inp: dict) -> ToolResult:
        return ToolResult(content=text)

    return _run_tool


@pytest.mark.asyncio
async def test_result_under_cap_not_truncated() -> None:
    """Content shorter than the cap passes through byte-identical, no marker."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor("short"),
        max_rounds=3,
        max_result_chars=100,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result == "short"
    assert "TRUNCATED" not in tc.result


@pytest.mark.asyncio
async def test_result_over_cap_truncated_with_marker() -> None:
    """Content longer than the cap → first `cap` chars kept + explicit marker."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor("A" * 500),
        max_rounds=3,
        max_result_chars=100,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result.startswith("A" * 100)
    assert "TRUNCATED BY agent-runtime" in tc.result
    assert "500 characters" in tc.result  # original size reported
    assert "400 characters were removed" in tc.result
    assert tc.is_error is False


@pytest.mark.asyncio
async def test_truncation_marker_carries_platform_provenance_prefix() -> None:
    """T-155: the genuine truncation notice is `[platform]`-framed, so a hostile server
    echoing the literal cannot pass its forged copy off as a first-party notice — the
    forged one loses the prefix at the sanitizer boundary (see the sibling sanitizer
    test), the genuine one is appended post-sanitization and keeps it."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor("A" * 500),
        max_rounds=3,
        max_result_chars=100,
    )
    tc = result.steps[0].tool_calls[0]
    # D2: the prefix sits AFTER the \n\n separator, not before it.
    assert tc.result.startswith("A" * 100)
    assert "\n\n[platform] [TRUNCATED BY agent-runtime:" in tc.result
    assert "\n\n[TRUNCATED BY agent-runtime:" not in tc.result


@pytest.mark.asyncio
async def test_cap_clip_seam_cannot_forge_platform_marker() -> None:
    """T-162 RC1: a clip that cuts `[platformer` short must not resurrect a first-party
    provenance opener. Exactly ONE platform-shaped opener survives — the genuine marker."""
    import re

    from agent_runtime.safety.prompt_sanitizer import sanitize_tool_result

    plat = re.compile(r"\[\s*platform\b", re.IGNORECASE)
    env = sanitize_tool_result("x" * 40 + "[platformer review notes]")
    cap = env.index("[platformer") + len("[platform")

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor(env),
        max_rounds=3,
        max_result_chars=cap,
    )
    tc = result.steps[0].tool_calls[0]
    assert "TRUNCATED BY agent-runtime" in tc.result  # the clip really fired
    assert len(plat.findall(tc.result)) == 1  # only the genuine marker


@pytest.mark.asyncio
async def test_cap_recloses_severed_envelope_and_marker_lands_outside() -> None:
    """T-162 RC2: the truncation notice's provenance is POSITIONAL — it must sit AFTER a
    closed `</tool_output>`, not inside a severed envelope."""
    from agent_runtime.safety.prompt_sanitizer import sanitize_tool_result

    env = sanitize_tool_result("body text " * 20)
    cap = len(env) - 5  # lands mid-`</tool_output>`

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor(env),
        max_rounds=3,
        max_result_chars=cap,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result.count("<tool_output>") == tc.result.count("</tool_output>")
    assert tc.result.index("</tool_output>") < tc.result.index(
        "[platform] [TRUNCATED BY agent-runtime:"
    )


@pytest.mark.asyncio
async def test_both_platform_markers_land_outside_the_reclosed_envelope() -> None:
    """T-162: the images-dropped marker rides the same repair as the truncation marker —
    once the severed envelope is re-closed, BOTH first-party notices sit outside it."""
    from agent_runtime.safety.prompt_sanitizer import sanitize_tool_result

    env = sanitize_tool_result("body text " * 20)
    cap = len(env) - 5  # lands mid-`</tool_output>`
    imgs = (_img(10), _img(20))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult(env, images=imgs)

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_result_chars=cap,
        max_turn_images=1,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == (imgs[0],)  # the image budget really fired
    assert tc.result.count("<tool_output>") == tc.result.count("</tool_output>")
    last_close = tc.result.rindex("</tool_output>")
    assert last_close < tc.result.index("[platform] [TRUNCATED BY agent-runtime:")
    assert last_close < tc.result.index("[platform] [IMAGES WITHHELD BY agent-runtime:")


@pytest.mark.asyncio
async def test_no_cap_default_leaves_huge_result_verbatim() -> None:
    """Regression guarantee: max_result_chars omitted → no truncation at any size."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    huge = "B" * 10_000
    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_big_executor(huge),
        max_rounds=3,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result == huge
    assert "TRUNCATED" not in tc.result


@pytest.mark.asyncio
async def test_truncation_preserves_is_error_and_audits() -> None:
    """is_error survives truncation; a tool_loop_result_truncated warning is emitted."""
    audit = _CapturingAudit()
    fake_sdk = FakeAsyncAnthropic()
    loop = ToolUseLoop(client=_make_client(fake_sdk), audit_logger=audit)  # type: ignore[arg-type]
    fake_sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    fake_sdk.messages.responses.append(make_ok(text="done"))

    async def _err_big(_name: str, _inp: dict) -> ToolResult:
        return ToolResult(content="C" * 500, is_error=True)

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_err_big,
        max_rounds=3,
        max_result_chars=100,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.is_error is True
    assert "TRUNCATED" in tc.result
    assert any(m == "tool_loop_result_truncated" for m, _ in audit.warnings)
    _, kw = next(w for w in audit.warnings if w[0] == "tool_loop_result_truncated")
    assert kw["original_chars"] == 500
    assert kw["cap_chars"] == 100
    assert kw["removed_chars"] == 400
    assert kw["tool_name"] == "search"


@pytest.mark.asyncio
async def test_resume_execute_decision_truncates() -> None:
    """The post-approval resume() executor path is capped too."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    # Round 1: model asks for a confirm-required tool -> loop suspends.
    sdk.messages.responses.append(make_tool_use(name="write", tool_input={"body": "x"}))
    # After resume executes + commits, model answers.
    sdk.messages.responses.append(make_ok(text="ok"))

    def _confirm(name: str, _inp: dict) -> bool:
        return name == "write"

    suspend = await loop.run(
        static_system_prefix="SYS",
        user_message="do it",
        tools=[{"name": "write", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_confirm,
        max_result_chars=100,
    )
    assert suspend.pending_confirmation is not None
    resumed = await loop.resume(
        state=suspend.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "write", "input_schema": {}}],
        executor=_big_executor("D" * 500),
        confirm=_confirm,
        static_system_prefix="SYS",
        max_rounds=3,
        max_result_chars=100,
    )
    # The suspending round's call is in the committed steps of the resumed result.
    all_calls = [c for s in resumed.steps for c in s.tool_calls]
    write_call = next(c for c in all_calls if c.name == "write")
    assert write_call.result.startswith("D" * 100)
    assert "TRUNCATED" in write_call.result


# --- T-115j: round context visible to the executor -------------------------------


@pytest.mark.asyncio
async def test_executor_sees_round_index_and_remaining() -> None:
    """T-115j: each round binds a 1-based index and the remaining tool-round budget."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    for i in range(3):
        sdk.messages.responses.append(make_tool_use(tool_id=f"tu_{i}", name="search"))
    sdk.messages.responses.append(make_ok(text="done"))
    seen: list[tuple[int, int, int]] = []

    async def recording(_name: str, _inp: dict) -> ToolResult:
        ctx = current_tool_round()
        assert ctx is not None
        seen.append((ctx.round_index, ctx.max_rounds, ctx.rounds_remaining))
        return ToolResult("hit")

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=recording,
        max_rounds=5,
    )

    assert seen == [(1, 5, 4), (2, 5, 3), (3, 5, 2)]
    assert current_tool_round() is None  # unbound once the loop returns


@pytest.mark.asyncio
async def test_executor_sees_final_round_on_cap_exhaustion() -> None:
    """rounds_remaining==0 on the LAST round — the round after which tools=None."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search"))
    sdk.messages.responses.append(make_ok(text="forced"))  # the tools=None final call
    seen: list[bool] = []

    async def recording(_name: str, _inp: dict) -> ToolResult:
        ctx = current_tool_round()
        assert ctx is not None
        seen.append(ctx.is_final_round)
        return ToolResult("hit")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=recording,
        max_rounds=1,
    )

    assert seen == [True]
    assert result.cap_exhausted is True


@pytest.mark.asyncio
async def test_resume_binds_the_suspending_round_index() -> None:
    """The resumed round is the SUSPENDING round — state['rounds'] already counts it,
    so resume must not re-increment (tool_loop `_suspend` docstring)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="write", tool_input={"x": 1}))
    sdk.messages.responses.append(make_ok(text="done"))
    seen: list[tuple[int, int]] = []

    async def recording(_name: str, _inp: dict) -> ToolResult:
        ctx = current_tool_round()
        assert ctx is not None
        seen.append((ctx.round_index, ctx.rounds_remaining))
        return ToolResult("written")

    tools = [{"name": "write", "input_schema": {}}]
    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=tools,
        executor=recording,
        max_rounds=4,
        confirm=lambda n, _i: n == "write",
    )
    assert suspended.pending_confirmation is not None
    assert seen == []  # suspended BEFORE dispatch

    await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=tools,
        executor=recording,
        confirm=lambda _n, _i: False,
        static_system_prefix="SYS",
        max_rounds=4,
    )
    assert seen == [(1, 3)]
    assert current_tool_round() is None


@pytest.mark.asyncio
async def test_two_arg_executor_that_ignores_the_context_is_unchanged() -> None:
    """D1 compat contract, pinned: an executor written before T-115j takes exactly two
    positional args, never reads the contextvar, and behaves byte-for-byte as before."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search"))
    sdk.messages.responses.append(make_ok(text="done"))

    async def legacy(name: str, tool_input: dict) -> ToolResult:
        return ToolResult(f"legacy:{name}:{sorted(tool_input)}")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=legacy,
        max_rounds=3,
    )
    assert result.final_text == "done"
    assert result.steps[0].tool_calls[0].result == "legacy:search:[]"


# ── T-118a: tool_result image carry-back ─────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_result_without_images_is_byte_identical() -> None:
    """No images -> committed tool_result content is the bare str, byte-for-byte
    identical to pre-T-118a. This is the regression guarantee."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search"))
    sdk.messages.responses.append(make_ok(text="done"))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok")

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
    )
    block = sdk.messages.captured_requests[1]["messages"][-1]["content"][0]
    assert block == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "ok",
        "is_error": False,
    }
    assert isinstance(block["content"], str)


@pytest.mark.asyncio
async def test_tool_result_with_images_emits_text_then_image_blocks() -> None:
    """With images -> content becomes [text, image...], text-then-images order (D2)."""
    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="render"))
    sdk.messages.responses.append(make_ok(text="done"))
    img = LLMImage(media_type="image/png", data_b64="aGk=")

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("caption", images=(img,))

    await loop.run(
        static_system_prefix="SYS",
        user_message="render it",
        tools=[{"name": "render", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
    )
    block = sdk.messages.captured_requests[1]["messages"][-1]["content"][0]
    assert block["content"] == [
        {"type": "text", "text": "caption"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}},
    ]


@pytest.mark.asyncio
async def test_tool_result_with_images_and_empty_text_omits_text_block() -> None:
    """Empty text + images -> content is an image-only list; the empty text block is
    OMITTED because the Anthropic API rejects an empty text block (D3)."""
    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="render"))
    sdk.messages.responses.append(make_ok(text="done"))
    img = LLMImage(media_type="image/png", data_b64="aGk=")

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("", images=(img,))

    await loop.run(
        static_system_prefix="SYS",
        user_message="render it",
        tools=[{"name": "render", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
    )
    content = sdk.messages.captured_requests[1]["messages"][-1]["content"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image"


@pytest.mark.asyncio
async def test_cap_preserves_images_and_truncates_only_text() -> None:
    """max_result_chars caps TEXT only; images pass through untouched (D4)."""
    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="render"))
    sdk.messages.responses.append(make_ok(text="done"))
    img = LLMImage(media_type="image/png", data_b64="aGk=")

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("x" * 100, images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="render it",
        tools=[{"name": "render", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_result_chars=10,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == (img,)
    assert tc.result.startswith("x" * 10)
    assert "TRUNCATED BY agent-runtime" in tc.result


def test_call_dict_round_trip_preserves_images() -> None:
    """_call_to_dict -> json.dumps -> json.loads -> _call_from_dict yields an equal
    ToolCall including images (proves JSON-safety of PendingConfirmation.state)."""
    import json

    from agent_runtime.llm.models import LLMImage
    from agent_runtime.llm.tool_loop import ToolCall, ToolUseLoop

    img = LLMImage(media_type="image/png", data_b64="aGk=")
    call = ToolCall(
        id="tu_1", name="render", input={"q": "x"}, result="caption", is_error=False, images=(img,)
    )
    d = ToolUseLoop._call_to_dict(call)
    round_tripped = json.loads(json.dumps(d))
    restored = ToolUseLoop._call_from_dict(round_tripped)
    assert restored == call


def test_call_from_dict_without_images_key_defaults_empty() -> None:
    """Deploy safety (D5): a dict with NO 'images' key (pre-T-118a suspended state)
    loads with images == () and does not raise. An image-free ToolCall's
    _call_to_dict emits no 'images' key at all."""
    from agent_runtime.llm.tool_loop import ToolCall, ToolUseLoop

    d = {"id": "tu_1", "name": "search", "input": {}, "result": "ok", "is_error": False}
    restored = ToolUseLoop._call_from_dict(d)
    assert restored.images == ()

    call_no_images = ToolCall(id="tu_1", name="search", input={}, result="ok", is_error=False)
    assert "images" not in ToolUseLoop._call_to_dict(call_no_images)


@pytest.mark.asyncio
async def test_suspend_state_stores_image_bytes_once() -> None:
    """D6-bis: round 1 returns a tool result WITH images and commits; round 2 suspends
    on a confirm-required tool. The base64 payload must appear exactly once in the
    JSON-dumped suspend state (it lives in state['messages']) — state['steps'][0]'s
    tool_calls must carry NO 'images' key. Guards against 2x suspend-state
    amplification."""
    import json

    from agent_runtime.llm.models import LLMImage

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    img = LLMImage(media_type="image/png", data_b64="aGk=")
    sdk.messages.responses.append(make_tool_use(tool_id="tu_r", name="render"))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_w", name="send_email"))

    async def ex(name: str, _inp: dict) -> ToolResult:
        if name == "render":
            return ToolResult("caption", images=(img,))
        return ToolResult("sent")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[
            {"name": "render", "input_schema": {}},
            {"name": "send_email", "input_schema": {}},
        ],
        executor=ex,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
    )
    assert result.pending_confirmation is not None
    state = result.pending_confirmation.state
    dumped = json.dumps(state)
    assert dumped.count("aGk=") == 1
    assert "images" not in state["steps"][0]["tool_calls"][0]


def test_compaction_does_not_leak_tool_result_image_base64() -> None:
    """`_content_to_text` collapses a tool_result block (whose nested content may
    include image base64) to '[tool_result]' — it never inspects nested content, so
    image bytes never reach the compaction merge prompt (guards the compaction.py
    behaviour T-118a depends on)."""
    from agent_runtime.llm.compaction import _content_to_text

    content = [
        {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": [
                {"type": "text", "text": "caption"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
                },
            ],
            "is_error": False,
        }
    ]
    text = _content_to_text(content)
    assert "aGk=" not in text
    assert text == "[tool_result]"


# ── T-135: per-turn tool-result image budget ────────────────────────────────


def _img(nbytes: int, media_type: str = "image/png"):
    from agent_runtime.llm.models import LLMImage  # local — this file's convention

    return LLMImage.from_bytes(b"x" * nbytes, media_type)


@pytest.mark.asyncio
async def test_no_image_budget_default_leaves_images_verbatim() -> None:
    """Regression guarantee: neither ceiling supplied -> images pass through untouched
    no matter how large, and no marker is appended."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    img = _img(9_000_000)

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == (img,)
    assert "IMAGES WITHHELD" not in tc.result


@pytest.mark.asyncio
async def test_image_count_budget_drops_excess_with_marker() -> None:
    """3 images, max_turn_images=1 -> only the FIRST is kept (deterministic order,
    D5); the marker names both the original and dropped counts and the reason."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    imgs = (_img(10), _img(20), _img(30))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=imgs)

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_images=1,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == (imgs[0],)
    assert "3 image(s)" in tc.result
    assert "2 of them" in tc.result
    assert "2 exceeded the 1-image per-turn budget" in tc.result


@pytest.mark.asyncio
async def test_image_byte_budget_drops_excess_with_marker() -> None:
    """2 images of 1000 decoded bytes each, max_turn_image_bytes=1500 -> only the
    first fits; the marker names the byte budget."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    imgs = (_img(1000), _img(1000))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=imgs)

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_image_bytes=1500,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == (imgs[0],)
    assert "1500-byte per-turn budget" in tc.result


@pytest.mark.asyncio
async def test_zero_image_budget_drops_all_images() -> None:
    """D9 guard: max_turn_images=0 means 'no images at all', not 'unset'. A
    truthiness bug (`if max_turn_images:`) would let this image through."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    img = _img(10)

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_images=0,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.images == ()
    assert "IMAGES WITHHELD" in tc.result


@pytest.mark.asyncio
async def test_image_drop_audits() -> None:
    """Dropping images emits a tool_loop_result_images_dropped warning with the
    count/byte breakdown."""
    audit = _CapturingAudit()
    fake_sdk = FakeAsyncAnthropic()
    loop = ToolUseLoop(client=_make_client(fake_sdk), audit_logger=audit)  # type: ignore[arg-type]
    fake_sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    fake_sdk.messages.responses.append(make_ok(text="done"))
    imgs = (_img(10), _img(20), _img(30))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=imgs)

    await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_images=1,
    )
    assert any(m == "tool_loop_result_images_dropped" for m, _ in audit.warnings)
    _, kw = next(w for w in audit.warnings if w[0] == "tool_loop_result_images_dropped")
    assert kw["tool_name"] == "search"
    assert kw["original_images"] == 3
    assert kw["dropped_images"] == 2
    assert kw["dropped_by_count"] == 2
    assert kw["dropped_by_bytes"] == 0


@pytest.mark.asyncio
async def test_all_images_dropped_with_empty_content_yields_marker_string() -> None:
    """D4 invariant: empty text + all images dropped -> content is exactly the marker
    string (non-empty, no leading blank line), and the committed tool_result block's
    `content` is a bare string, not a list (an empty parts list would be rejected)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search"))
    sdk.messages.responses.append(make_ok(text="done"))
    img = _img(100)

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("", images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_images=0,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result.startswith("[platform] [IMAGES WITHHELD")
    assert not tc.result.startswith("\n")
    content = sdk.messages.captured_requests[1]["messages"][-1]["content"][0]["content"]
    assert isinstance(content, str)
    assert content == tc.result


@pytest.mark.asyncio
async def test_text_and_image_caps_compose() -> None:
    """D7: the text cap and the image budget are independent and both apply to the
    same result — final content carries both markers."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="search", tool_input={"q": "x"}))
    sdk.messages.responses.append(make_ok(text="done"))
    img = _img(100)

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("x" * 100, images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find x",
        tools=[{"name": "search", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_result_chars=10,
        max_turn_images=0,
    )
    tc = result.steps[0].tool_calls[0]
    assert tc.result.startswith("x" * 10)
    assert "TRUNCATED BY agent-runtime" in tc.result
    assert "IMAGES WITHHELD" in tc.result
    assert tc.images == ()


# ── T-135: per-turn budget across rounds + suspend/resume ───────────────────


@pytest.mark.asyncio
async def test_image_budget_is_per_turn_across_rounds() -> None:
    """D2: the budget is a PER-TURN total spanning multiple tool rounds, not a
    per-result cap. Round 1 admits its image; round 2's image is entirely dropped."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="render"))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_2", name="render"))
    sdk.messages.responses.append(make_ok(text="done"))
    img1 = _img(10)
    img2 = _img(10)
    seen: list[int] = []

    async def ex(_name: str, _inp: dict) -> ToolResult:
        img = img1 if not seen else img2
        seen.append(1)
        return ToolResult("ok", images=(img,))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "render", "input_schema": {}}],
        executor=ex,
        max_rounds=3,
        max_turn_images=1,
    )
    round1_call = result.steps[0].tool_calls[0]
    round2_call = result.steps[1].tool_calls[0]
    assert round1_call.images == (img1,)
    assert "IMAGES WITHHELD" not in round1_call.result
    assert round2_call.images == ()
    assert "IMAGES WITHHELD" in round2_call.result


@pytest.mark.asyncio
async def test_suspend_state_records_image_budget() -> None:
    """A confirm-gated suspend after one image was admitted stores the running
    images_used counter in state, JSON-safely."""
    import json

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    img = _img(1000)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_r", name="render"))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_w", name="send_email"))

    async def ex(name: str, _inp: dict) -> ToolResult:
        if name == "render":
            return ToolResult("caption", images=(img,))
        return ToolResult("sent")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[
            {"name": "render", "input_schema": {}},
            {"name": "send_email", "input_schema": {}},
        ],
        executor=ex,
        max_rounds=3,
        confirm=_CONFIRM_WRITES,
        max_turn_images=5,
    )
    assert result.pending_confirmation is not None
    state = result.pending_confirmation.state
    assert state["images_used"] == {"count": 1, "bytes": 1000}
    assert json.dumps(state)  # JSON-safety of the whole blob


@pytest.mark.asyncio
async def test_resume_carries_image_budget_forward() -> None:
    """D3, the highest-value test: the per-turn budget SURVIVES the suspend/resume
    boundary. If resume() reset the counter (like it does for `agg`), this second
    image would wrongly be admitted — that is the precise bug D3 warns about."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    img1 = _img(10)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="render"))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_2", name="send_email"))

    def _confirm(name: str, _inp: dict) -> bool:
        return name == "send_email"

    async def ex1(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("caption", images=(img1,))

    suspend = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[
            {"name": "render", "input_schema": {}},
            {"name": "send_email", "input_schema": {}},
        ],
        executor=ex1,
        max_rounds=3,
        confirm=_confirm,
        max_turn_images=1,
    )
    assert suspend.pending_confirmation is not None
    assert suspend.pending_confirmation.state["images_used"] == {"count": 1, "bytes": 10}

    img2 = _img(10)
    sdk.messages.responses.append(make_ok(text="done"))

    async def ex2(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("sent", images=(img2,))

    result = await loop.resume(
        state=suspend.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[
            {"name": "render", "input_schema": {}},
            {"name": "send_email", "input_schema": {}},
        ],
        executor=ex2,
        confirm=_confirm,
        static_system_prefix="SYS",
        max_rounds=3,
        max_turn_images=1,
    )
    all_calls = [c for s in result.steps for c in s.tool_calls]
    send_call = next(c for c in all_calls if c.name == "send_email")
    assert send_call.images == ()
    assert "IMAGES WITHHELD" in send_call.result


@pytest.mark.asyncio
async def test_resume_without_images_used_key_starts_fresh_budget() -> None:
    """Deploy safety: a suspend state persisted before T-135 has no 'images_used'
    key. resume() must not KeyError and must start with a fresh budget (mirrors
    test_call_from_dict_without_images_key_defaults_empty)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(name="write", tool_input={"body": "x"}))

    def _confirm(name: str, _inp: dict) -> bool:
        return name == "write"

    suspend = await loop.run(
        static_system_prefix="SYS",
        user_message="do it",
        tools=[{"name": "write", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        confirm=_confirm,
    )
    assert suspend.pending_confirmation is not None
    state = suspend.pending_confirmation.state
    del state["images_used"]

    img = _img(10)
    sdk.messages.responses.append(make_ok(text="done"))

    async def ex(_name: str, _inp: dict) -> ToolResult:
        return ToolResult("ok", images=(img,))

    result = await loop.resume(
        state=state,
        decision=ExecuteDecision(),
        tools=[{"name": "write", "input_schema": {}}],
        executor=ex,
        confirm=_confirm,
        static_system_prefix="SYS",
        max_rounds=3,
        max_turn_images=1,
        max_turn_image_bytes=1000,
    )
    all_calls = [c for s in result.steps for c in s.tool_calls]
    write_call = next(c for c in all_calls if c.name == "write")
    assert write_call.images == (img,)
    assert "IMAGES WITHHELD" not in write_call.result


# ---- T-190: cache_tool_rounds moving marker --------------------------------


@pytest.mark.asyncio
async def test_cache_tool_rounds_marks_last_tool_result() -> None:
    """2-round run with cache_tool_rounds=True:
    - request 2 (round 1 committed): round 1's user message has cache_control on
      its LAST tool_result block and nowhere else among tool_result blocks.
    - request 3 (final): round 1's marker stripped; round 2's last tool_result marked.
    """
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    # Round 1: tool_use -> round 2: tool_use -> final answer
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_1", name="search", tool_input={"q": "a"})
    )
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_2", name="search", tool_input={"q": "b"})
    )
    sdk.messages.responses.append(make_ok(text="done"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=5,
        cache_tool_rounds=True,
    )
    assert result.final_text == "done"
    reqs = sdk.messages.captured_requests

    # request index 1 = after round 1 committed; last user msg = round 1's results
    req2 = reqs[1]
    user_turn_r1 = req2["messages"][-1]
    assert user_turn_r1["role"] == "user"
    tr_blocks_r1 = [b for b in user_turn_r1["content"] if b.get("type") == "tool_result"]
    # last block must be marked; no other tool_result block in the whole request is marked
    assert tr_blocks_r1[-1].get("cache_control") == {"type": "ephemeral"}
    all_tr_in_req2 = [
        b
        for m in req2["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    marked_in_req2 = [b for b in all_tr_in_req2 if "cache_control" in b]
    assert len(marked_in_req2) == 1

    # request index 2 = final no-tools call; round 1's marker stripped, round 2's marked
    req3 = reqs[2]
    all_tr_in_req3 = [
        b
        for m in req3["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    marked_in_req3 = [b for b in all_tr_in_req3 if "cache_control" in b]
    assert len(marked_in_req3) == 1
    # The marked block must be in the LAST user message (round 2's results)
    last_user_msg_r2 = req3["messages"][-1]
    tr_in_last_r2 = [b for b in last_user_msg_r2["content"] if b.get("type") == "tool_result"]
    assert tr_in_last_r2[-1].get("cache_control") == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_cache_tool_rounds_default_off_byte_identical() -> None:
    """Default (cache_tool_rounds=False): no tool_result block ever carries cache_control."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search", tool_input={}))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_2", name="search", tool_input={}))
    sdk.messages.responses.append(make_ok(text="done"))

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=5,
    )
    for req in sdk.messages.captured_requests:
        for m in req["messages"]:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if b.get("type") == "tool_result":
                    assert "cache_control" not in b, f"unexpected marker on {b}"


@pytest.mark.asyncio
async def test_cache_tool_rounds_marker_count_at_limit() -> None:
    """With retrieval_block + cache_history + non-empty history + cache_tool_rounds + 2 rounds:
    every captured request must have ≤4 cache_control markers total."""

    def _count_markers(req: dict) -> int:
        count = 0
        for block in req.get("system", []):
            if "cache_control" in block:
                count += 1
        for m in req.get("messages", []):
            content = m.get("content")
            if isinstance(content, list):
                for b in content:
                    if "cache_control" in b:
                        count += 1
            elif isinstance(content, str):
                pass
        return count

    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search", tool_input={}))
    sdk.messages.responses.append(make_tool_use(tool_id="tu_2", name="search", tool_input={}))
    sdk.messages.responses.append(make_ok(text="done"))

    history_msg: dict = {
        "role": "user",
        "content": [{"type": "text", "text": "prior turn"}],
    }

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=5,
        retrieval_block="RETRIEVED",
        history=(history_msg,),
        cache_history=True,
        cache_tool_rounds=True,
    )
    for req in sdk.messages.captured_requests:
        count = _count_markers(req)
        assert count <= 4, f"marker count={count} exceeds limit 4 in request"


@pytest.mark.asyncio
async def test_cache_tool_rounds_resume_strips_state_marker() -> None:
    """Suspend after round 1 committed with marker; JSON round-trip state; resume with
    cache_tool_rounds=True: exactly one moving marker in resumed request, on that round's
    last tool_result."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)

    # Round 1: two tool uses, but second triggers confirm → round committed first tool,
    # suspends on second.  We actually want round 1 to COMMIT fully (with the marker),
    # then suspend on round 2.  Use two separate rounds instead: round 1 commits (search),
    # round 2 triggers confirm on send_email.
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search", tool_input={}))
    sdk.messages.responses.append(
        make_tool_use(tool_id="tu_w", name="send_email", tool_input={"to": "x"})
    )

    _confirm_email = lambda name, _inp: name == "send_email"  # noqa: E731

    suspended = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=5,
        confirm=_confirm_email,
        cache_tool_rounds=True,
    )
    assert suspended.pending_confirmation is not None

    import json

    # JSON round-trip state to simulate persistence
    state_json = json.dumps(suspended.pending_confirmation.state)
    state = json.loads(state_json)

    sdk.messages.responses.append(make_ok(text="sent"))

    result = await loop.resume(
        state=state,
        decision=ExecuteDecision(),
        tools=[{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}],
        executor=_ok_executor,
        confirm=_confirm_email,
        static_system_prefix="SYS",
        max_rounds=5,
        cache_tool_rounds=True,
    )
    assert result.final_text == "sent"

    # The resumed request: after the resumed round commits (send_email), the final call
    # carries exactly one moving marker.
    final_req = sdk.messages.captured_requests[-1]
    all_tr_blocks = [
        b
        for m in final_req["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    marked = [b for b in all_tr_blocks if "cache_control" in b]
    assert len(marked) == 1


@pytest.mark.asyncio
async def test_single_tool_result_marker_placement() -> None:
    """1-round run: final no-tools request has the marker on that round's last tool_result."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search", tool_input={}))
    sdk.messages.responses.append(make_ok(text="answer"))

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        cache_tool_rounds=True,
    )
    # final (no-tools) request
    final_req = sdk.messages.captured_requests[-1]
    last_user = final_req["messages"][-1]
    assert last_user["role"] == "user"
    tr_blocks = [b for b in last_user["content"] if b.get("type") == "tool_result"]
    assert len(tr_blocks) == 1
    assert tr_blocks[-1].get("cache_control") == {"type": "ephemeral"}


# ── T-284: pre_completion_hook ────────────────────────────────────────────────


class _CapturingAuditLogger:
    """Minimal AuditLogger that records (level, message, kwargs) tuples."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def debug(self, message: str, **kwargs: object) -> None:
        self.events.append(("debug", message, kwargs))

    def info(self, message: str, **kwargs: object) -> None:
        self.events.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs: object) -> None:
        self.events.append(("warning", message, kwargs))

    def error(self, message: str, **kwargs: object) -> None:
        self.events.append(("error", message, kwargs))

    def security(self, message: str, **kwargs: object) -> None:
        self.events.append(("security", message, kwargs))

    def action(self, action: str, result: str, **kwargs: object) -> None:
        self.events.append(("action", action, kwargs))


@pytest.mark.asyncio
async def test_pre_completion_hook_returns_none_no_change() -> None:
    """Hook always returns None → loop behaves identically to no-hook (1 tool round + answer)."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search"))
    sdk.messages.responses.append(make_ok(text="done"))

    call_count = 0

    def hook() -> str | None:
        nonlocal call_count
        call_count += 1
        return None

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=3,
        pre_completion_hook=hook,
    )

    assert result.cap_exhausted is False
    assert result.stop_reason == "end_turn"
    assert len(result.steps) == 1
    assert result.final_text == "done"
    # Hook was called at least once (before the first round)
    assert call_count >= 1


@pytest.mark.asyncio
async def test_pre_completion_hook_fires_before_first_round() -> None:
    """Hook returns text on first call → no tool rounds, forced-final carries injected text."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    # Only queue the forced-final response — no tool rounds should execute
    sdk.messages.responses.append(make_ok(text="wrapped up"))

    _wrap_up_text = "Please wrap up your response now."

    def hook() -> str | None:
        return _wrap_up_text

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_never_called,
        max_rounds=3,
        pre_completion_hook=hook,
    )

    assert result.cap_exhausted is False
    assert result.stop_reason == "end_turn"
    assert result.final_text == "wrapped up"
    # Exactly 1 SDK call (the forced-final, no tool rounds)
    assert len(sdk.messages.captured_requests) == 1
    # The forced-final carries the injected text in system
    final_req = sdk.messages.captured_requests[0]
    assert any(
        b.get("type") == "text" and b.get("text") == _wrap_up_text for b in final_req["system"]
    )
    # No tools in forced-final call
    assert "tools" not in final_req


@pytest.mark.asyncio
async def test_pre_completion_hook_fires_between_rounds() -> None:
    """Hook returns None first, then text: only 1 tool round executes, forced-final injected."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_tool_use(tool_id="tu_1", name="search"))
    sdk.messages.responses.append(make_ok(text="wrapped after 1 round"))

    call_count = 0
    _wrap_up_text = "Time to wrap up."

    def hook() -> str | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None  # Let first round proceed
        return _wrap_up_text

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=5,
        pre_completion_hook=hook,
    )

    assert result.cap_exhausted is False
    assert result.stop_reason == "end_turn"
    assert len(result.steps) == 1  # Only 1 tool round completed
    assert result.final_text == "wrapped after 1 round"
    # Forced-final carries the injected text in system
    final_req = sdk.messages.captured_requests[-1]
    assert any(
        b.get("type") == "text" and b.get("text") == _wrap_up_text for b in final_req["system"]
    )
    assert "tools" not in final_req


@pytest.mark.asyncio
async def test_pre_completion_hook_fires_at_cap_exhaustion() -> None:
    """Hook returns None inside loop; AFTER cap exhaustion returns text.
    Both tool_loop_wrap_up_injected (INFO) and tool_loop_cap_exhausted (WARNING) fire."""
    fake_sdk = FakeAsyncAnthropic()
    audit = _CapturingAuditLogger()
    client = _make_client(fake_sdk)
    loop = ToolUseLoop(client=client, audit_logger=audit)

    fake_sdk.messages.responses.append(make_tool_use(tool_id="tu_a"))
    fake_sdk.messages.responses.append(make_tool_use(tool_id="tu_b"))
    fake_sdk.messages.responses.append(make_ok(text="forced final"))

    call_count = 0
    _wrap_up_text = "Summarise what you found so far."

    def hook() -> str | None:
        nonlocal call_count
        call_count += 1
        # max_rounds=2 → hook is called twice inside loop (before each LLM call),
        # then once after the loop exits normally (cap reached)
        if call_count <= 2:
            return None
        return _wrap_up_text

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_ok_executor,
        max_rounds=2,
        pre_completion_hook=hook,
    )

    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert result.final_text == "forced final"

    # Both audit events must fire
    info_events = [
        e for e in audit.events if e[0] == "info" and e[1] == "tool_loop_wrap_up_injected"
    ]
    warn_events = [
        e for e in audit.events if e[0] == "warning" and e[1] == "tool_loop_cap_exhausted"
    ]
    assert len(info_events) == 1, f"expected 1 info wrap_up event, got {info_events}"
    assert len(warn_events) == 1, f"expected 1 warning cap_exhausted event, got {warn_events}"

    # Forced-final carries the injected text in system
    final_req = fake_sdk.messages.captured_requests[-1]
    assert any(
        b.get("type") == "text" and b.get("text") == _wrap_up_text for b in final_req["system"]
    )


@pytest.mark.asyncio
async def test_pre_completion_hook_at_max_rounds_zero() -> None:
    """max_rounds=0: while loop skipped, hook fires at post-loop check, wrap-up injected."""
    fake_sdk = FakeAsyncAnthropic()
    loop, sdk = _make_loop(fake_sdk)
    sdk.messages.responses.append(make_ok(text="immediate wrap"))

    _wrap_up_text = "Zero rounds — wrap up immediately."

    def hook() -> str | None:
        return _wrap_up_text

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=[{"name": "search", "input_schema": {}}],
        executor=_never_called,
        max_rounds=0,
        pre_completion_hook=hook,
    )

    assert result.cap_exhausted is True
    assert result.final_text == "immediate wrap"
    # Only 1 SDK call (the forced-final)
    assert len(sdk.messages.captured_requests) == 1
    final_req = sdk.messages.captured_requests[0]
    assert any(
        b.get("type") == "text" and b.get("text") == _wrap_up_text for b in final_req["system"]
    )
    assert "tools" not in final_req
