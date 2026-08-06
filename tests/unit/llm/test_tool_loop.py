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
