"""TBP T-7250 -- ToolLoopResult.model_calls: the model calls one run() / resume() made.

Every `complete_messages` call the loop issues counts once: each tool round, the final
answer, a call that raised LLMResponseError (no content) and was absorbed by T-7219
recovery, and the T-7219 recovery call itself. A consumer that derived the count from
`len(steps) + 1` could not see a SUCCESSFUL recovery (the reply carries the retry's own
stop_reason). After resume() the count is continuation-only, like the token counts (D6).
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import (
    CONSUMER_STOP_REASON,
    EMPTY_RESPONSE_STOP_REASON,
    AnthropicClient,
    ExecuteDecision,
    ToolLoopResult,
    ToolUseLoop,
)
from agent_runtime.llm.tool_loop import ToolResult

from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
    make_tool_use,
)

NUDGE = "[platform] Your previous reply was empty. Answer now from the results above."
TOOLS = [{"name": "search", "input_schema": {}}, {"name": "send", "input_schema": {}}]


def _empty() -> FakeMessage:
    return FakeMessage(
        content=[],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=7, output_tokens=2),
    )


def _loop(*responses: FakeMessage, nudge: str | None = NUDGE):
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    loop = ToolUseLoop(
        client=AnthropicClient(client=sdk),  # type: ignore[arg-type]
        empty_final_nudge=nudge,
    )
    return loop, sdk.messages.captured_requests


async def _ok(_name: str, _inp: dict) -> ToolResult:
    return ToolResult(content="rows")


def _confirm_send(name: str, _inp: dict) -> bool:
    return name == "send"


async def _run(loop: ToolUseLoop, *, max_rounds: int = 5, **kw) -> ToolLoopResult:
    return await loop.run(
        static_system_prefix="SYS",
        user_message="count transactions",
        tools=TOOLS,
        executor=_ok,
        max_rounds=max_rounds,
        **kw,
    )


async def _resume(loop: ToolUseLoop, first: ToolLoopResult) -> ToolLoopResult:
    assert first.pending_confirmation is not None
    return await loop.resume(
        state=first.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_ok,
        confirm=_confirm_send,
        static_system_prefix="SYS",
        max_rounds=5,
    )


def test_default_is_none_for_a_result_built_outside_the_loop() -> None:
    result = ToolLoopResult(
        final_text="x",
        stop_reason="end_turn",
        cap_exhausted=False,
        steps=(),
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    assert result.model_calls is None
    # appended last: every earlier field keeps its position
    assert [f.name for f in dataclasses.fields(ToolLoopResult)][-1] == "model_calls"


async def test_plain_answer_is_one_call() -> None:
    loop, reqs = _loop(make_ok(text="hi"))
    result = await _run(loop)
    assert result.model_calls == len(reqs) == 1


async def test_rounds_plus_final() -> None:
    loop, reqs = _loop(make_tool_use(tool_id="a"), make_tool_use(tool_id="b"), make_ok())
    result = await _run(loop)
    assert len(result.steps) == 2
    assert result.model_calls == len(reqs) == 3


async def test_recovered_empty_reply_counts_the_recovery_call() -> None:
    """The bug: a SUCCESSFUL recovery reads as len(steps) + 1 == 2 without this field."""
    loop, reqs = _loop(make_tool_use(), _empty(), make_ok(text="12 months: ..."))
    result = await _run(loop)
    assert result.final_text == "12 months: ..."
    assert result.stop_reason == "end_turn"  # the retry's own stop_reason
    assert len(result.steps) == 1
    assert result.model_calls == len(reqs) == 3


async def test_empty_twice_counts_both_empty_calls() -> None:
    loop, reqs = _loop(make_tool_use(), _empty(), _empty())
    result = await _run(loop)
    assert result.stop_reason == EMPTY_RESPONSE_STOP_REASON
    assert result.model_calls == len(reqs) == 3


async def test_cap_forced_final_recovered() -> None:
    loop, reqs = _loop(make_tool_use(), _empty(), make_ok(text="partial"))
    result = await _run(loop, max_rounds=1)
    assert result.cap_exhausted is True
    assert result.model_calls == len(reqs) == 3


async def test_wrap_up_before_any_call_is_the_forced_final_only() -> None:
    loop, reqs = _loop(make_ok(text="wrapped"))
    result = await _run(loop, pre_completion_hook=lambda: "wrap up now")
    assert result.model_calls == len(reqs) == 1


async def test_consumer_stop_makes_no_final_call() -> None:
    loop, reqs = _loop(make_tool_use(), make_ok(text="never sent"))
    result = await _run(loop, stop_after_round=lambda: "done")
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert result.model_calls == len(reqs) == 1


async def test_suspend_then_resume_is_continuation_only() -> None:
    loop, reqs = _loop(
        make_tool_use(tool_id="r", name="search"),
        make_tool_use(tool_id="w", name="send"),
        _empty(),
        make_ok(text="Sent."),
    )
    first = await _run(loop, confirm=_confirm_send)
    assert first.stop_reason == "pending_confirmation"
    assert first.model_calls == len(reqs) == 2  # the round + the round that hit the card
    # the persisted state is unchanged: only the four D6 counts ride it
    assert first.pending_confirmation is not None
    assert set(first.pending_confirmation.state["agg"]) == {"in", "out", "cc", "cr"}
    resumed = await _resume(loop, first)
    assert resumed.final_text == "Sent."
    assert resumed.model_calls == len(reqs) - 2 == 2  # the empty call + its recovery


async def test_re_suspend_in_the_same_round_makes_no_call() -> None:
    two_sends = FakeMessage(
        content=[
            FakeToolUseBlock(id="w1", name="send", input={}),
            FakeToolUseBlock(id="w2", name="send", input={}),
        ],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=100, output_tokens=20),
    )
    loop, reqs = _loop(two_sends)
    first = await _run(loop, confirm=_confirm_send)
    assert first.model_calls == 1
    again = await _resume(loop, first)
    assert again.stop_reason == "pending_confirmation"
    assert again.model_calls == 0
    assert len(reqs) == 1


async def test_recovery_off_counts_like_before() -> None:
    loop, reqs = _loop(make_tool_use(), make_ok(text="answer"), nudge=None)
    result = await _run(loop)
    assert result.model_calls == len(reqs) == 2
