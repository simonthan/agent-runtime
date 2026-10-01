"""T-7219 -- ToolUseLoop empty-reply recovery (`empty_final_nudge`).

A model call that comes back empty after at least one committed tool round gets ONE
recovery call (no tools, the consumer's nudge after the last tool results). Still empty ->
final_text "" + EMPTY_RESPONSE_STOP_REASON with the rounds intact, never an LLMResponseError
that throws the rounds away. Off by default: byte-for-byte unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import (
    EMPTY_RESPONSE_STOP_REASON,
    AnthropicClient,
    ExecuteDecision,
    LLMAPIError,
    LLMResponseError,
    ToolUseLoop,
)
from agent_runtime.llm.tool_loop import ToolResult, _with_trailing_user_text

from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeUsage,
    make_ok,
    make_tool_use,
)

NUDGE = "[platform] Your previous reply was empty. Answer now from the results above."
TOOLS = [{"name": "search", "input_schema": {}}]


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def debug(self, message: str, **kw: object) -> None: ...

    def info(self, message: str, **kw: object) -> None:
        self.events.append(("info", message, kw))

    def warning(self, message: str, **kw: object) -> None:
        self.events.append(("warning", message, kw))

    def error(self, message: str, **kw: object) -> None:
        self.events.append(("error", message, kw))

    def security(self, message: str, **kw: object) -> None: ...

    def action(self, *a: object, **kw: object) -> None: ...

    def named(self, name: str) -> list[tuple[str, dict]]:
        return [(lvl, kw) for lvl, msg, kw in self.events if msg == name]


def _empty(*, stop_reason: str = "end_turn") -> FakeMessage:
    """content=[] -- what the 2026-09-29 incident returned (`response has no content blocks`)."""
    return FakeMessage(
        content=[],
        model="claude-sonnet-4-6",
        stop_reason=stop_reason,
        usage=FakeUsage(input_tokens=7, output_tokens=2),
    )


def _blank_text(*, stop_reason: str = "end_turn") -> FakeMessage:
    return FakeMessage(
        content=[FakeTextBlock(text="  \n")],
        model="claude-sonnet-4-6",
        stop_reason=stop_reason,
        usage=FakeUsage(input_tokens=7, output_tokens=2),
    )


def _loop(*responses: FakeMessage, nudge: str | None = NUDGE):
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    audit = _Audit()
    loop = ToolUseLoop(
        client=AnthropicClient(client=sdk),  # type: ignore[arg-type]
        audit_logger=audit,
        empty_final_nudge=nudge,
    )
    return loop, sdk, audit


async def _ok(_name: str, _inp: dict) -> ToolResult:
    return ToolResult(content="rows")


async def _run(loop: ToolUseLoop, *, max_rounds: int = 5, **kw):
    return await loop.run(
        static_system_prefix="SYS",
        user_message="count transactions",
        tools=TOOLS,
        executor=_ok,
        max_rounds=max_rounds,
        **kw,
    )


def _last_user_blocks(request: dict) -> list[dict]:
    last = request["messages"][-1]
    assert last["role"] == "user"
    return last["content"]


# --- off by default: unchanged ------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_off_empty_after_round_still_raises() -> None:
    loop, sdk, _ = _loop(make_tool_use(), _empty(), nudge=None)
    with pytest.raises(LLMResponseError, match="no content blocks"):
        await _run(loop)
    assert len(sdk.messages.captured_requests) == 2


@pytest.mark.asyncio
async def test_empty_string_nudge_is_off() -> None:
    loop, sdk, _ = _loop(make_tool_use(), _empty(), nudge="")
    with pytest.raises(LLMResponseError):
        await _run(loop)
    assert len(sdk.messages.captured_requests) == 2


@pytest.mark.asyncio
async def test_default_off_blank_text_after_round_returned_as_is() -> None:
    loop, sdk, _ = _loop(make_tool_use(), _blank_text(), nudge=None)
    result = await _run(loop)
    assert result.final_text.strip() == ""
    assert result.stop_reason == "end_turn"
    assert len(sdk.messages.captured_requests) == 2


# --- recovery -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_content_after_round_retries_once_and_returns_the_answer() -> None:
    loop, sdk, audit = _loop(make_tool_use(), _empty(), make_ok(text="12 months: ..."))
    result = await _run(loop)
    assert result.final_text == "12 months: ..."
    assert result.stop_reason == "end_turn"
    assert result.cap_exhausted is False
    assert len(result.steps) == 1
    reqs = sdk.messages.captured_requests
    assert len(reqs) == 3
    assert reqs[1].get("tools")  # the in-loop call that came back empty offered tools
    assert "tools" not in reqs[2]  # the recovery call offers none
    blocks = _last_user_blocks(reqs[2])
    assert blocks[0]["type"] == "tool_result"
    assert blocks[-1] == {"type": "text", "text": NUDGE}
    # the live message list is never mutated: the empty call's request carries no nudge
    assert all(b.get("text") != NUDGE for b in _last_user_blocks(reqs[1]))
    assert audit.named("tool_loop_empty_reply_retry") == [
        ("warning", {"rounds": 1, "max_rounds": 5, "cause": "no_content", "stage": "round"})
    ]
    assert audit.named("tool_loop_empty_reply") == [
        (
            "info",
            {
                "rounds": 1,
                "max_rounds": 5,
                "cause": "no_content",
                "stage": "round",
                "recovered": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_blank_end_turn_after_round_is_recovered_too() -> None:
    loop, sdk, audit = _loop(make_tool_use(), _blank_text(), make_ok(text="answer"))
    result = await _run(loop)
    assert result.final_text == "answer"
    assert len(sdk.messages.captured_requests) == 3
    assert audit.named("tool_loop_empty_reply_retry")[0][1]["cause"] == "empty_text"


@pytest.mark.asyncio
async def test_empty_twice_returns_empty_response_with_steps() -> None:
    loop, sdk, audit = _loop(
        make_tool_use(tool_id="a"), make_tool_use(tool_id="b"), _empty(), _empty()
    )
    result = await _run(loop)
    assert result.final_text == ""
    assert result.stop_reason == EMPTY_RESPONSE_STOP_REASON == "empty_response"
    assert result.cap_exhausted is False
    assert result.pending_confirmation is None
    assert [c.id for s in result.steps for c in s.tool_calls] == ["a", "b"]
    assert len(sdk.messages.captured_requests) == 4  # 2 rounds + empty + ONE retry
    assert audit.named("tool_loop_empty_reply") == [
        (
            "error",
            {
                "rounds": 2,
                "max_rounds": 5,
                "cause": "no_content",
                "stage": "round",
                "recovered": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_retry_blank_text_also_counts_as_empty() -> None:
    loop, _, _ = _loop(make_tool_use(), _empty(), _blank_text())
    result = await _run(loop)
    assert result.final_text == ""
    assert result.stop_reason == EMPTY_RESPONSE_STOP_REASON


@pytest.mark.asyncio
async def test_empty_with_no_round_behind_it_still_raises() -> None:
    loop, sdk, audit = _loop(_empty())
    with pytest.raises(LLMResponseError):
        await _run(loop)
    assert len(sdk.messages.captured_requests) == 1
    assert audit.named("tool_loop_empty_reply_retry") == []


@pytest.mark.asyncio
async def test_blank_text_with_no_round_is_returned_unchanged() -> None:
    loop, sdk, _ = _loop(_blank_text())
    result = await _run(loop)
    assert result.stop_reason == "end_turn"
    assert len(sdk.messages.captured_requests) == 1


@pytest.mark.asyncio
async def test_max_tokens_blank_after_round_is_not_retried() -> None:
    """An output-limit stop is the consumer's (T-311 / T-219b), not an empty reply."""
    loop, sdk, _ = _loop(make_tool_use(), _blank_text(stop_reason="max_tokens"))
    result = await _run(loop)
    assert result.stop_reason == "max_tokens"
    assert len(sdk.messages.captured_requests) == 2


@pytest.mark.asyncio
async def test_wrap_up_forced_final_empty_is_retried_with_wrap_up_and_nudge() -> None:
    """The incident shape: a round, the wrap-up hook fires, the forced final comes back empty."""
    calls = {"n": 0}

    def hook() -> str | None:
        calls["n"] += 1
        return None if calls["n"] == 1 else "WRAP UP NOW"

    loop, sdk, audit = _loop(make_tool_use(), _empty(), make_ok(text="summary"))
    result = await _run(loop, pre_completion_hook=hook)
    assert result.final_text == "summary"
    reqs = sdk.messages.captured_requests
    assert len(reqs) == 3
    assert "tools" not in reqs[1] and "tools" not in reqs[2]
    assert reqs[2]["system"][-1] == {"type": "text", "text": "WRAP UP NOW"}
    assert _last_user_blocks(reqs[2])[-1]["text"] == NUDGE
    retry = audit.named("tool_loop_empty_reply_retry")
    assert retry == [
        ("warning", {"rounds": 1, "max_rounds": 5, "cause": "no_content", "stage": "final"})
    ]


@pytest.mark.asyncio
async def test_in_loop_empty_then_wrap_up_after_break_is_carried() -> None:
    """R3 L4: the hook first fires AFTER the in-loop empty broke the loop -> stage=round, and
    the recovery call still carries the wrap-up block."""
    calls = {"n": 0}

    def hook() -> str | None:
        calls["n"] += 1
        return None if calls["n"] <= 2 else "WRAP UP NOW"

    loop, sdk, audit = _loop(make_tool_use(), _empty(), make_ok(text="ok"))
    result = await _run(loop, pre_completion_hook=hook)
    assert result.final_text == "ok"
    reqs = sdk.messages.captured_requests
    assert reqs[1].get("tools") and "tools" not in reqs[2]
    assert reqs[2]["system"][-1] == {"type": "text", "text": "WRAP UP NOW"}
    assert audit.named("tool_loop_empty_reply_retry")[0][1]["stage"] == "round"


def _markers(request: dict) -> int:
    n = sum(1 for b in request["system"] if isinstance(b, dict) and "cache_control" in b)
    for m in request["messages"]:
        if isinstance(m["content"], list):
            n += sum(1 for b in m["content"] if "cache_control" in b)
    return n


@pytest.mark.asyncio
async def test_recovery_call_keeps_cache_markers_within_limit() -> None:
    """R3 L4: with the T-190 moving marker on, the nudge goes AFTER the marked tool_result and
    the recovery request carries no more cache_control markers than the call it replaces."""
    loop, sdk, _ = _loop(
        make_tool_use(tool_id="a"), make_tool_use(tool_id="b"), _empty(), make_ok(text="ok")
    )
    await loop.run(
        static_system_prefix="SYS",
        user_message="q",
        tools=TOOLS,
        executor=_ok,
        max_rounds=5,
        retrieval_block="RET",
        cache_tool_rounds=True,
    )
    reqs = sdk.messages.captured_requests
    retry_blocks = _last_user_blocks(reqs[3])
    assert retry_blocks[-2]["type"] == "tool_result" and "cache_control" in retry_blocks[-2]
    assert retry_blocks[-1] == {"type": "text", "text": NUDGE}
    assert _markers(reqs[3]) == _markers(reqs[2]) <= 4


@pytest.mark.asyncio
async def test_cap_reached_forced_final_empty_twice_stays_cap_exhausted() -> None:
    loop, sdk, _ = _loop(make_tool_use(), _empty(), _empty())
    result = await _run(loop, max_rounds=1)
    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert result.final_text == ""
    assert len(result.steps) == 1
    assert len(sdk.messages.captured_requests) == 3


@pytest.mark.asyncio
async def test_cap_reached_forced_final_recovered_keeps_cap_flag() -> None:
    loop, _, _ = _loop(make_tool_use(), _blank_text(), make_ok(text="partial answer"))
    result = await _run(loop, max_rounds=1)
    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert result.final_text == "partial answer"


@pytest.mark.asyncio
async def test_retry_api_error_propagates() -> None:
    """Only an EMPTY retry is absorbed; any other LLMError still raises."""
    loop, sdk, _ = _loop(make_tool_use(), _empty())
    real = loop._client.complete_messages
    n = {"calls": 0}

    async def flaky(**kw):
        n["calls"] += 1
        if n["calls"] == 3:
            raise LLMAPIError("upstream 500")
        return await real(**kw)

    loop._client.complete_messages = flaky  # type: ignore[method-assign]
    with pytest.raises(LLMAPIError):
        await _run(loop)
    assert n["calls"] == 3


@pytest.mark.asyncio
async def test_retry_tokens_are_aggregated() -> None:
    loop, _, _ = _loop(
        make_tool_use(input_tokens=100, output_tokens=20),
        _blank_text(),
        make_ok(text="x", input_tokens=300, output_tokens=40),
    )
    result = await _run(loop)
    assert result.input_tokens == 100 + 7 + 300
    assert result.output_tokens == 20 + 2 + 40


@pytest.mark.asyncio
async def test_resume_path_recovers_too() -> None:
    loop, sdk, _ = _loop(
        make_tool_use(tool_id="w", name="send"),
        _empty(),
        make_ok(text="Sent; here is the summary."),
    )
    first = await loop.run(
        static_system_prefix="SYS",
        user_message="send it",
        tools=[{"name": "send", "input_schema": {}}],
        executor=_ok,
        max_rounds=5,
        confirm=lambda name, _inp: name == "send",
    )
    assert first.pending_confirmation is not None
    resumed = await loop.resume(
        state=first.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=[{"name": "send", "input_schema": {}}],
        executor=_ok,
        confirm=lambda name, _inp: name == "send",
        static_system_prefix="SYS",
        max_rounds=5,
    )
    assert resumed.final_text == "Sent; here is the summary."
    assert "tools" not in sdk.messages.captured_requests[-1]


# --- the message helper -------------------------------------------------------------------


def test_with_trailing_user_text_appends_after_tool_results_without_mutating() -> None:
    tail = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}],
    }
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": []}, tail]
    out = _with_trailing_user_text(msgs, NUDGE)
    assert out[-1]["content"][-1] == {"type": "text", "text": NUDGE}
    assert out[-1]["content"][0]["type"] == "tool_result"
    assert len(tail["content"]) == 1 and msgs[-1] is tail  # live list untouched
    assert out[:-1] == msgs[:-1]


def test_with_trailing_user_text_string_and_assistant_tails() -> None:
    out = _with_trailing_user_text([{"role": "user", "content": "q"}], NUDGE)
    assert out[-1]["content"] == [{"type": "text", "text": "q"}, {"type": "text", "text": NUDGE}]
    out = _with_trailing_user_text([{"role": "assistant", "content": "a"}], NUDGE)
    assert out[-1] == {"role": "user", "content": [{"type": "text", "text": NUDGE}]}
