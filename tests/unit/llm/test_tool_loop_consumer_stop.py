"""T-7423a -- ToolUseLoop.run(stop_after_round=...): end a run right after a committed round.

TBP order staging (2026-10-12 replay-harness review): every staging step made one model call
AFTER its terminal sink tool had already recorded the step's result -- model calls = rounds + 1
in every run, the extra call 2.5-3.6 s and 6-11% of model time, its reply discarded. The hook
lets the consumer say "my work is done" after a round commits; the loop then returns with no
further model call (not even the cap's forced final). Default None: byte-for-byte unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import (
    CONSUMER_STOP_REASON,
    AnthropicClient,
    ExecuteDecision,
    StopAfterRoundHook,
    ToolUseLoop,
)
from agent_runtime.llm.tool_loop import ToolResult

from .fakes import FakeAsyncAnthropic, FakeMessage, make_ok, make_tool_use

TOOLS = [{"name": "read", "input_schema": {}}, {"name": "record", "input_schema": {}}]
DONE = "[platform] step recorded"


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


class _Sink:
    """A consumer's terminal tool: `record` sets `done`; the hook reports it."""

    def __init__(self) -> None:
        self.done = False
        self.hook_calls = 0
        self.executed: list[str] = []

    async def execute(self, name: str, _inp: dict) -> ToolResult:
        self.executed.append(name)
        if name == "record":
            self.done = True
        return ToolResult(content="ok")

    def hook(self) -> str | None:
        self.hook_calls += 1
        return DONE if self.done else None


def _loop(
    *responses: FakeMessage, audit: _Audit | None = None
) -> tuple[ToolUseLoop, list[dict], FakeAsyncAnthropic]:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    loop = ToolUseLoop(client=AnthropicClient(client=sdk), audit_logger=audit)  # type: ignore[arg-type]
    return loop, sdk.messages.captured_requests, sdk


async def _run(loop: ToolUseLoop, sink: _Sink, *, max_rounds: int = 5, **kw):
    return await loop.run(
        static_system_prefix="SYS",
        user_message="stage it",
        tools=TOOLS,
        executor=sink.execute,
        max_rounds=max_rounds,
        **kw,
    )


def test_exports() -> None:
    assert CONSUMER_STOP_REASON == "consumer_stop"
    hook: StopAfterRoundHook = lambda: None  # noqa: E731
    assert hook() is None


async def test_default_none_keeps_the_final_call() -> None:
    """Regression guarantee: no hook -> rounds + 1 calls, the model's own final text."""
    sink = _Sink()
    loop, reqs, _ = _loop(
        make_tool_use(tool_id="t1", name="read"),
        make_tool_use(tool_id="t2", name="record"),
        make_ok(text="Recorded."),
    )
    result = await _run(loop, sink)
    assert result.final_text == "Recorded."
    assert result.stop_reason == "end_turn"
    assert len(reqs) == 3
    assert len(result.steps) == 2


async def test_hook_that_never_stops_changes_nothing() -> None:
    sink = _Sink()
    loop, reqs, _ = _loop(make_tool_use(name="read"), make_ok(text="answer"))
    result = await _run(loop, sink, stop_after_round=lambda: None)
    assert result.final_text == "answer"
    assert result.stop_reason == "end_turn"
    assert len(reqs) == 2


async def test_stop_after_the_terminal_round_makes_no_further_call() -> None:
    audit = _Audit()
    sink = _Sink()
    loop, reqs, sdk = _loop(
        make_tool_use(tool_id="t1", name="read", input_tokens=100, output_tokens=20),
        make_tool_use(tool_id="t2", name="record", input_tokens=200, output_tokens=30),
        make_ok(text="never sent"),
        audit=audit,
    )
    result = await _run(loop, sink, stop_after_round=sink.hook)
    assert result.final_text == DONE
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert result.cap_exhausted is False
    assert result.pending_confirmation is None
    assert [c.name for s in result.steps for c in s.tool_calls] == ["read", "record"]
    assert len(reqs) == 2  # one per round, no final call
    assert len(sdk.messages.responses) == 1  # the scripted final reply was never consumed
    assert sink.executed == ["read", "record"]
    assert sink.hook_calls == 2  # once per committed round
    assert (result.input_tokens, result.output_tokens) == (300, 50)
    assert audit.named("tool_loop_consumer_stop") == [("info", {"rounds": 2, "max_rounds": 5})]


async def test_stop_on_the_last_round_the_cap_allows_skips_the_forced_final() -> None:
    audit = _Audit()
    sink = _Sink()
    loop, reqs, _ = _loop(make_tool_use(name="record"), make_ok(text="forced"), audit=audit)
    result = await _run(loop, sink, max_rounds=1, stop_after_round=sink.hook)
    assert result.final_text == DONE
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert result.cap_exhausted is False
    assert len(reqs) == 1
    assert audit.named("tool_loop_cap_exhausted") == []


async def test_cap_without_a_stop_still_forces_the_final_call() -> None:
    sink = _Sink()
    loop, reqs, _ = _loop(make_tool_use(name="read"), make_ok(text="forced"))
    result = await _run(loop, sink, max_rounds=1, stop_after_round=sink.hook)
    assert result.final_text == "forced"
    assert result.cap_exhausted is True
    assert len(reqs) == 2
    assert "tools" not in reqs[1]


async def test_stop_wins_over_the_wrap_up_hook() -> None:
    """The stop hook runs right after the commit, before the next iteration's
    pre_completion_hook check, so a wrap-up never forces a final call past a stop."""
    sink = _Sink()
    wrap_calls: list[int] = []

    def wrap_up() -> str | None:
        wrap_calls.append(1)
        return "[platform] wrap up" if len(wrap_calls) > 1 else None

    loop, reqs, _ = _loop(make_tool_use(name="record"), make_ok(text="wrapped"))
    result = await _run(loop, sink, stop_after_round=sink.hook, pre_completion_hook=wrap_up)
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert len(reqs) == 1
    assert wrap_calls == [1]  # only the pre-round-1 check ran


async def test_hook_is_not_consulted_without_a_committed_round() -> None:
    sink = _Sink()
    sink.done = True  # would stop at once if it were asked
    loop, reqs, _ = _loop(make_ok(text="direct answer"))
    result = await _run(loop, sink, stop_after_round=sink.hook)
    assert result.final_text == "direct answer"
    assert result.stop_reason == "end_turn"
    assert sink.hook_calls == 0
    assert len(reqs) == 1


async def test_empty_string_is_a_stop_with_empty_text() -> None:
    sink = _Sink()
    loop, reqs, _ = _loop(make_tool_use(name="read"), make_ok(text="never"))
    result = await _run(loop, sink, stop_after_round=lambda: "")
    assert result.final_text == ""
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert len(reqs) == 1


async def test_suspended_round_is_not_consulted_and_resume_stops_after_its_commit() -> None:
    def confirm_record(name: str, _inp: dict) -> bool:
        return name == "record"

    sink = _Sink()
    loop, reqs, sdk = _loop(make_tool_use(tool_id="t1", name="record"))
    suspended = await _run(loop, sink, confirm=confirm_record, stop_after_round=sink.hook)
    assert suspended.pending_confirmation is not None
    assert sink.hook_calls == 0  # the round never committed
    assert sink.executed == []
    sdk.messages.responses.append(make_ok(text="never sent"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=sink.execute,
        confirm=confirm_record,
        static_system_prefix="SYS",
        max_rounds=5,
        stop_after_round=sink.hook,
    )
    assert sink.executed == ["record"]
    assert result.final_text == DONE
    assert result.stop_reason == CONSUMER_STOP_REASON
    assert len(reqs) == 1  # resume made no model call
    assert (result.input_tokens, result.output_tokens) == (0, 0)  # D6: continuation-only
    assert len(result.steps) == 1


async def test_resume_without_the_hook_keeps_the_final_call() -> None:
    def confirm_record(name: str, _inp: dict) -> bool:
        return name == "record"

    sink = _Sink()
    loop, reqs, sdk = _loop(make_tool_use(tool_id="t1", name="record"))
    suspended = await _run(loop, sink, confirm=confirm_record)
    assert suspended.pending_confirmation is not None
    sdk.messages.responses.append(make_ok(text="sent"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=sink.execute,
        confirm=confirm_record,
        static_system_prefix="SYS",
        max_rounds=5,
    )
    assert result.final_text == "sent"
    assert len(reqs) == 2
