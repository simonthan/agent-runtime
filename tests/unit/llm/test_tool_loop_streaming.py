"""T-316b -- ToolUseLoop(on_stream=...): every model call of a run is streamed and reported.

The consumer (TBP) shows the answer while it is written. It must know where each model call
starts and ends, whether that call could only be the answer (no tools offered), and when a
call turns out to be a tool round (its text was narration). The ToolLoopResult is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, ExecuteDecision, ToolUseLoop
from agent_runtime.llm.streaming import (
    ModelCallEnded,
    ModelCallStarted,
    StreamEvent,
    TextDelta,
    ToolUseStarted,
)
from agent_runtime.llm.tool_loop import ToolResult

from .fakes import FakeAsyncAnthropic, FakeMessage, FakeTextBlock, FakeUsage, make_ok, make_tool_use

TOOLS = [{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}]


class _Collect:
    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    async def __call__(self, event: StreamEvent) -> None:
        self.events.append(event)

    def shape(self) -> list[Any]:
        """The events with consecutive TextDeltas merged, for compact assertions."""
        out: list[Any] = []
        for e in self.events:
            if isinstance(e, TextDelta) and out and isinstance(out[-1], TextDelta):
                out[-1] = TextDelta(text=out[-1].text + e.text)
            else:
                out.append(e)
        return out


class _Audit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def debug(self, message: str, **kw: object) -> None: ...

    def info(self, message: str, **kw: object) -> None: ...

    def warning(self, message: str, **kw: object) -> None:
        self.events.append((message, kw))

    def error(self, message: str, **kw: object) -> None:
        self.events.append((message, kw))

    def security(self, message: str, **kw: object) -> None: ...


def _loop(
    *responses: FakeMessage, nudge: str | None = None, audit: _Audit | None = None
) -> tuple[ToolUseLoop, FakeAsyncAnthropic]:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    loop = ToolUseLoop(
        client=AnthropicClient(client=sdk),  # type: ignore[arg-type]
        audit_logger=audit,
        empty_final_nudge=nudge,
    )
    return loop, sdk


async def _ok(_name: str, _inp: dict) -> ToolResult:
    return ToolResult(content="rows")


async def _run(loop: ToolUseLoop, **kw: Any):
    kw.setdefault("max_rounds", 5)
    return await loop.run(
        static_system_prefix="SYS",
        user_message="question",
        tools=TOOLS,
        executor=_ok,
        **kw,
    )


async def test_tool_round_then_answer_is_reported_per_call() -> None:
    seen = _Collect()
    loop, sdk = _loop(
        make_tool_use(name="search", text="Let me look."), make_ok(text="Here it is.")
    )
    result = await _run(loop, on_stream=seen)
    assert result.final_text == "Here it is."
    assert seen.shape() == [
        ModelCallStarted(round_index=0, tools_offered=True),
        TextDelta(text="Let me look."),
        ToolUseStarted(name="search"),
        ModelCallEnded(stop_reason="tool_use"),
        ModelCallStarted(round_index=1, tools_offered=True),
        TextDelta(text="Here it is."),
        ModelCallEnded(stop_reason="end_turn"),
    ]
    assert len(sdk.messages.stream_requests) == 2


async def test_result_is_identical_with_and_without_a_listener() -> None:
    script = (make_tool_use(name="search", text="Looking."), make_ok(text="Answer."))
    plain_loop, plain_sdk = _loop(*script)
    streamed_loop, streamed_sdk = _loop(*script)
    plain = await _run(plain_loop)
    streamed = await _run(streamed_loop, on_stream=_Collect())
    assert streamed == plain
    assert streamed_sdk.messages.captured_requests == plain_sdk.messages.captured_requests
    assert plain_sdk.messages.stream_requests == []  # default: no streaming at all


async def test_forced_final_call_is_reported_without_tools() -> None:
    seen = _Collect()
    loop, _ = _loop(make_tool_use(name="search"), make_ok(text="Out of rounds."))
    result = await _run(loop, max_rounds=1, on_stream=seen)
    assert result.cap_exhausted is True
    starts = [e for e in seen.events if isinstance(e, ModelCallStarted)]
    assert starts == [
        ModelCallStarted(round_index=0, tools_offered=True),
        ModelCallStarted(round_index=1, tools_offered=False),
    ]


async def test_empty_reply_recovery_call_is_reported() -> None:
    seen = _Collect()
    empty = FakeMessage(
        content=[],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=1, output_tokens=1),
    )
    loop, _ = _loop(make_tool_use(name="search"), empty, make_ok(text="Recovered."), nudge="N")
    result = await _run(loop, on_stream=seen)
    assert result.final_text == "Recovered."
    # the empty reply raised inside the client (no content blocks) -> Ended(None)
    ends = [e for e in seen.events if isinstance(e, ModelCallEnded)]
    assert ends == [
        ModelCallEnded(stop_reason="tool_use"),
        ModelCallEnded(stop_reason=None),
        ModelCallEnded(stop_reason="end_turn"),
    ]
    assert seen.shape()[-3:] == [
        ModelCallStarted(round_index=1, tools_offered=False),
        TextDelta(text="Recovered."),
        ModelCallEnded(stop_reason="end_turn"),
    ]


async def test_resume_reports_the_continuation() -> None:
    loop, sdk = _loop(make_tool_use(tool_id="w", name="send_email", tool_input={"to": "x"}))
    suspended = await _run(
        loop, confirm=lambda name, _i: name == "send_email", on_stream=_Collect()
    )
    assert suspended.pending_confirmation is not None
    sdk.messages.responses.append(make_ok(text="Sent."))
    seen = _Collect()
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_ok,
        confirm=lambda name, _i: name == "send_email",
        static_system_prefix="SYS",
        max_rounds=5,
        on_stream=seen,
    )
    assert result.final_text == "Sent."
    assert seen.shape() == [
        ModelCallStarted(round_index=1, tools_offered=True),
        TextDelta(text="Sent."),
        ModelCallEnded(stop_reason="end_turn"),
    ]


async def test_raising_listener_is_detached_and_the_run_completes() -> None:
    audit = _Audit()
    calls = 0

    async def broken(_event: StreamEvent) -> None:
        nonlocal calls
        calls += 1
        raise ValueError("bug")

    loop, _ = _loop(make_tool_use(name="search"), make_ok(text="Fine."), audit=audit)
    result = await _run(loop, on_stream=broken)
    assert result.final_text == "Fine."
    assert calls == 1
    assert [name for name, _ in audit.events].count("llm_stream_listener_failed") == 1


async def test_client_double_without_on_stream_keyword_still_works_when_off() -> None:
    """Consumers wrap the client (TBP's timing proxy); without a listener the loop never
    passes `on_stream`, so a wrapper that does not know the keyword is unaffected."""

    class _Double:
        def __init__(self) -> None:
            self.kwargs: list[dict] = []

        async def complete_messages(self, **kwargs: Any):
            self.kwargs.append(kwargs)
            sdk = FakeAsyncAnthropic()
            sdk.messages.responses.append(
                FakeMessage(
                    content=[FakeTextBlock(text="ok")],
                    model="m",
                    stop_reason="end_turn",
                    usage=FakeUsage(input_tokens=1, output_tokens=1),
                )
            )
            return await AnthropicClient(client=sdk).complete_messages(**kwargs)  # type: ignore[arg-type]

    double = _Double()
    loop = ToolUseLoop(client=double)  # type: ignore[arg-type]
    result = await _run(loop)
    assert result.final_text == "ok"
    assert all("on_stream" not in k for k in double.kwargs)
