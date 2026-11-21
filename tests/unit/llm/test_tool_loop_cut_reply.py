"""TBP T-70372 -- a reply cut at max_tokens with tool calls fails its calls back to the model.

Before: `_drive` ended the run on any stop reason but `tool_use`, so a reply that hit the output
cap while writing a tool call was returned as the answer, cut off, and its calls were dropped.
Now that reply is a round in which no call runs: every call gets an `is_error` tool_result
saying so, the model re-issues what it still needs, and the round counts against `max_rounds`.
A `max_tokens` reply with no tool call still ends the run. Always on.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import CUT_REPLY_RESULT_PREFIX, AnthropicClient, ExecuteDecision, ToolUseLoop
from agent_runtime.llm.tool_loop import ToolResult

from .conftest import RecordingAudit
from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
    make_tool_use,
)

TOOLS = [{"name": "search", "input_schema": {}}, {"name": "send", "input_schema": {}}]
NOT_RUN = CUT_REPLY_RESULT_PREFIX


def _cut(*blocks: FakeToolUseBlock, text: str = "Let me look.") -> FakeMessage:
    """A reply that hit the output cap while it was writing its last tool call."""
    content: list[Any] = [FakeTextBlock(text=text)] if text else []
    content.extend(blocks)
    return FakeMessage(
        content=content,
        model="claude-sonnet-4-6",
        stop_reason="max_tokens",
        usage=FakeUsage(input_tokens=300, output_tokens=4096),
    )


def _block(tool_id: str = "tu_cut", name: str = "search", **tool_input: Any) -> FakeToolUseBlock:
    return FakeToolUseBlock(id=tool_id, name=name, input=dict(tool_input))


class _Executor:
    def __init__(self) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        self.ran.append((name, tool_input))
        return ToolResult(content=f"{name} ok")


def _loop(
    *responses: FakeMessage, audit: RecordingAudit | None = None
) -> tuple[ToolUseLoop, FakeAsyncAnthropic]:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    client = AnthropicClient(client=sdk)  # type: ignore[arg-type]
    return ToolUseLoop(client=client, audit_logger=audit), sdk


def _tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool_result blocks of a request's last (user) message."""
    content = request["messages"][-1]["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


async def test_a_cut_call_is_failed_back_and_the_model_answers() -> None:
    loop, sdk = _loop(_cut(_block(q="half")), make_ok(text="Here it is."))
    executor = _Executor()

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find it",
        tools=TOOLS,
        executor=executor,
        max_rounds=4,
    )

    assert executor.ran == []
    assert result.final_text == "Here it is."
    assert result.stop_reason == "end_turn"
    assert result.cap_exhausted is False
    assert result.model_calls == 2
    assert result.output_tokens == 4096 + 50
    # the cut round is a step whose call failed
    (step,) = result.steps
    assert step.assistant_text == "Let me look."
    (call,) = step.tool_calls
    assert (call.id, call.name, call.input, call.is_error) == (
        "tu_cut",
        "search",
        {"q": "half"},
        True,
    )
    assert call.result.startswith(NOT_RUN)
    assert "output token limit" in call.result
    # the second request carries the cut assistant turn as generated, then the failure
    second = sdk.messages.captured_requests[1]
    assistant = second["messages"][-2]
    assert assistant == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look."},
            {"type": "tool_use", "id": "tu_cut", "name": "search", "input": {"q": "half"}},
        ],
    }
    (tr,) = _tool_results(second)
    assert tr["tool_use_id"] == "tu_cut"
    assert tr["is_error"] is True
    assert tr["content"] == call.result
    assert second["tools"]  # the model can re-issue the call


async def test_the_re_issued_call_runs() -> None:
    loop, _ = _loop(
        _cut(_block(q="very long que")),
        make_tool_use(tool_id="tu_2", name="search", tool_input={"q": "short"}),
        make_ok(text="done"),
    )
    executor = _Executor()

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find it",
        tools=TOOLS,
        executor=executor,
        max_rounds=4,
    )

    assert executor.ran == [("search", {"q": "short"})]
    assert result.final_text == "done"
    assert [len(s.tool_calls) for s in result.steps] == [1, 1]
    assert [c.is_error for s in result.steps for c in s.tool_calls] == [True, False]


async def test_no_call_of_a_cut_reply_runs_not_even_a_complete_one() -> None:
    loop, sdk = _loop(
        _cut(_block("tu_a", q="done"), _block("tu_b", "send", to="x")),
        make_ok(text="ok"),
    )
    executor = _Executor()

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=executor,
        max_rounds=4,
    )

    assert executor.ran == []
    trs = _tool_results(sdk.messages.captured_requests[1])
    assert [t["tool_use_id"] for t in trs] == ["tu_a", "tu_b"]
    assert all(t["is_error"] is True and t["content"].startswith(NOT_RUN) for t in trs)
    assert [c.name for c in result.steps[0].tool_calls] == ["search", "send"]


async def test_confirm_is_never_asked_for_a_cut_call() -> None:
    def confirm(_name: str, _input: dict[str, Any]) -> bool:
        msg = "confirm must not be asked for a call that never runs"
        raise AssertionError(msg)

    loop, _ = _loop(_cut(_block("tu_w", "send", to="x")), make_ok(text="ok"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="send it",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
        confirm=confirm,
    )

    assert result.pending_confirmation is None
    assert result.final_text == "ok"


async def test_the_cut_round_counts_against_the_round_cap() -> None:
    loop, sdk = _loop(_cut(_block(q="x")), make_ok(text="what I could do"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find it",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=1,
    )

    assert result.cap_exhausted is True
    assert result.stop_reason == "cap_exhausted"
    assert result.final_text == "what I could do"
    assert len(result.steps) == 1
    # the forced final call offers no tools
    assert "tools" not in sdk.messages.captured_requests[1]


async def test_a_text_only_cut_still_ends_the_run() -> None:
    loop, sdk = _loop(make_ok(text="A long answer that stops mid", stop_reason="max_tokens"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="explain",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assert result.stop_reason == "max_tokens"
    assert result.final_text == "A long answer that stops mid"
    assert result.steps == ()
    assert len(sdk.messages.captured_requests) == 1


async def test_a_text_cut_after_a_round_still_ends_the_run() -> None:
    loop, sdk = _loop(
        make_tool_use(tool_id="tu_1", tool_input={"q": "x"}),
        make_ok(text="The answer is", stop_reason="max_tokens"),
    )

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="find it",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assert result.stop_reason == "max_tokens"
    assert result.final_text == "The answer is"
    assert len(result.steps) == 1
    assert len(sdk.messages.captured_requests) == 2


async def test_the_cut_is_audited() -> None:
    audit = RecordingAudit()
    loop, _ = _loop(
        _cut(_block("tu_a", q="x"), _block("tu_b", "send")), make_ok(text="ok"), audit=audit
    )

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    events = [(lvl, kw) for lvl, msg, kw in audit.events if msg == "tool_loop_cut_tool_calls"]
    assert events == [
        (
            "warning",
            {
                "round_index": 1,
                "max_rounds": 4,
                "calls": 2,
                "tool_names": ["search", "send"],
                "stopped_early": None,
                "output_tokens": 4096,
            },
        )
    ]


async def test_a_non_dict_input_is_echoed_as_an_empty_object() -> None:
    odd = FakeToolUseBlock(id="tu_odd", name="search", input='{"q": "ha')  # type: ignore[arg-type]
    loop, sdk = _loop(_cut(odd), make_ok(text="ok"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assistant = sdk.messages.captured_requests[1]["messages"][-2]
    assert assistant["content"][-1]["input"] == {}
    assert result.steps[0].tool_calls[0].input == {}


async def test_stop_after_round_is_asked_after_the_cut_round() -> None:
    asked: list[int] = []

    def hook() -> str | None:
        asked.append(1)
        return "[platform] done"

    loop, sdk = _loop(_cut(_block(q="x")))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
        stop_after_round=hook,
    )

    assert asked == [1]
    assert result.final_text == "[platform] done"
    assert len(sdk.messages.captured_requests) == 1


async def test_a_cut_after_resume_is_failed_back_too() -> None:
    loop, sdk = _loop(
        make_tool_use(tool_id="tu_w", name="send", tool_input={"to": "x"}),
        _cut(_block("tu_c", q="y")),
        make_ok(text="sent"),
    )
    executor = _Executor()

    def confirm(name: str, _input: dict[str, Any]) -> bool:
        return name == "send"

    first = await loop.run(
        static_system_prefix="SYS",
        user_message="send it",
        tools=TOOLS,
        executor=executor,
        max_rounds=4,
        confirm=confirm,
    )
    assert first.pending_confirmation is not None

    result = await loop.resume(
        state=first.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=executor,
        confirm=confirm,
        static_system_prefix="SYS",
        max_rounds=4,
    )

    assert executor.ran == [("send", {"to": "x"})]
    assert result.final_text == "sent"
    assert [c.is_error for s in result.steps for c in s.tool_calls] == [False, True]
    (tr,) = _tool_results(sdk.messages.captured_requests[2])
    assert tr["tool_use_id"] == "tu_c"
    assert tr["is_error"] is True


def test_the_public_prefix_is_the_marker_s_first_party_start() -> None:
    assert CUT_REPLY_RESULT_PREFIX == "[platform] [NOT RUN BY agent-runtime:"


async def test_a_cut_reply_with_no_text_echoes_only_its_tool_calls() -> None:
    loop, sdk = _loop(_cut(_block(q="x"), text=""), make_ok(text="ok"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assistant = sdk.messages.captured_requests[1]["messages"][-2]
    assert [b["type"] for b in assistant["content"]] == ["tool_use"]
    assert result.steps[0].assistant_text == ""


async def test_a_cut_reply_with_only_blank_text_echoes_no_text_block() -> None:
    # The API rejects a text block that holds only whitespace, which would 400 the
    # follow-up call; the step still records the text as generated (iav review).
    loop, sdk = _loop(_cut(_block(q="x"), text="\n\n"), make_ok(text="ok"))

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assistant = sdk.messages.captured_requests[1]["messages"][-2]
    assert [b["type"] for b in assistant["content"]] == ["tool_use"]
    assert result.steps[0].assistant_text == "\n\n"


async def test_the_cache_marker_moves_onto_the_cut_round() -> None:
    loop, sdk = _loop(
        make_tool_use(tool_id="tu_1", tool_input={"q": "a"}),
        _cut(_block("tu_c1", q="b"), _block("tu_c2", q="c")),
        make_ok(text="ok"),
    )

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
        cache_tool_rounds=True,
    )

    third = sdk.messages.captured_requests[2]["messages"]
    marked = [
        b["tool_use_id"]
        for m in third
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result" and "cache_control" in b
    ]
    assert marked == ["tu_c2"]


async def test_an_empty_reply_after_a_cut_round_gets_the_recovery_call() -> None:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(
        [
            _cut(_block(q="x")),
            make_ok(text="  ", stop_reason="end_turn"),
            make_ok(text="recovered"),
        ]
    )
    client = AnthropicClient(client=sdk)  # type: ignore[arg-type]
    loop = ToolUseLoop(client=client, empty_final_nudge="[platform] answer now")

    result = await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
    )

    assert result.final_text == "recovered"
    assert result.model_calls == 3
    assert len(result.steps) == 1


async def test_a_forced_first_choice_is_not_repeated_after_a_cut() -> None:
    loop, sdk = _loop(_cut(_block(q="x")), make_ok(text="ok"))

    await loop.run(
        static_system_prefix="SYS",
        user_message="go",
        tools=TOOLS,
        executor=_Executor(),
        max_rounds=4,
        first_tool_choice={"type": "tool", "name": "search"},
    )

    first, second = sdk.messages.captured_requests
    assert first["tool_choice"] == {"type": "tool", "name": "search"}
    assert "tool_choice" not in second
