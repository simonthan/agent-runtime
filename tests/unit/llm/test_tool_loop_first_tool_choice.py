"""T-7254a -- ToolUseLoop.run(first_tool_choice=...) and AnthropicClient tool_choice.

TBP liveeval 2026-10-04: a guard re-run whose [platform] nudge named the tool still answered
with zero tool calls (`retry_rounds=0`), because every call goes out with the API default
tool_choice (auto). `first_tool_choice` forces the FIRST call of one run; every later call
keeps auto, so a forced choice cannot loop. Default None: byte-for-byte unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, ToolUseLoop
from agent_runtime.llm.tool_loop import ExecuteDecision, ToolResult

from .fakes import FakeAsyncAnthropic, FakeMessage, FakeUsage, make_ok, make_tool_use

TOOLS = [{"name": "search", "input_schema": {}}, {"name": "send", "input_schema": {}}]
FORCE = {"type": "tool", "name": "search"}
NUDGE = "[platform] empty -- answer now."


def _loop(*responses: FakeMessage, nudge: str | None = None) -> tuple[ToolUseLoop, list[dict]]:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    loop = ToolUseLoop(
        client=AnthropicClient(client=sdk),  # type: ignore[arg-type]
        empty_final_nudge=nudge,
    )
    return loop, sdk.messages.captured_requests


async def _ok(_name: str, _inp: dict) -> ToolResult:
    return ToolResult(content="rows")


async def _run(loop: ToolUseLoop, *, max_rounds: int = 3, **kw):
    return await loop.run(
        static_system_prefix="SYS",
        user_message="find it",
        tools=TOOLS,
        executor=_ok,
        max_rounds=max_rounds,
        **kw,
    )


def _choices(requests: list[dict]) -> list[object]:
    return [r.get("tool_choice", "<absent>") for r in requests]


async def test_default_none_sends_no_tool_choice() -> None:
    loop, reqs = _loop(make_tool_use(name="search"), make_ok(text="done"))
    result = await _run(loop)
    assert result.final_text == "done"
    assert _choices(reqs) == ["<absent>", "<absent>"]


async def test_forced_choice_rides_the_first_call_only() -> None:
    loop, reqs = _loop(
        make_tool_use(tool_id="t1", name="search"),
        make_tool_use(tool_id="t2", name="send"),
        make_ok(text="done"),
    )
    result = await _run(loop, first_tool_choice=FORCE)
    assert result.final_text == "done"
    assert [c.name for s in result.steps for c in s.tool_calls] == ["search", "send"]
    assert _choices(reqs) == [FORCE, "<absent>", "<absent>"]
    assert all("tools" in r for r in reqs)


async def test_forced_final_call_carries_no_choice() -> None:
    """max_rounds=1: the forced round, then the no-tools forced-final call (no choice)."""
    loop, reqs = _loop(make_tool_use(name="search"), make_ok(text="answer"))
    result = await _run(loop, max_rounds=1, first_tool_choice=FORCE)
    assert result.final_text == "answer"
    assert _choices(reqs) == [FORCE, "<absent>"]
    assert "tools" not in reqs[1]


async def test_zero_rounds_never_forces() -> None:
    loop, reqs = _loop(make_ok(text="answer"))
    await _run(loop, max_rounds=0, first_tool_choice=FORCE)
    assert _choices(reqs) == ["<absent>"]
    assert "tools" not in reqs[0]


async def test_wrap_up_hook_before_first_call_never_forces() -> None:
    loop, reqs = _loop(make_ok(text="short answer"))
    result = await _run(loop, first_tool_choice=FORCE, pre_completion_hook=lambda: "WRAP UP")
    assert result.final_text == "short answer"
    assert _choices(reqs) == ["<absent>"]
    assert "tools" not in reqs[0]


async def test_empty_reply_recovery_call_carries_no_choice() -> None:
    empty = FakeMessage(
        content=[], model="claude-sonnet-4-6", stop_reason="end_turn", usage=FakeUsage(7, 2)
    )
    loop, reqs = _loop(make_tool_use(name="search"), empty, make_ok(text="ok"), nudge=NUDGE)
    result = await _run(loop, first_tool_choice=FORCE)
    assert result.final_text == "ok"
    assert _choices(reqs) == [FORCE, "<absent>", "<absent>"]


async def test_any_choice_is_passed_without_a_name_check() -> None:
    loop, reqs = _loop(make_tool_use(name="send"), make_ok(text="done"))
    await _run(loop, first_tool_choice={"type": "any"})
    assert _choices(reqs) == [{"type": "any"}, "<absent>"]


async def test_tool_choice_naming_an_absent_tool_raises_before_any_call() -> None:
    loop, reqs = _loop(make_ok(text="never"))
    with pytest.raises(ValueError, match="not in tools"):
        await _run(loop, first_tool_choice={"type": "tool", "name": "recall"})
    assert reqs == []


async def test_resume_never_forces() -> None:
    def confirm_send(name: str, _inp: dict) -> bool:
        return name == "send"

    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.append(make_tool_use(tool_id="t1", name="send"))
    loop = ToolUseLoop(client=AnthropicClient(client=sdk))  # type: ignore[arg-type]
    reqs = sdk.messages.captured_requests
    suspended = await _run(loop, confirm=confirm_send, first_tool_choice={"type": "any"})
    assert suspended.pending_confirmation is not None
    assert _choices(reqs) == [{"type": "any"}]
    reqs_before = len(reqs)
    sdk.messages.responses.append(make_ok(text="sent"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_ok,
        confirm=confirm_send,
        static_system_prefix="SYS",
        max_rounds=3,
    )
    assert result.final_text == "sent"
    assert _choices(reqs[reqs_before:]) == ["<absent>"]


async def test_client_passes_tool_choice_only_with_tools() -> None:
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend([make_ok(), make_ok(), make_ok()])
    client = AnthropicClient(client=sdk)  # type: ignore[arg-type]
    blocks = [{"type": "text", "text": "SYS"}]
    msgs = [{"role": "user", "content": "hi"}]
    await client.complete_messages(system_blocks=blocks, messages=msgs, tools=TOOLS)
    await client.complete_messages(
        system_blocks=blocks, messages=msgs, tools=TOOLS, tool_choice=FORCE
    )
    await client.complete_messages(system_blocks=blocks, messages=msgs, tool_choice=FORCE)
    reqs = sdk.messages.captured_requests
    assert _choices(reqs) == ["<absent>", FORCE, "<absent>"]
    assert "tools" not in reqs[2]
