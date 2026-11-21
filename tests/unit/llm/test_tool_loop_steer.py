"""TBP T-70375 -- ToolUseLoop.run(steer=...): a user message sent mid-run joins the run.

TBP runs one turn per (user, bot) and queues the next message behind it, so a correction typed
while a turn works ("actually PO 2") waits until the turn has spent its rounds on the wrong ask.
The hook lets the consumer hand such messages to the run: before each model call that follows a
committed tool round and still offers tools, the loop asks the hook and appends what it returns
as text blocks after that round's tool_results (pi's steering queue, `agent-loop.ts:185-304`).
Default None: byte-for-byte unchanged.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, ExecuteDecision, SteerHook, ToolUseLoop
from agent_runtime.llm.tool_loop import ToolResult

from .conftest import RecordingAudit
from .fakes import FakeAsyncAnthropic, FakeMessage, make_ok, make_tool_use

TOOLS = [{"name": "read", "input_schema": {}}, {"name": "send", "input_schema": {}}]


class _Steer:
    """A consumer inbox: `pending` is handed over (and emptied) each time the loop asks."""

    def __init__(self, *batches: list[str]) -> None:
        self.batches = list(batches)
        self.asked = 0

    def __call__(self) -> list[str]:
        self.asked += 1
        return self.batches.pop(0) if self.batches else []


async def _execute(name: str, _inp: dict[str, Any]) -> ToolResult:
    return ToolResult(content=f"{name} ok")


def _loop(
    *responses: FakeMessage, audit: RecordingAudit | None = None
) -> tuple[ToolUseLoop, list[list[dict[str, Any]]]]:
    """The loop, plus each request's `messages` AS SENT (the loop mutates the live list later)."""
    sdk = FakeAsyncAnthropic()
    sdk.messages.responses.extend(responses)
    sent: list[list[dict[str, Any]]] = []
    create = sdk.messages.create

    async def snapshot_create(**kwargs: Any) -> FakeMessage:
        sent.append(copy.deepcopy(kwargs["messages"]))
        return await create(**kwargs)

    sdk.messages.create = snapshot_create  # type: ignore[method-assign]
    return ToolUseLoop(client=AnthropicClient(client=sdk), audit_logger=audit), sent  # type: ignore[arg-type]


async def _run(loop: ToolUseLoop, *, max_rounds: int = 5, **kw: Any) -> Any:
    return await loop.run(
        static_system_prefix="SYS",
        user_message="find order 1",
        tools=TOOLS,
        executor=_execute,
        max_rounds=max_rounds,
        **kw,
    )


def _steered(audit: RecordingAudit) -> list[dict[str, Any]]:
    return [kw for _lvl, msg, kw in audit.events if msg == "tool_loop_steered"]


def test_exports() -> None:
    hook: SteerHook = lambda: ()  # noqa: E731
    assert list(hook()) == []


async def test_default_none_is_byte_for_byte_unchanged() -> None:
    script = (make_tool_use(tool_id="t1", name="read"), make_ok(text="done"))
    loop_a, sent_a = _loop(*script)
    loop_b, sent_b = _loop(*copy.deepcopy(script))
    a = await _run(loop_a)
    b = await _run(loop_b, steer=None)
    assert sent_a == sent_b
    assert (a.final_text, a.stop_reason) == (b.final_text, b.stop_reason) == ("done", "end_turn")


async def test_a_steered_message_follows_the_round_tool_results() -> None:
    audit = RecordingAudit()
    steer = _Steer(["actually it is order 2"])
    loop, sent = _loop(
        make_tool_use(tool_id="t1", name="read", text="Looking up order 1."),
        make_ok(text="Order 2 is open."),
        audit=audit,
    )
    result = await _run(loop, steer=steer)

    assert result.final_text == "Order 2 is open."
    assert result.stop_reason == "end_turn"
    assert steer.asked == 1  # before the 2nd call only: the 1st call's message is the run's own
    first, second = sent
    assert first[-1]["content"][-1] == {"type": "text", "text": "find order 1"}
    last_user = second[-1]
    assert last_user["role"] == "user"
    assert last_user["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "read ok", "is_error": False},
        {"type": "text", "text": "actually it is order 2"},
    ]
    # the steered text is not a step: steps carry rounds and their calls only
    (step,) = result.steps
    assert [c.id for c in step.tool_calls] == ["t1"]
    assert _steered(audit) == [{"round_index": 1, "max_rounds": 5, "messages": 1, "chars": 22}]


async def test_several_messages_keep_their_order_and_blank_ones_are_dropped() -> None:
    audit = RecordingAudit()
    loop, sent = _loop(make_tool_use(tool_id="t1", name="read"), make_ok(), audit=audit)
    await _run(loop, steer=_Steer(["one", "  ", "", "two"]))
    texts = [b["text"] for b in sent[1][-1]["content"] if b["type"] == "text"]
    assert texts == ["one", "two"]
    assert _steered(audit) == [{"round_index": 1, "max_rounds": 5, "messages": 2, "chars": 6}]


async def test_nothing_pending_changes_nothing() -> None:
    audit = RecordingAudit()
    script = (make_tool_use(tool_id="t1", name="read"), make_ok(text="done"))
    loop_a, sent_a = _loop(*script, audit=audit)
    loop_b, sent_b = _loop(*copy.deepcopy(script))
    steer = _Steer([], ["", " \n"])
    await _run(loop_a, steer=steer)
    await _run(loop_b)
    assert sent_a == sent_b
    assert steer.asked == 1
    assert _steered(audit) == []


async def test_never_asked_when_the_model_answers_at_once() -> None:
    steer = _Steer(["late"])
    loop, sent = _loop(make_ok(text="hi"))
    result = await _run(loop, steer=steer)
    assert result.final_text == "hi"
    assert steer.asked == 0
    assert len(sent) == 1


async def test_never_asked_for_the_forced_final_call() -> None:
    """The cap's forced-final call offers no tools: a correction there could not be acted on,
    so the message stays the consumer's (it can run it as its own turn, with tools)."""
    steer = _Steer(["late"])
    loop, sent = _loop(make_tool_use(tool_id="t1", name="read"), make_ok(text="capped"))
    result = await _run(loop, max_rounds=1, steer=steer)
    assert result.cap_exhausted is True
    assert steer.asked == 0
    assert all(b.get("type") != "text" for b in sent[1][-1]["content"])


async def test_never_asked_after_a_wrap_up() -> None:
    steer = _Steer(["late"])
    wraps = iter([None, "wrap up now"])
    loop, sent = _loop(make_tool_use(tool_id="t1", name="read"), make_ok(text="wrapped"))
    result = await _run(loop, steer=steer, pre_completion_hook=lambda: next(wraps))
    assert result.final_text == "wrapped"
    assert steer.asked == 0
    assert all(b.get("type") != "text" for b in sent[1][-1]["content"])


async def test_never_asked_once_the_consumer_stopped_the_run() -> None:
    steer = _Steer(["late"])
    loop, sent = _loop(make_tool_use(tool_id="t1", name="read"), make_ok(text="never sent"))
    result = await _run(loop, steer=steer, stop_after_round=lambda: "[platform] done")
    assert result.final_text == "[platform] done"
    assert steer.asked == 0
    assert len(sent) == 1


async def test_asked_after_every_round_and_the_text_stays_in_later_rounds() -> None:
    steer = _Steer(["use order 2"], [])
    loop, sent = _loop(
        make_tool_use(tool_id="t1", name="read"),
        make_tool_use(tool_id="t2", name="read"),
        make_ok(text="done"),
    )
    await _run(loop, steer=steer, cache_tool_rounds=True)
    assert steer.asked == 2
    second, third = sent[1], sent[2]
    # request 2: the T-190 marker sits on the round's last tool_result, not on the text after it
    assert second[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert second[-1]["content"][1] == {"type": "text", "text": "use order 2"}
    # request 3: round 1's turn still carries the text (marker moved on to round 2's result)
    round_1_user = third[-3]
    assert round_1_user["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "read ok", "is_error": False},
        {"type": "text", "text": "use order 2"},
    ]
    assert third[-1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t2",
            "content": "read ok",
            "is_error": False,
            "cache_control": {"type": "ephemeral"},
        }
    ]


async def test_a_steered_message_rides_the_suspend_state_and_resume_asks_again() -> None:
    steer = _Steer(["before the card"])
    loop, sent = _loop(
        make_tool_use(tool_id="t1", name="read"),
        make_tool_use(tool_id="t2", name="send", tool_input={"to": "a"}),
        make_ok(text="sent"),
    )
    suspended = await _run(loop, steer=steer, confirm=lambda name, _inp: name == "send")
    card = suspended.pending_confirmation
    assert card is not None
    state = json.loads(json.dumps(card.state))  # the consumer persists it across processes

    resume_steer = _Steer(["and cc Bob"])
    result = await loop.resume(
        state=state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_execute,
        confirm=lambda name, _inp: name == "send",
        static_system_prefix="SYS",
        max_rounds=5,
        steer=resume_steer,
    )
    assert result.final_text == "sent"
    assert resume_steer.asked == 1
    after_resume = sent[-1]
    texts = [
        b["text"]
        for m in after_resume
        if m["role"] == "user" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "text"
    ]
    assert texts == ["find order 1", "before the card", "and cc Bob"]
    assert after_resume[-1]["content"][-1] == {"type": "text", "text": "and cc Bob"}
    assert after_resume[-1]["content"][0]["tool_use_id"] == "t2"


async def test_resume_without_the_hook_never_asks() -> None:
    loop, sent = _loop(
        make_tool_use(tool_id="t1", name="send"),
        make_ok(text="sent"),
    )
    suspended = await _run(loop, confirm=lambda name, _inp: name == "send")
    assert suspended.pending_confirmation is not None
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_execute,
        confirm=lambda name, _inp: name == "send",
        static_system_prefix="SYS",
        max_rounds=5,
    )
    assert result.final_text == "sent"
    assert all(b.get("type") != "text" for b in sent[-1][-1]["content"])


async def test_a_raising_hook_propagates() -> None:
    def boom() -> list[str]:
        raise RuntimeError("inbox broke")

    loop, _ = _loop(make_tool_use(tool_id="t1", name="read"), make_ok())
    with pytest.raises(RuntimeError, match="inbox broke"):
        await _run(loop, steer=boom)


async def test_a_bare_string_is_refused_not_split_into_characters() -> None:
    """R3 F1 -- `str` is a `Sequence[str]`, so a hook returning one passes the type checker; the
    loop refuses it instead of sending one text block per character."""
    loop, _ = _loop(make_tool_use(tool_id="t1", name="read"), make_ok())
    with pytest.raises(TypeError, match="not a str"):
        await _run(loop, steer=lambda: "actually PO 2")  # type: ignore[arg-type,return-value]
