"""v0.45.0 (TBP T-7683) -- an opt-in 1-hour TTL on breakpoint 1 (the static system prefix),
and the cache write split by TTL (`usage.cache_creation`) on every usage surface.

Default (`prefix_cache_ttl=None`) must leave every request byte-for-byte unchanged; with
"1h" only breakpoint 1 carries `ttl`, every other marker stays 5 minutes (the API requires
a longer TTL to come before any shorter one)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient, PromptUsage, ToolUseLoop
from agent_runtime.llm.tool_loop import ExecuteDecision, ToolResult

from .conftest import RecordingAudit
from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
)

FIVE_MIN = {"type": "ephemeral"}
ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}
TOOLS = [{"name": "search", "input_schema": {}}, {"name": "send_email", "input_schema": {}}]


@dataclass(frozen=True)
class _Split:
    ephemeral_1h_input_tokens: int
    ephemeral_5m_input_tokens: int


@dataclass(frozen=True)
class _UsageWithSplit(FakeUsage):
    cache_creation: Any = None


def _ok_split(
    one_hour: int, five_min: int, *, text: str = "done", as_dict: bool = False
) -> FakeMessage:
    split: Any = (
        {"ephemeral_1h_input_tokens": one_hour, "ephemeral_5m_input_tokens": five_min}
        if as_dict
        else _Split(one_hour, five_min)
    )
    return FakeMessage(
        content=[FakeTextBlock(text=text)],
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
        usage=_UsageWithSplit(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=one_hour + five_min,
            cache_creation=split,
        ),
    )


def _tool_use_split(one_hour: int, five_min: int, *, tool_id: str, name: str) -> FakeMessage:
    return FakeMessage(
        content=[FakeToolUseBlock(id=tool_id, name=name, input={})],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=_UsageWithSplit(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=one_hour + five_min,
            cache_creation=_Split(one_hour, five_min),
        ),
    )


def _markers(req: dict[str, Any]) -> list[dict[str, Any]]:
    """Every cache_control marker in API order (tools -> system -> messages)."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "cache_control" in node:
                found.append(node["cache_control"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for key in ("tools", "system", "messages"):
        walk(req.get(key))
    return found


def _assert_ttl_order(req: dict[str, Any]) -> None:
    """No 1-hour marker after a 5-minute one (the API's ordering rule)."""
    seen_five = False
    for marker in _markers(req):
        if marker.get("ttl") == "1h":
            assert not seen_five, f"1h marker after a 5m marker: {_markers(req)}"
        else:
            seen_five = True


def _confirm_email(name: str, _inp: dict) -> bool:
    return name == "send_email"


async def _ok_executor(_name: str, _inp: dict) -> ToolResult:
    return ToolResult(content="hit")


# --- complete() -------------------------------------------------------------------------


async def _complete_request(**kwargs: Any) -> dict[str, Any]:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok())
    await AnthropicClient(client=fake).complete(
        static_system_prefix="STATIC",
        retrieval_block="RETRIEVED",
        user_message="hi",
        history=({"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}),
        cache_history=True,
        **kwargs,
    )
    return fake.messages.captured_requests[0]


async def test_complete_default_request_is_byte_identical() -> None:
    default = await _complete_request()
    explicit_none = await _complete_request(prefix_cache_ttl=None)
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit_none, sort_keys=True)
    assert '"ttl"' not in json.dumps(default)
    assert default["system"] == [{"type": "text", "text": "STATIC", "cache_control": FIVE_MIN}]


async def test_complete_1h_marks_breakpoint_one_only() -> None:
    req = await _complete_request(prefix_cache_ttl="1h")
    assert req["system"] == [{"type": "text", "text": "STATIC", "cache_control": ONE_HOUR}]
    # retrieval (BP2) and history (BP3) stay 5 minutes
    assert _markers(req) == [ONE_HOUR, FIVE_MIN, FIVE_MIN]
    _assert_ttl_order(req)


@pytest.mark.parametrize("bad", ["5m", "2h", "", "1H"])
async def test_complete_rejects_any_other_ttl_before_the_call(bad: str) -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok())
    with pytest.raises(ValueError, match="prefix_cache_ttl"):
        await AnthropicClient(client=fake).complete(
            static_system_prefix="S",
            user_message="hi",
            prefix_cache_ttl=bad,  # type: ignore[arg-type]
        )
    assert fake.messages.captured_requests == []


# --- usage split ------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["object", "dict"])
async def test_response_carries_the_ttl_split(shape: str) -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok_split(900, 40, as_dict=shape == "dict"))
    audit = RecordingAudit()
    resp = await AnthropicClient(client=fake, audit_logger=audit).complete(
        static_system_prefix="S", user_message="hi", prefix_cache_ttl="1h"
    )
    assert resp.cache_creation_input_tokens == 940
    assert resp.cache_creation_1h_input_tokens == 900
    assert resp.cache_creation_5m_input_tokens == 40
    success = [kw for _lvl, msg, kw in audit.events if msg == "llm_response_success"]
    assert success[0]["cache_creation_1h_input_tokens"] == 900


async def test_the_real_sdk_usage_model_is_read() -> None:
    """The attribute names are the SDK's own (`anthropic.types.Usage.cache_creation`)."""
    from anthropic.types import CacheCreation, Usage

    usage = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=0,
        cache_creation=CacheCreation(ephemeral_1h_input_tokens=960, ephemeral_5m_input_tokens=40),
    )
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(
        FakeMessage(
            content=[FakeTextBlock(text="ok")],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,  # type: ignore[arg-type]
        )
    )
    resp = await AnthropicClient(client=fake).complete(static_system_prefix="S", user_message="hi")
    assert (resp.cache_creation_1h_input_tokens, resp.cache_creation_5m_input_tokens) == (960, 40)


async def test_missing_split_reads_as_zero_and_does_not_add_up() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(cache_creation=700))  # no usage.cache_creation
    resp = await AnthropicClient(client=fake).complete(static_system_prefix="S", user_message="hi")
    assert resp.cache_creation_input_tokens == 700
    assert (resp.cache_creation_1h_input_tokens, resp.cache_creation_5m_input_tokens) == (0, 0)


async def test_a_non_int_split_reads_as_zero() -> None:
    """A consumer's spec-less Mock usage must not fail the parse."""
    from unittest.mock import Mock

    usage = Mock(input_tokens=10, output_tokens=5, cache_creation_input_tokens=3)
    usage.cache_read_input_tokens = 0
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(
        FakeMessage(
            content=[FakeTextBlock(text="ok")],
            model="claude-sonnet-4-6",
            stop_reason="end_turn",
            usage=usage,
        )
    )
    resp = await AnthropicClient(client=fake).complete(static_system_prefix="S", user_message="hi")
    assert (resp.cache_creation_1h_input_tokens, resp.cache_creation_5m_input_tokens) == (0, 0)


async def test_streamed_prompt_usage_carries_the_split() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_ok_split(500, 20))
    seen: list[PromptUsage] = []
    await AnthropicClient(client=fake, stream_transport=True).complete_messages(
        system_blocks=[{"type": "text", "text": "S"}],
        messages=[{"role": "user", "content": "hi"}],
        on_usage=seen.append,
    )
    assert seen == [
        PromptUsage(
            input_tokens=10,
            cache_creation_input_tokens=520,
            cache_read_input_tokens=0,
            cache_creation_1h_input_tokens=500,
            cache_creation_5m_input_tokens=20,
        )
    ]


# --- ToolUseLoop.run() / resume() -------------------------------------------------------


async def _run_two_calls(**kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_tool_use_split(800, 0, tool_id="tu_1", name="search"))
    fake.messages.responses.append(_ok_split(0, 60))
    result = await ToolUseLoop(client=AnthropicClient(client=fake)).run(
        static_system_prefix="SYS",
        user_message="find x",
        retrieval_block="CHUNKS",
        tools=TOOLS,
        executor=_ok_executor,
        max_rounds=3,
        cache_tool_rounds=True,
        **kwargs,
    )
    return result, fake.messages.captured_requests


async def test_run_default_requests_are_byte_identical() -> None:
    _, default = await _run_two_calls()
    _, explicit_none = await _run_two_calls(prefix_cache_ttl=None)
    assert json.dumps(default, sort_keys=True) == json.dumps(explicit_none, sort_keys=True)
    assert all('"ttl"' not in json.dumps(req) for req in default)


async def test_run_1h_marks_breakpoint_one_on_every_call() -> None:
    result, reqs = await _run_two_calls(prefix_cache_ttl="1h")
    assert len(reqs) == 2
    for req in reqs:
        assert req["system"][0]["cache_control"] == ONE_HOUR
        assert [m for m in _markers(req) if m != ONE_HOUR] == [FIVE_MIN] * (len(_markers(req)) - 1)
        _assert_ttl_order(req)
    # second call: retrieval BP2 + the moving tool-round marker, both 5 minutes
    assert _markers(reqs[1]) == [ONE_HOUR, FIVE_MIN, FIVE_MIN]
    assert result.final_text == "done"


async def test_run_sums_the_split_over_its_calls() -> None:
    result, _ = await _run_two_calls(prefix_cache_ttl="1h")
    assert result.cache_creation_input_tokens == 860
    assert result.cache_creation_1h_input_tokens == 800
    assert result.cache_creation_5m_input_tokens == 60


async def test_run_rejects_a_bad_ttl_before_any_call() -> None:
    fake = FakeAsyncAnthropic()
    with pytest.raises(ValueError, match="prefix_cache_ttl"):
        await ToolUseLoop(client=AnthropicClient(client=fake)).run(
            static_system_prefix="SYS",
            user_message="x",
            tools=TOOLS,
            executor=_ok_executor,
            max_rounds=1,
            prefix_cache_ttl="5m",  # type: ignore[arg-type]
        )
    assert fake.messages.captured_requests == []


async def _suspend(fake: FakeAsyncAnthropic, loop: ToolUseLoop, **kwargs: Any) -> Any:
    fake.messages.responses.append(_tool_use_split(700, 0, tool_id="tu_w", name="send_email"))
    return await loop.run(
        static_system_prefix="SYS",
        user_message="email x",
        tools=TOOLS,
        executor=_ok_executor,
        max_rounds=3,
        confirm=_confirm_email,
        **kwargs,
    )


async def test_suspend_reports_the_split_but_state_agg_is_unchanged() -> None:
    fake = FakeAsyncAnthropic()
    loop = ToolUseLoop(client=AnthropicClient(client=fake))
    suspended = await _suspend(fake, loop, prefix_cache_ttl="1h")
    assert suspended.pending_confirmation is not None
    assert suspended.cache_creation_1h_input_tokens == 700
    # the persisted state keeps exactly the four D6 counts it always had
    assert set(suspended.pending_confirmation.state["agg"]) == {"in", "out", "cc", "cr"}


@pytest.mark.parametrize(("ttl", "marker"), [("1h", ONE_HOUR), (None, FIVE_MIN)])
async def test_resume_marks_breakpoint_one_as_resupplied(ttl: str | None, marker: dict) -> None:
    fake = FakeAsyncAnthropic()
    loop = ToolUseLoop(client=AnthropicClient(client=fake))
    suspended = await _suspend(fake, loop, prefix_cache_ttl="1h")
    fake.messages.responses.append(_ok_split(0, 30, text="sent"))
    result = await loop.resume(
        state=suspended.pending_confirmation.state,
        decision=ExecuteDecision(),
        tools=TOOLS,
        executor=_ok_executor,
        confirm=_confirm_email,
        static_system_prefix="SYS",
        max_rounds=3,
        prefix_cache_ttl=ttl,  # type: ignore[arg-type]
    )
    assert result.final_text == "sent"
    resumed_req = fake.messages.captured_requests[-1]
    assert resumed_req["system"][0]["cache_control"] == marker
    _assert_ttl_order(resumed_req)
    # continuation-only counts (D6): the suspend call's 700 is not in here
    assert (result.cache_creation_1h_input_tokens, result.cache_creation_5m_input_tokens) == (0, 30)


async def test_loop_tolerates_a_response_without_the_split_fields() -> None:
    """A consumer's test double may return its own response object (no split attributes)."""

    @dataclass(frozen=True)
    class _BareResponse:
        content: str = "done"
        model: str = "m"
        stop_reason: str = "end_turn"
        input_tokens: int = 1
        output_tokens: int = 1
        cache_creation_input_tokens: int = 5
        cache_read_input_tokens: int = 0
        tool_use: tuple = ()

    class _Client:
        async def complete_messages(self, **_kw: Any) -> _BareResponse:
            return _BareResponse()

    result = await ToolUseLoop(client=_Client()).run(  # type: ignore[arg-type]
        static_system_prefix="SYS",
        user_message="x",
        tools=TOOLS,
        executor=_ok_executor,
        max_rounds=1,
        prefix_cache_ttl="1h",
    )
    assert result.cache_creation_input_tokens == 5
    assert (result.cache_creation_1h_input_tokens, result.cache_creation_5m_input_tokens) == (0, 0)
