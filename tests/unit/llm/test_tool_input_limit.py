"""v0.47.0 (TBP T-70172) -- stop a streamed call whose tool-call arguments run away.

TBP T-70157: a transaction-desk call wrote 16,000 output tokens (127 s, the whole output cap)
into a get_attachment_content call whose parsed input was 170 characters: the argument being
written was one endless string (an id) or a run of blank space. `AnthropicClient(
tool_input_limit=...)` stops reading such a stream at the bound and returns what was generated
as a `max_tokens` stop, with a `tool_input_report` saying why.
"""

from __future__ import annotations

import gc
import json
import time
from typing import Any

import pytest

pytest.importorskip("anthropic")

import anthropic
import httpx

from agent_runtime.llm import (
    AnthropicClient,
    ToolInputLimit,
    ToolInputScanner,
    ToolResult,
    ToolUseLoop,
)
from agent_runtime.llm.tool_input import KEY_TOO_LONG, STRING_TOO_LONG, WHITESPACE_RUN

from .conftest import RecordingAudit
from .fakes import (
    FakeAsyncAnthropic,
    FakeMessage,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    make_ok,
)

SYSTEM = [{"type": "text", "text": "sys"}]
MSGS = [{"role": "user", "content": "enter this"}]
TOOLS = [{"name": "get_attachment_content", "input_schema": {"type": "object"}}]
ID_LIMIT = 2000


def _id_limit(_tool: str, key: str) -> int | None:
    return ID_LIMIT if key == "id" or key.endswith("_id") else None


LIMIT = ToolInputLimit(max_whitespace_run=1000, max_key_chars=256, string_limit=_id_limit)


def _scan(text: str, limit: ToolInputLimit = LIMIT, *, step: int = 0) -> ToolInputScanner:
    scanner = ToolInputScanner("t", limit)
    if step <= 0:
        scanner.feed(text)
    else:
        for i in range(0, len(text), step):
            if scanner.feed(text[i : i + step]) is not None:
                break
    return scanner


# -- the scanner ----------------------------------------------------------------------------


def test_a_long_id_value_trips_with_its_key() -> None:
    s = _scan('{"message_id":"m1","attachment_id":"' + "A" * 5000)
    assert s.stopped == STRING_TOO_LONG
    assert s.stopped_key == "attachment_id"
    # it stopped at the bound, not at the end of the text
    assert s.report().open_string_chars == ID_LIMIT + 1


def test_an_id_at_the_bound_does_not_trip() -> None:
    s = _scan('{"attachment_id":"' + "A" * ID_LIMIT + '"}')
    assert s.stopped is None
    assert s.report().open_string_chars == 0


def test_a_long_value_under_an_unbounded_key_does_not_trip() -> None:
    body = "def f():\n    return 1\n" * 4000  # ~90k chars of real code
    s = _scan(json.dumps({"code": body}))
    assert s.stopped is None
    assert s.report().max_whitespace_run < 10


def test_nested_objects_and_arrays_resolve_the_value_key() -> None:
    seen: list[str] = []

    def record(_tool: str, key: str) -> int | None:
        seen.append(key)
        return None

    _scan(
        json.dumps({"q": "x", "lines": [{"item_id": "i1", "qty": "2"}], "ids": ["a", "b"]}),
        ToolInputLimit(string_limit=record),
    )
    assert seen == ["q", "item_id", "qty", "ids", "ids"]


def test_an_escaped_quote_does_not_close_the_string() -> None:
    s = _scan('{"attachment_id":"' + 'x\\"' * 1500)
    assert s.stopped == STRING_TOO_LONG


def test_a_whitespace_run_between_tokens_trips() -> None:
    s = _scan('{"message_id":"m1",' + " \n" * 800)
    assert s.stopped == WHITESPACE_RUN
    assert s.report().max_whitespace_run == 1001


def test_a_run_of_newline_escapes_inside_a_string_trips() -> None:
    s = _scan('{"body":"Hello' + "\\n" * 1500)
    assert s.stopped == WHITESPACE_RUN
    assert s.stopped_key == "body"


def test_a_blank_run_inside_a_key_reports_that_key() -> None:
    """R3 F3 -- a run inside a key string names the key being written, not the last value's."""
    s = _scan('{"message_id":"m1","attach' + " " * 1500, ToolInputLimit(max_whitespace_run=1000))
    assert s.stopped == WHITESPACE_RUN
    assert s.stopped_key == "attach" + " " * 58  # the kept start of the key


def test_any_other_character_resets_the_run() -> None:
    s = _scan('{"body":"' + ("\\n" * 900 + "x") * 5 + '"}')
    assert s.stopped is None
    assert s.report().max_whitespace_run == 900


def test_escaped_characters_in_a_key_keep_their_text() -> None:
    """iav review F2 -- an escape in a key is part of the key (a \\uXXXX escape decoded), so
    `string_limit` sees the real key: `"\\u0069\\u0064"` is `"id"`."""
    seen: list[str] = []

    def record(_tool: str, key: str) -> int | None:
        seen.append(key)
        return None

    _scan(
        '{"a\\"b":"1","k\\u00e9y":"2","\\u0069\\u0064":"3","x\\u12":"4"}',
        ToolInputLimit(string_limit=record),
        step=1,
    )
    assert seen == ['a"b', "k\u00e9y", "id", "x"]
    s = _scan('{"\\u0069\\u0064":"' + "A" * 5000)
    assert (s.stopped, s.stopped_key) == (STRING_TOO_LONG, "id")


def test_a_runaway_key_trips() -> None:
    s = _scan('{"message_id":"m1","' + "k" * 400)
    assert s.stopped == KEY_TOO_LONG
    assert s.stopped_key == "k" * 64  # the kept start of the key


def test_chunking_does_not_change_the_result() -> None:
    text = '{"message_id":"m1","attachment_id":"' + "AB" * 3000
    whole, by_char, by_7 = _scan(text), _scan(text, step=1), _scan(text, step=7)
    assert whole.report().open_string_chars == by_char.report().open_string_chars
    assert (whole.stopped, by_char.stopped, by_7.stopped) == (STRING_TOO_LONG,) * 3


def test_report_counts_the_tail_and_the_open_string() -> None:
    r = _scan('{"note":"' + "ab" * 400).report()
    assert r.stopped is None
    assert r.key == "note"
    assert r.open_string_chars == 800
    assert r.tail_distinct == 2
    assert r.chars == len('{"note":"') + 800


def test_no_bounds_never_trip() -> None:
    s = _scan('{"attachment_id":"' + " " * 50_000, ToolInputLimit())
    assert s.stopped is None


def test_malformed_json_never_raises() -> None:
    for text in ('}}]]"', '{"a":::,,,}', "\\\\\\", '["x', '{"a"'):
        _scan(text)


def _cpu(fn: Any, arg: Any) -> float:
    t0 = time.thread_time()
    fn(arg)
    return time.thread_time() - t0


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda n: '{"body":"' + " " * n, id="space-run"),
        pytest.param(lambda n: '{"body":"' + "\\n" * (n // 2), id="newline-escape-run"),
        pytest.param(lambda n: '{"body":"' + "A" * n, id="long-string"),
        pytest.param(lambda n: '{"a":[' + '{"k":"v"},' * (n // 10), id="deep-list"),
    ],
)
def test_the_scan_is_linear_in_small_chunks(make: Any) -> None:
    """A stream arrives in small deltas: the scan must not re-read what it has seen. CPU time
    of the calling thread at n vs n/10 (linear ~10, quadratic ~100), as TBP's
    tests/perf_scaling.assert_scales_linearly."""
    unbounded = ToolInputLimit(string_limit=lambda _t, _k: None)

    def run(text: str) -> None:
        scanner = ToolInputScanner("t", unbounded)
        for i in range(0, len(text), 8):
            scanner.feed(text[i : i + 8])

    big_in, small_in = make(400_000), make(40_000)
    gc.disable()
    try:
        big = min(_cpu(run, big_in) for _ in range(3))
        small = max(min(_cpu(run, small_in) for _ in range(3)), 50e-6)
    finally:
        gc.enable()
    assert big / small < 30, (big, small)


# -- the client (fake SDK stream) ----------------------------------------------------------


def _runaway(raw: str, *, name: str = "get_attachment_content", text: str = "") -> FakeMessage:
    """A response whose last tool call streams `raw`; as generated to the cap (max_tokens)."""
    content: list[Any] = [FakeTextBlock(text=text)] if text else []
    content.append(FakeToolUseBlock(id="tu_1", name=name, input={}, raw_json=raw))
    return FakeMessage(
        content=content,
        model="claude-sonnet-4-6",
        stop_reason="max_tokens",
        usage=FakeUsage(input_tokens=900, output_tokens=16000, cache_read_input_tokens=50),
    )


def _client(fake: FakeAsyncAnthropic, audit: RecordingAudit | None = None, **kw: Any) -> Any:
    kw.setdefault("tool_input_limit", LIMIT)
    kw.setdefault("stream_transport", True)
    return AnthropicClient(client=fake, audit_logger=audit, **kw)


async def test_a_runaway_id_is_stopped_early_as_a_max_tokens_stop() -> None:
    fake, audit = FakeAsyncAnthropic(), RecordingAudit()
    raw = '{"message_id":"m1","attachment_id":"' + "A" * 60_000
    fake.messages.responses.append(_runaway(raw, text="Reading it."))
    resp = await _client(fake, audit).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, tools=TOOLS
    )
    stream = fake.messages.streams[0]
    assert stream.yielded < len(raw) // 4 // 10  # stopped within the first tenth of the stream
    assert resp.stop_reason == "max_tokens"
    assert resp.content == "Reading it."
    assert [b.name for b in resp.tool_use] == ["get_attachment_content"]
    assert resp.tool_use[0].input == {"message_id": "m1"}  # the open string is dropped
    report = resp.tool_input_report
    assert report is not None
    assert (report.stopped, report.key, report.tool) == (
        STRING_TOO_LONG,
        "attachment_id",
        "get_attachment_content",
    )
    assert resp.input_tokens == 900
    assert resp.cache_read_input_tokens == 50
    # no message_delta came: the output is estimated from what streamed (~4 chars a token)
    assert resp.output_tokens == -(-(len("Reading it.") + report.chars) // 4)
    stopped = [e for e in audit.events if e[1] == "llm_tool_input_stopped"]
    assert len(stopped) == 1
    assert stopped[0][2]["reason"] == STRING_TOO_LONG


async def test_a_whitespace_runaway_is_stopped_early() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_runaway('{"message_id":"m1",' + "\n" * 60_000))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.stop_reason == "max_tokens"
    assert resp.tool_input_report is not None
    assert resp.tool_input_report.stopped == WHITESPACE_RUN
    assert resp.tool_use[0].input == {"message_id": "m1"}


async def test_a_long_legitimate_argument_runs_to_the_end() -> None:
    fake = FakeAsyncAnthropic()
    body = "def f(x):\n    return x * 2\n\n" * 2000
    raw = json.dumps({"path": "a.py", "content": body})
    msg = FakeMessage(
        content=[
            FakeToolUseBlock(id="tu_1", name="write_file", input=json.loads(raw), raw_json=raw)
        ],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=10, output_tokens=15000),
    )
    fake.messages.responses.append(msg)
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.stop_reason == "tool_use"
    assert resp.tool_input_report is None
    assert resp.output_tokens == 15000
    assert fake.messages.streams[0].yielded == len(fake.messages.streams[0]._events)


async def test_a_cut_at_the_cap_inside_a_tool_call_is_reported_not_stopped() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_runaway('{"message_id":"m1","content":"' + "word " * 3000))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.stop_reason == "max_tokens"
    assert resp.output_tokens == 16000  # the endpoint's own count
    report = resp.tool_input_report
    assert report is not None
    assert report.stopped is None
    assert report.key == "content"
    assert report.open_string_chars == len("word ") * 3000


async def test_a_cut_in_text_after_no_tool_call_has_no_report() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(make_ok(text="long answer", stop_reason="max_tokens"))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.tool_input_report is None


async def test_no_limit_reads_the_whole_stream() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_runaway('{"attachment_id":"' + "A" * 20_000))
    resp = await _client(fake, tool_input_limit=None).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, tools=TOOLS
    )
    assert resp.tool_input_report is None
    assert resp.output_tokens == 16000
    assert fake.messages.streams[0].yielded == len(fake.messages.streams[0]._events)


async def test_a_call_without_tools_is_not_watched() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_runaway('{"attachment_id":"' + "A" * 20_000))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS)
    assert resp.tool_input_report is None
    assert fake.messages.streams[0].yielded == len(fake.messages.streams[0]._events)


async def test_a_call_that_is_not_streamed_is_not_watched() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(_runaway('{"attachment_id":"' + "A" * 20_000))
    resp = await _client(fake, stream_transport=False).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, tools=TOOLS
    )
    assert resp.tool_input_report is None
    assert fake.messages.stream_requests == []


async def test_the_watch_starts_over_on_the_stream_retry() -> None:
    """A mid-stream failure before anything was forwarded is retried once (T-316b): the
    second attempt is judged on its own events."""
    fake = FakeAsyncAnthropic(max_retries=1)
    fake.messages.responses.append(_runaway('{"message_id":"m1","note":"' + "x" * 50))
    fake.messages.responses.append(
        FakeMessage(
            content=[
                FakeToolUseBlock(
                    id="tu_2",
                    name="get_attachment_content",
                    input={"message_id": "m2"},
                    raw_json='{"message_id":"m2"}',
                )
            ],
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=10, output_tokens=12),
        )
    )
    fake.messages.stream_mid_exceptions.append((5, httpx.ReadError("dropped")))
    resp = await _client(fake).complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.stop_reason == "tool_use"
    assert resp.tool_input_report is None


async def test_the_loop_ends_on_a_stopped_call_and_runs_no_tool() -> None:
    fake = FakeAsyncAnthropic()
    fake.messages.responses.append(
        _runaway('{"message_id":"m1","attachment_id":"' + "A" * 60_000, text="On it.")
    )
    ran: list[str] = []

    async def executor(name: str, _input: dict[str, Any]) -> ToolResult:
        ran.append(name)
        return ToolResult(content="x")

    loop = ToolUseLoop(client=_client(fake))
    result = await loop.run(
        static_system_prefix="sys",
        user_message="enter this",
        tools=TOOLS,
        executor=executor,
        max_rounds=4,
    )
    assert result.stop_reason == "max_tokens"
    assert ran == []
    assert result.final_text == "On it."


# -- the real SDK stream --------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def test_the_real_sdk_stream_is_closed_at_the_bound() -> None:
    """End to end through anthropic.AsyncAnthropic's own stream accumulator: a 16k-token
    runaway id is stopped after about 2,000 characters and the response body is not read on."""
    pieces = 4000
    sent = [0]

    async def body() -> Any:
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 900, "output_tokens": 1},
                },
            },
        )
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_attachment_content",
                    "input": {},
                },
            },
        )
        head = '{"message_id": "m1", "attachment_id": "'
        delta = {"type": "input_json_delta", "partial_json": head}
        yield _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": delta},
        )
        for _ in range(pieces):
            sent[0] += 1
            delta = {"type": "input_json_delta", "partial_json": "QUFBQUFB" * 2}
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": delta},
            )
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
                "usage": {"output_tokens": 16000},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body())

    sdk = anthropic.AsyncAnthropic(
        api_key="test",
        base_url="http://llm.invalid",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    client = AnthropicClient(client=sdk, stream_transport=True, tool_input_limit=LIMIT)
    resp = await client.complete_messages(system_blocks=SYSTEM, messages=MSGS, tools=TOOLS)
    assert resp.stop_reason == "max_tokens"
    assert resp.tool_use[0].name == "get_attachment_content"
    assert resp.tool_use[0].input == {"message_id": "m1"}
    assert resp.tool_input_report is not None
    assert resp.tool_input_report.stopped == STRING_TOO_LONG
    assert sent[0] < 200  # ~2,000 characters of 16-character pieces, not all 4,000 pieces


@pytest.mark.parametrize("result", ["raise", "not-an-int"])
async def test_a_failing_string_limit_is_unbounded_and_audited_once(result: str) -> None:
    """iav review F1 -- a consumer bug in `string_limit` (raises, or returns a non-int) never
    fails the call: the value is unbounded and one `llm_tool_input_limit_failed` is audited."""

    def broken(_tool: str, key: str) -> Any:
        if result == "raise":
            raise KeyError(key)
        return "2000"

    fake, audit = FakeAsyncAnthropic(), RecordingAudit()
    raw = json.dumps({"message_id": "m1", "attachment_id": "A" * 5000})
    msg = FakeMessage(
        content=[FakeToolUseBlock(id="tu_1", name="t", input=json.loads(raw), raw_json=raw)],
        model="claude-sonnet-4-6",
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=10, output_tokens=1500),
    )
    fake.messages.responses.append(msg)
    limit = ToolInputLimit(max_whitespace_run=1000, string_limit=broken)
    resp = await _client(fake, audit, tool_input_limit=limit).complete_messages(
        system_blocks=SYSTEM, messages=MSGS, tools=TOOLS
    )
    assert resp.stop_reason == "tool_use"
    assert resp.tool_input_report is None
    failed = [e for e in audit.events if e[1] == "llm_tool_input_limit_failed"]
    assert len(failed) == 1
    assert failed[0][2]["error_type"] == ("KeyError" if result == "raise" else "str")
