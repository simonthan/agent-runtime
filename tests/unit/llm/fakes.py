"""Hand-rolled fake satisfying the structural Protocol of ``anthropic.AsyncAnthropic``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import asyncio


@dataclass(frozen=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True)
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True)
class FakeNonTextBlock:
    """Used to test ``LLMResponseError`` on non-text first block."""

    type: str = "image"


@dataclass(frozen=True)
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"
    # v0.47.0 (T-70172): the raw JSON a stream sends for this block as input_json_delta
    # events; None = no json deltas (the pre-v0.47.0 fake stream)
    raw_json: str | None = None


@dataclass(frozen=True)
class FakeMessage:
    content: list[Any]
    model: str
    stop_reason: str
    usage: FakeUsage


@dataclass(frozen=True)
class FakeDelta:
    text: str
    type: str = "text_delta"


@dataclass(frozen=True)
class FakeJsonDelta:
    partial_json: str
    type: str = "input_json_delta"


@dataclass(frozen=True)
class FakeStreamEvent:
    """A raw SDK stream event: only the fields AnthropicClient reads (T-316b)."""

    type: str
    delta: Any = None
    content_block: Any = None
    message: Any = None  # T-7624: message_start carries the message (its usage)


def stream_events(message: FakeMessage, *, chunk: int = 4) -> list[FakeStreamEvent]:
    """The raw events a streamed `message` would arrive as: text blocks in `chunk`-char
    deltas, a content_block_start per block, framed by message_start / message_stop."""
    events = [FakeStreamEvent(type="message_start", message=message)]
    for block in message.content:
        events.append(FakeStreamEvent(type="content_block_start", content_block=block))
        text = getattr(block, "text", None)
        if getattr(block, "type", None) == "text" and text:
            events.extend(
                FakeStreamEvent(type="content_block_delta", delta=FakeDelta(text[i : i + chunk]))
                for i in range(0, len(text), chunk)
            )
        raw = getattr(block, "raw_json", None)
        if raw:
            events.extend(
                FakeStreamEvent(type="content_block_delta", delta=FakeJsonDelta(raw[i : i + chunk]))
                for i in range(0, len(raw), chunk)
            )
        events.append(FakeStreamEvent(type="content_block_stop"))
    events.append(FakeStreamEvent(type="message_stop"))
    return events


class _FakeStream:
    def __init__(
        self,
        message: FakeMessage | None,
        fail_after: int | None,
        exc: Exception | None,
        *,
        truncate: bool = False,
        content_type: str = "text/event-stream",
        pause: tuple[int, asyncio.Event] | None = None,
    ) -> None:
        # the SDK's AsyncMessageStream exposes the httpx response (T-316b review M2)
        self.response = httpx.Response(200, headers={"content-type": content_type})
        self._message = message
        self._events = stream_events(message) if message is not None else []
        if truncate:
            self._events = self._events[:-1]  # the body ends before message_stop
        self._fail_after = fail_after
        self._exc = exc
        self._pause = pause
        self.yielded = 0  # v0.47.0: events handed out (an early stop reads fewer than all)
        self._snap: list[Any] = []  # blocks started so far (current_message_snapshot)
        self._json = ""

    def __aiter__(self) -> Any:
        return self._iter()

    def _accumulate(self, event: FakeStreamEvent) -> None:
        if event.type == "content_block_start":
            block = event.content_block
            if getattr(block, "type", None) == "tool_use":
                block = FakeToolUseBlock(id=block.id, name=block.name, input={})
            elif getattr(block, "type", None) == "text":
                block = FakeTextBlock(text="")
            self._snap.append(block)
            self._json = ""
        elif event.type == "content_block_delta" and self._snap:
            last = self._snap[-1]
            if isinstance(event.delta, FakeJsonDelta):
                from jiter import from_json  # the SDK's own partial parser

                self._json += event.delta.partial_json
                parsed = from_json(self._json.encode(), partial_mode=True)
                self._snap[-1] = FakeToolUseBlock(id=last.id, name=last.name, input=parsed)
            elif isinstance(event.delta, FakeDelta):
                self._snap[-1] = FakeTextBlock(text=last.text + event.delta.text)

    @property
    def current_message_snapshot(self) -> FakeMessage:
        """v0.47.0 -- like the SDK's: the blocks generated so far, tool input parsed in
        jiter partial mode (a string still being written is dropped), no stop reason yet,
        the usage `message_start` carried (output 1)."""
        assert self._message is not None
        usage = self._message.usage
        return FakeMessage(
            content=list(self._snap),
            model=self._message.model,
            stop_reason=None,  # type: ignore[arg-type]
            usage=FakeUsage(
                input_tokens=usage.input_tokens,
                output_tokens=1,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
            ),
        )

    async def _iter(self) -> Any:
        for i, event in enumerate(self._events):
            if self._pause is not None and i == self._pause[0]:
                await self._pause[1].wait()  # T-7624: a response still being generated
            if self._fail_after is not None and i == self._fail_after:
                assert self._exc is not None
                raise self._exc
            self._accumulate(event)
            self.yielded += 1
            yield event

    async def get_final_message(self) -> FakeMessage:
        # the real SDK asserts a snapshot exists: a stream with no events has none
        assert self._message is not None
        return self._message


class _FakeStreamManager:
    def __init__(self, owner: FakeMessages, kwargs: dict[str, Any]) -> None:
        self._owner = owner
        self._kwargs = kwargs

    async def __aenter__(self) -> _FakeStream:
        owner = self._owner
        owner.captured_requests.append(self._kwargs)
        owner.stream_requests.append(self._kwargs)
        if owner.stream_open_exceptions:
            raise owner.stream_open_exceptions.pop(0)
        if owner.stream_no_events:
            owner.stream_no_events -= 1
            return _FakeStream(None, None, None, content_type="application/json")
        if owner.stream_empty_sse:
            owner.stream_empty_sse -= 1
            return _FakeStream(None, None, None)
        fail_after, exc = (None, None)
        if owner.stream_mid_exceptions:
            fail_after, exc = owner.stream_mid_exceptions.pop(0)
        truncate = owner.stream_truncated > 0
        if truncate:
            owner.stream_truncated -= 1
        pause = owner.stream_pause
        owner.stream_pause = None
        stream = _FakeStream(
            owner.responses.pop(0), fail_after, exc, truncate=truncate, pause=pause
        )
        owner.streams.append(stream)
        return stream

    async def __aexit__(self, *exc_info: object) -> None:
        return None


@dataclass
class FakeMessages:
    """Mimics ``AsyncAnthropic().messages`` — ``.create()`` and (T-316b) ``.stream()``.

    ``stream()`` pops the next ``responses`` entry and replays it as raw events
    (``stream_events``). ``stream_open_exceptions`` raise when the stream opens (an HTTP
    error before any event); ``stream_mid_exceptions`` are ``(event_index, exc)`` pairs that
    raise mid-stream. ``stream_no_events`` = how many next streams end with NO event (an
    endpoint that ignored ``stream: true`` and answered plain JSON). ``stream_truncated`` = how
    many next streams end before ``message_stop``. ``stream_empty_sse`` = how many next streams
    are a real event stream (SSE content-type) that ends with no event (a cut body).
    ``stream_pause`` = ``(event_index, asyncio.Event)``: the next stream waits on the event
    before yielding that index (T-7624). Every request (create or stream) lands in
    ``captured_requests``; streamed ones also in ``stream_requests``."""

    responses: list[FakeMessage] = field(default_factory=list)
    exceptions: list[Exception] = field(default_factory=list)
    captured_requests: list[dict[str, Any]] = field(default_factory=list)
    stream_requests: list[dict[str, Any]] = field(default_factory=list)
    stream_open_exceptions: list[Exception] = field(default_factory=list)
    stream_mid_exceptions: list[tuple[int, Exception]] = field(default_factory=list)
    stream_no_events: int = 0
    stream_truncated: int = 0
    stream_empty_sse: int = 0
    stream_pause: tuple[int, asyncio.Event] | None = None
    streams: list[_FakeStream] = field(default_factory=list)  # v0.47.0: each opened stream

    async def create(self, **kwargs: Any) -> FakeMessage:
        self.captured_requests.append(kwargs)
        if self.exceptions:
            raise self.exceptions.pop(0)
        return self.responses.pop(0)

    def stream(self, **kwargs: Any) -> _FakeStreamManager:
        return _FakeStreamManager(self, kwargs)


@dataclass
class FakeAsyncAnthropic:
    """Drop-in for ``anthropic.AsyncAnthropic`` in tests."""

    messages: FakeMessages = field(default_factory=FakeMessages)
    # T-316b: AnthropicClient reads it to decide whether a mid-stream failure may be retried
    max_retries: int = 0


def make_ok(
    *,
    text: str = "hello",
    model: str = "claude-sonnet-4-6",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> FakeMessage:
    """Build a successful fake Message."""
    return FakeMessage(
        content=[FakeTextBlock(text=text)],
        model=model,
        stop_reason=stop_reason,
        usage=FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def make_tool_use(
    *,
    tool_id: str = "tu_1",
    name: str = "search",
    tool_input: dict | None = None,
    text: str = "",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> FakeMessage:
    """Build a fake Message with stop_reason='tool_use' and a FakeToolUseBlock."""
    content: list = []
    if text:
        content.append(FakeTextBlock(text=text))
    content.append(FakeToolUseBlock(id=tool_id, name=name, input=tool_input or {}))
    return FakeMessage(
        content=content,
        model=model,
        stop_reason="tool_use",
        usage=FakeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
