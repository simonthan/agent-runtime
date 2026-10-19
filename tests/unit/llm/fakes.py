"""Hand-rolled fake satisfying the structural Protocol of ``anthropic.AsyncAnthropic``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class FakeStreamEvent:
    """A raw SDK stream event: only the fields AnthropicClient reads (T-316b)."""

    type: str
    delta: Any = None
    content_block: Any = None


def stream_events(message: FakeMessage, *, chunk: int = 4) -> list[FakeStreamEvent]:
    """The raw events a streamed `message` would arrive as: text blocks in `chunk`-char
    deltas, a content_block_start per block, framed by message_start / message_stop."""
    events = [FakeStreamEvent(type="message_start")]
    for block in message.content:
        events.append(FakeStreamEvent(type="content_block_start", content_block=block))
        text = getattr(block, "text", None)
        if getattr(block, "type", None) == "text" and text:
            events.extend(
                FakeStreamEvent(type="content_block_delta", delta=FakeDelta(text[i : i + chunk]))
                for i in range(0, len(text), chunk)
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
    ) -> None:
        self._message = message
        self._events = stream_events(message) if message is not None else []
        if truncate:
            self._events = self._events[:-1]  # the body ends before message_stop
        self._fail_after = fail_after
        self._exc = exc

    def __aiter__(self) -> Any:
        return self._iter()

    async def _iter(self) -> Any:
        for i, event in enumerate(self._events):
            if self._fail_after is not None and i == self._fail_after:
                assert self._exc is not None
                raise self._exc
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
            return _FakeStream(None, None, None)
        fail_after, exc = (None, None)
        if owner.stream_mid_exceptions:
            fail_after, exc = owner.stream_mid_exceptions.pop(0)
        truncate = owner.stream_truncated > 0
        if truncate:
            owner.stream_truncated -= 1
        return _FakeStream(owner.responses.pop(0), fail_after, exc, truncate=truncate)

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
    many next streams end before ``message_stop``. Every request (create or stream) lands in
    ``captured_requests``; streamed ones also in ``stream_requests``."""

    responses: list[FakeMessage] = field(default_factory=list)
    exceptions: list[Exception] = field(default_factory=list)
    captured_requests: list[dict[str, Any]] = field(default_factory=list)
    stream_requests: list[dict[str, Any]] = field(default_factory=list)
    stream_open_exceptions: list[Exception] = field(default_factory=list)
    stream_mid_exceptions: list[tuple[int, Exception]] = field(default_factory=list)
    stream_no_events: int = 0
    stream_truncated: int = 0

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
