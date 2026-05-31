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
class FakeMessage:
    content: list[Any]
    model: str
    stop_reason: str
    usage: FakeUsage


@dataclass
class FakeMessages:
    """Mimics ``AsyncAnthropic().messages`` — only ``.create()`` implemented."""

    responses: list[FakeMessage] = field(default_factory=list)
    exceptions: list[Exception] = field(default_factory=list)
    captured_requests: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> FakeMessage:
        self.captured_requests.append(kwargs)
        if self.exceptions:
            raise self.exceptions.pop(0)
        return self.responses.pop(0)


@dataclass
class FakeAsyncAnthropic:
    """Drop-in for ``anthropic.AsyncAnthropic`` in tests."""

    messages: FakeMessages = field(default_factory=FakeMessages)


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
