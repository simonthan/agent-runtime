"""Model-output stream events (v0.43.0, TBP T-316b).

A consumer that wants to show the model's text while it is being generated passes a
``StreamListener`` to ``AnthropicClient.complete_messages(on_stream=...)`` /
``complete(on_stream=...)`` or to ``ToolUseLoop.run(on_stream=...)`` / ``resume(...)``.

Events, in order, for one model call driven by the loop::

    ModelCallStarted(round_index, tools_offered)
    TextDelta(text) ...            # zero or more, in generation order
    ToolUseStarted(name) ...       # a tool_use block began (this call is a tool round)
    ModelCallEnded(stop_reason)    # None when the call raised

The client alone emits only ``TextDelta`` and ``ToolUseStarted``; ``ModelCallStarted`` /
``ModelCallEnded`` come from the loop, which knows the round. A call that offered tools may
still end in a tool round, so its text is not known to be the answer until
``ModelCallEnded(stop_reason != "tool_use")`` -- a ``ToolUseStarted`` says early that it is
narration. A call with ``tools_offered=False`` (the forced-final call, the empty-reply
recovery) can only end in text.

CONTRACT: the listener runs inline in the model call's read loop, so it must return
quickly (put the event on a queue; never await network I/O). An exception it raises is
caught, logged once as ``llm_stream_listener_failed`` and the listener is detached for the
rest of the run -- a consumer bug never fails the model call. ``CancelledError`` propagates.
The events carry no policy: the loop still returns the same ``ToolLoopResult``, and the
consumer still decides what to show (its post-processing of the final text is unchanged).

``PromptUsage`` (v0.44.0, TBP T-7624) is NOT a stream event: it goes to a separate, synchronous
``on_usage`` callback of ``AnthropicClient.complete_messages`` so that reporting it never
changes what counts as handed to a screen (a stream event does: see the client's one retry
after the stream started). A streamed call reports it once per attempt, when the response
starts (``message_start``), so a consumer can account for the prompt of a call that is then
cancelled or fails and never returns a ``ClaudeResponse``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_runtime.logging import AuditLogger

logger = logging.getLogger(__name__)

__all__ = [
    "ModelCallEnded",
    "ModelCallStarted",
    "PromptUsage",
    "StreamEvent",
    "StreamListener",
    "TextDelta",
    "ToolUseStarted",
    "UsageListener",
    "guard_listener",
]


@dataclass(frozen=True, slots=True)
class ModelCallStarted:
    """A model call is about to be sent. `round_index` = tool rounds committed before it
    (0 = the run's first call; on `resume()` it continues from the suspended round).
    `tools_offered=False` = a no-tools call whose text can only be the answer."""

    round_index: int
    tools_offered: bool


@dataclass(frozen=True, slots=True)
class TextDelta:
    """The next piece of the model's text, exactly as generated."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolUseStarted:
    """A tool_use block began in this call: the call's text so far is narration beside a
    tool round, not the answer."""

    name: str


@dataclass(frozen=True, slots=True)
class ModelCallEnded:
    """The call finished. `stop_reason` is the response's (e.g. "end_turn", "tool_use",
    "max_tokens"), or None when the call raised (the exception still propagates)."""

    stop_reason: str | None


StreamEvent = ModelCallStarted | TextDelta | ToolUseStarted | ModelCallEnded
StreamListener = Callable[[StreamEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PromptUsage:
    """v0.44.0 (TBP T-7624) -- the prompt side of a streamed call's usage, as the endpoint
    reports it when the response starts (``message_start``). ``input_tokens`` is the uncached
    part, as in ``ClaudeResponse``; the three counts add up to the whole prompt. The output
    count is not known until the response ends, so it is not here.

    ``cache_creation_1h_input_tokens`` / ``cache_creation_5m_input_tokens`` (v0.45.0, TBP
    T-7683) split the cache write by TTL, as ``ClaudeResponse``'s fields of the same name."""

    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cache_creation_1h_input_tokens: int = 0
    cache_creation_5m_input_tokens: int = 0


UsageListener = Callable[[PromptUsage], None]


class _GuardedListener:
    """Wraps a consumer listener so its failure detaches it instead of failing the call."""

    __slots__ = ("_audit", "_inner", "_live")

    def __init__(self, inner: StreamListener, audit: AuditLogger) -> None:
        self._inner = inner
        self._audit = audit
        self._live = True

    async def __call__(self, event: StreamEvent) -> None:
        if not self._live:
            return
        try:
            await self._inner(event)
        except Exception as exc:  # noqa: BLE001 -- a listener bug must never fail the call
            self._live = False
            self._audit.warning(
                "llm_stream_listener_failed",
                error_type=type(exc).__name__,
                event=type(event).__name__,
            )
            logger.warning(
                "llm_stream_listener_failed error_type=%s event=%s",
                type(exc).__name__,
                type(event).__name__,
            )


def guard_listener(listener: StreamListener | None, audit: AuditLogger) -> StreamListener | None:
    """`listener` wrapped once (idempotent); None stays None."""
    if listener is None or isinstance(listener, _GuardedListener):
        return listener
    return _GuardedListener(listener, audit)
