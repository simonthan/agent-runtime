"""Public model types for the LLM wrapper.

- ``Message`` / ``History`` — conversation history (passed verbatim to the SDK)
- ``ClaudeResponse`` — frozen result with token-usage and cache statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class Message(TypedDict):
    """A single conversation-history entry. ``system`` role is NOT a Message —
    system prompts are first-class ``complete()`` params, not history entries."""

    role: Literal["user", "assistant"]
    content: str


History = tuple[Message, ...]
"""Immutable conversation history. Callers slice or extend explicitly via tuple ops."""


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """One tool call the model requested in a `tool_use` content block."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ClaudeResponse:
    """Frozen result of a successful ``complete()`` call.

    Cache fields are 0 when no cache write/read happened.
    """

    content: str
    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    # MUST be the LAST field — all 7 preceding fields are non-defaulted;
    # a defaulted field can only follow them (slots=True + defaulted field
    # is valid — defaults live on the class, not the instance dict).
    tool_use: tuple[ToolUseBlock, ...] = ()
