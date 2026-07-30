"""Public model types for the LLM wrapper.

- ``Message`` / ``History`` — conversation history (passed verbatim to the SDK)
- ``ClaudeResponse`` — frozen result with token-usage and cache statistics
- ``LLMImage`` — base64 image content block for vision passthrough (T-067d)
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Literal, TypedDict


class Message(TypedDict):
    """A single conversation-history entry. ``system`` role is NOT a Message —
    system prompts are first-class ``complete()`` params, not history entries."""

    role: Literal["user", "assistant"]
    content: str


History = tuple[Message, ...]
"""Immutable conversation history. Callers slice or extend explicitly via tuple ops."""

ANTHROPIC_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
"""Media types the Anthropic Messages API accepts in base64 image source blocks."""


@dataclass(frozen=True, slots=True)
class LLMImage:
    """One image to include in the first user message of a turn (T-067d).

    ``data_b64`` is standard (non-urlsafe) base64 of the raw bytes. Construct
    via ``from_bytes`` unless the payload is already encoded. ``media_type``
    must be one of ``ANTHROPIC_IMAGE_MEDIA_TYPES`` — anything else raises
    ``ValueError`` at construction (fail-fast; the API would 400 mid-turn).
    """

    media_type: str
    data_b64: str

    def __post_init__(self) -> None:
        if self.media_type not in ANTHROPIC_IMAGE_MEDIA_TYPES:
            msg = (
                f"unsupported image media_type {self.media_type!r}; "
                f"expected one of {sorted(ANTHROPIC_IMAGE_MEDIA_TYPES)}"
            )
            raise ValueError(msg)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str) -> LLMImage:
        """Encode raw bytes (e.g. ``DownloadedImage.data``) to base64."""
        return cls(media_type=media_type, data_b64=base64.b64encode(data).decode("ascii"))

    def to_block(self) -> dict[str, Any]:
        """Render the Anthropic base64 image content block."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.data_b64,
            },
        }


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
