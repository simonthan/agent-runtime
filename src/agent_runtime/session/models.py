"""Conversation-session value types — ORM-free wire shapes.

Consumers own their SQLAlchemy/asyncpg schema; this module describes the
agent-runtime boundary contract.
"""

# Note: `from __future__ import annotations` is intentionally omitted.
# Pydantic v2 and dataclasses evaluate annotations at class-creation time
# via get_type_hints(); moving datetime/UUID into TYPE_CHECKING would cause
# NameError at runtime. TC003 is suppressed via per-file-ignores.
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


@dataclass(slots=True)
class SessionData:
    """In-memory snapshot of an active conversation session."""

    id: str
    user_id: str
    bot_id: str
    created_at: datetime
    updated_at: datetime
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    conversation_history: list[dict] = field(default_factory=list)  # type: ignore[type-arg]
    client_context: dict[str, Any] = field(default_factory=dict)


class ConversationState(BaseModel):
    """Pydantic baseline conversation state. Consumers subclass to add
    domain sub-states (e.g. a ticket-draft state in a helpdesk consumer).

    Slimmed from ithelpdesk SessionState — IHD-named sub-states drop out;
    consumers extend per ``session_state_ihd.py`` precedent.
    """

    model_config = ConfigDict(extra="allow")
    history: list[dict] = Field(default_factory=list)  # type: ignore[type-arg]
    response: str | None = None


class ResumeRow(BaseModel):
    """Repository payload for resume lookups (ORM-free wire shape).

    ``last_message_at`` may be ``None`` for sessions that have not yet
    exchanged a message — caller falls back to ``created_at`` for TTL
    window comparison.

    Datetime invariant: ``created_at`` and ``last_message_at`` SHOULD be
    timezone-aware (UTC). SessionManager defensively coerces naive values at
    the ingestion boundary to protect against asyncpg's default driver which
    returns naive UTC datetimes from TIMESTAMP columns; concrete repositories
    should still emit tz-aware values to make the contract explicit at the
    SQL layer (cast with ``AT TIME ZONE 'UTC'`` or use ``TIMESTAMPTZ``).
    """

    id: UUID
    user_id: str
    bot_id: str
    status: str = "active"
    created_at: datetime
    last_message_at: datetime | None = None
    client_context: dict[str, Any] = Field(default_factory=dict)


class SessionSummaryRow(BaseModel):
    """Durable per-session summary for the user-facing history list (T-036).

    One row per past session of a (user_id, bot_id). ``title`` is derived by the
    concrete repository (first user message truncated, or a persisted summary) —
    agent-runtime defines only the wire shape. ``message_count`` is the durable
    transcript length. ``last_message_at`` may be ``None`` for a session that never
    exchanged a message.
    """

    id: UUID
    title: str | None = None
    status: str = "active"
    created_at: datetime
    last_message_at: datetime | None = None
    message_count: int = 0
