"""Tests for agent_runtime.session models."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agent_runtime.session.models import ConversationState, ResumeRow, SessionData


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# SessionData
# ---------------------------------------------------------------------------


def test_session_data_round_trip():
    now = _now()
    sd = SessionData(
        id="s1",
        user_id="u1",
        bot_id="b1",
        created_at=now,
        updated_at=now,
    )
    assert sd.id == "s1"
    assert sd.user_id == "u1"
    assert sd.bot_id == "b1"
    assert sd.created_at == now
    assert sd.updated_at == now
    assert sd.data == {}
    assert sd.status == "active"
    assert sd.conversation_history == []
    assert sd.client_context == {}


def test_session_data_with_overrides():
    now = _now()
    sd = SessionData(
        id="s2",
        user_id="u2",
        bot_id="b2",
        created_at=now,
        updated_at=now,
        data={"key": "value"},
        status="ended",
        conversation_history=[{"role": "user", "content": "hi"}],
        client_context={"tenant": "t1"},
    )
    assert sd.data == {"key": "value"}
    assert sd.status == "ended"
    assert sd.conversation_history[0]["role"] == "user"
    assert sd.client_context == {"tenant": "t1"}


# ---------------------------------------------------------------------------
# ResumeRow
# ---------------------------------------------------------------------------


def test_resume_row_minimal():
    uid = uuid4()
    now = _now()
    row = ResumeRow(id=uid, user_id="u1", bot_id="b1", created_at=now)
    assert isinstance(row.id, UUID)
    assert row.last_message_at is None
    assert row.client_context == {}
    assert row.status == "active"


def test_resume_row_with_last_message_at():
    uid = uuid4()
    now = _now()
    row = ResumeRow(
        id=uid,
        user_id="u1",
        bot_id="b1",
        created_at=now,
        last_message_at=now,
    )
    assert row.last_message_at == now


def test_resume_row_uuid_coercion_from_string():
    """ResumeRow accepts a string UUID and coerces it to UUID."""
    uid = uuid4()
    now = _now()
    row = ResumeRow(id=str(uid), user_id="u1", bot_id="b1", created_at=now)  # type: ignore[arg-type]
    assert isinstance(row.id, UUID)
    assert row.id == uid


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------


def test_conversation_state_default():
    cs = ConversationState()
    assert cs.history == []
    assert cs.response is None


def test_conversation_state_subclass_extension():
    """Consumers subclass ConversationState and add domain fields."""

    class ITState(ConversationState):
        ticket_number: str | None = None
        account_locked: bool = False

    s = ITState(ticket_number="INC001", account_locked=True)
    assert s.ticket_number == "INC001"
    assert s.account_locked is True
    # base fields still work
    assert s.history == []


def test_conversation_state_extra_allow_round_trip():
    """extra='allow' persists arbitrary keys through model_dump / model_validate."""
    cs = ConversationState(foo=1, bar="baz")  # type: ignore[call-arg]
    dumped = cs.model_dump()
    assert dumped["foo"] == 1
    assert dumped["bar"] == "baz"

    cs2 = ConversationState.model_validate({"foo": 1, "bar": "baz"})
    assert cs2.foo == 1  # type: ignore[attr-defined]
    assert cs2.bar == "baz"  # type: ignore[attr-defined]
