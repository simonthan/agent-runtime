"""Tests for agent_runtime.session events (ResumeDecision sealed union)."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from agent_runtime.session.events import (
    Active,
    NewSession,
    Resumable,
    ResumeDecision,
    SessionAlreadyActive,
)


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# ResumeDecision match-exhaustiveness
# ---------------------------------------------------------------------------


def _classify(decision: ResumeDecision) -> str:
    match decision:
        case NewSession():
            return "new"
        case Resumable(session_id=sid):
            return f"resumable:{sid}"
        case Active(session_id=sid):
            return f"active:{sid}"


def test_new_session_match():
    assert _classify(NewSession()) == "new"


def test_resumable_match():
    result = _classify(Resumable(session_id="s1", last_activity_ts=_now()))
    assert result == "resumable:s1"


def test_active_match():
    assert _classify(Active(session_id="s2")) == "active:s2"


def test_resume_decision_kind_literals():
    assert NewSession().kind == "new"
    assert Resumable(session_id="s", last_activity_ts=_now()).kind == "resumable"
    assert Active(session_id="s").kind == "active"


# ---------------------------------------------------------------------------
# SessionAlreadyActive
# ---------------------------------------------------------------------------


def test_session_already_active_carries_attrs():
    ts = _now()
    exc = SessionAlreadyActive(session_id="s1", last_activity_ts=ts)
    assert exc.session_id == "s1"
    assert exc.last_activity_ts == ts
    assert "s1" in str(exc)


def test_session_already_active_is_exception():
    exc = SessionAlreadyActive(session_id="x", last_activity_ts=_now())
    assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# Frozen-ness
# ---------------------------------------------------------------------------


def test_active_is_frozen():
    a = Active(session_id="s1")
    with pytest.raises(FrozenInstanceError):
        a.kind = "new"  # type: ignore[misc]


def test_resumable_is_frozen():
    r = Resumable(session_id="s1", last_activity_ts=_now())
    with pytest.raises(FrozenInstanceError):
        r.session_id = "s2"  # type: ignore[misc]


def test_new_session_is_frozen():
    n = NewSession()
    with pytest.raises(FrozenInstanceError):
        n.kind = "active"  # type: ignore[misc]
