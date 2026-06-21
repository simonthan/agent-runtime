"""Tests for SessionManager.update_session — conversation_history bounding (SEC-6)."""

from datetime import timedelta

from agent_runtime.session.manager import SessionManager
from agent_runtime.session.testing import FakeRedisClient, FakeSessionRepository


def _make_manager(max_history: int | None = None) -> SessionManager:
    return SessionManager(
        session_repo=FakeSessionRepository(),
        redis_client=FakeRedisClient(),
        idle_timeout=timedelta(minutes=30),
        max_history=max_history,
    )


async def test_history_unbounded_by_default():
    """max_history=None preserves prior behaviour — no cap."""
    mgr = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    for i in range(50):
        await mgr.update_session(session.id, add_message={"role": "user", "text": str(i)})
    updated = await mgr.get_session(session.id)
    assert len(updated.conversation_history) == 50


async def test_history_capped_drops_oldest():
    """SEC-6: history length never exceeds the cap; oldest entries drop first."""
    mgr = _make_manager(max_history=5)
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    for i in range(20):
        await mgr.update_session(session.id, add_message={"role": "user", "text": str(i)})
    updated = await mgr.get_session(session.id)
    assert len(updated.conversation_history) == 5
    # Oldest (0-14) dropped; newest five (15-19) retained, in order.
    assert [m["text"] for m in updated.conversation_history] == ["15", "16", "17", "18", "19"]


async def test_history_cap_not_applied_to_data_only_update():
    """A data-only update (no add_message) must not trim existing history."""
    mgr = _make_manager(max_history=2)
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    for i in range(3):
        await mgr.update_session(session.id, add_message={"role": "user", "text": str(i)})
    # history is now capped at 2; a pure data update should leave it untouched
    await mgr.update_session(session.id, data={"k": "v"})
    updated = await mgr.get_session(session.id)
    assert len(updated.conversation_history) == 2
    assert updated.data["k"] == "v"
