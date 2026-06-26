"""T-036: durable conversation history + fork primitive on SessionManager."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agent_runtime.session.events import SessionAlreadyActive
from agent_runtime.session.manager import SessionManager
from agent_runtime.session.testing import FakeRedisClient, FakeSessionRepository


def _make_manager(*, repo: Any | None = None, max_history: int | None = None) -> SessionManager:
    return SessionManager(
        session_repo=repo or FakeSessionRepository(),
        redis_client=FakeRedisClient(),
        idle_timeout=timedelta(minutes=30),
        max_history=max_history,
    )


class _BareRepo:
    """Implements only SessionRepositoryProtocol — no durable support, no flag."""

    async def upsert_resume_data(self, **_: Any) -> None: ...
    async def get_session_for_resume(self, **_: Any) -> None:
        return None

    async def get_active_session(self, **_: Any) -> None:
        return None


class _NameCollisionRepo(_BareRepo):
    """Has all three durable method NAMES but NOT the capability flag — must NOT be
    treated as durable (proves the gate is the flag, not coincidental names)."""

    async def append_message(self, **_: Any) -> None: ...
    async def get_conversation_history(self, **_: Any) -> list[dict[str, Any]]:
        return []

    async def list_sessions(self, **_: Any) -> list[Any]:
        return []


async def test_update_session_writes_through_to_durable_repo():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    s = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(s.id, add_message={"role": "user", "content": "hi"})
    await mgr.update_session(s.id, add_message={"role": "assistant", "content": "yo"})
    durable = await repo.get_conversation_history(session_id=s.id, user_id="u1", bot_id="b1")
    assert [m["content"] for m in durable] == ["hi", "yo"]
    assert all("timestamp" in m for m in durable)


async def test_durable_keeps_full_transcript_past_redis_cap():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo, max_history=2)
    s = await mgr.create_session(user_id="u1", bot_id="b1")
    for i in range(5):
        await mgr.update_session(s.id, add_message={"role": "user", "content": str(i)})
    redis_session = await mgr.get_session(s.id)
    assert len(redis_session.conversation_history) == 2  # Redis trimmed
    durable = await repo.get_conversation_history(session_id=s.id, user_id="u1", bot_id="b1")
    assert [m["content"] for m in durable] == ["0", "1", "2", "3", "4"]  # durable full


async def test_bare_repo_degrades_gracefully():
    """A repo without the capability flag → no write-through, no error."""
    mgr = _make_manager(repo=_BareRepo())
    assert mgr._durable is None
    s = await mgr.create_session(user_id="u1", bot_id="b1")
    updated = await mgr.update_session(s.id, add_message={"role": "user", "content": "x"})
    assert updated is not None
    assert await mgr.list_sessions(user_id="u1", bot_id="b1") == []


async def test_name_collision_without_flag_not_durable():
    """All three durable method NAMES present but no flag → NOT durable (CRITICAL-1)."""
    mgr = _make_manager(repo=_NameCollisionRepo())
    assert mgr._durable is None


async def test_write_through_best_effort_swallows_repo_error(monkeypatch):
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    s = await mgr.create_session(user_id="u1", bot_id="b1")

    async def _boom(**_: Any) -> None:
        raise RuntimeError("durable store down")

    monkeypatch.setattr(repo, "append_message", _boom)
    updated = await mgr.update_session(s.id, add_message={"role": "user", "content": "x"})
    assert updated is not None  # turn still succeeds
    assert updated.conversation_history[-1]["content"] == "x"


async def test_fork_session_seeds_history_into_new_session():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    src = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(src.id, add_message={"role": "user", "content": "old q"})
    await mgr.update_session(src.id, add_message={"role": "assistant", "content": "old a"})
    await mgr.end_session(src.id)  # free the active-index for the fork

    forked = await mgr.fork_session(source_session_id=src.id, user_id="u1", bot_id="b1")
    assert forked.id != src.id
    assert forked.status == "active"
    assert [m["content"] for m in forked.conversation_history] == ["old q", "old a"]


async def test_fork_does_not_repersist_seeded_history():
    """Forked transcript is Redis context only — new session_id's durable store is empty."""
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    src = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(src.id, add_message={"role": "user", "content": "old"})
    await mgr.end_session(src.id)
    forked = await mgr.fork_session(source_session_id=src.id, user_id="u1", bot_id="b1")
    durable_new = await repo.get_conversation_history(
        session_id=forked.id, user_id="u1", bot_id="b1"
    )
    assert durable_new == []


async def test_fork_raises_if_active_session_exists():
    """The caller MUST end the active session first; fork surfaces SessionAlreadyActive."""
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    src = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(src.id, add_message={"role": "user", "content": "old"})
    await mgr.end_session(src.id)
    await mgr.create_session(user_id="u1", bot_id="b1")  # a new active session now exists
    with pytest.raises(SessionAlreadyActive):
        await mgr.fork_session(source_session_id=src.id, user_id="u1", bot_id="b1")


async def test_fork_ownership_wrong_user_gets_empty_session():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    src = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(src.id, add_message={"role": "user", "content": "secret"})
    forked = await mgr.fork_session(source_session_id=src.id, user_id="u2", bot_id="b1")
    assert forked.conversation_history == []  # ownership guard → empty


async def test_list_sessions_returns_summaries_newest_first():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    s1 = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(s1.id, add_message={"role": "user", "content": "one"})
    await mgr.end_session(s1.id)
    s2 = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(s2.id, add_message={"role": "user", "content": "two"})
    repo._by_id[s2.id].created_at = datetime.now(UTC) + timedelta(seconds=1)  # s2 newer

    rows = await mgr.list_sessions(user_id="u1", bot_id="b1", limit=10)
    assert [str(r.id) for r in rows] == [s2.id, s1.id]
    assert rows[0].message_count == 1


async def test_list_sessions_before_cursor_paginates():
    repo = FakeSessionRepository()
    mgr = _make_manager(repo=repo)
    base = datetime.now(UTC)
    ids = []
    for i in range(3):
        s = await mgr.create_session(user_id="u1", bot_id="b1")
        repo._by_id[s.id].created_at = base + timedelta(seconds=i)
        await mgr.end_session(s.id)
        ids.append(s.id)
    cursor = base + timedelta(seconds=2)  # newest is ids[2]@+2s; before it → [1],[0]
    rows = await mgr.list_sessions(user_id="u1", bot_id="b1", limit=10, before=cursor)
    assert [str(r.id) for r in rows] == [ids[1], ids[0]]


async def test_create_session_initial_history_respects_cap():
    mgr = _make_manager(max_history=2)
    seeded = [{"role": "user", "content": str(i)} for i in range(5)]
    s = await mgr.create_session(user_id="u1", bot_id="b1", initial_history=seeded)
    assert [m["content"] for m in s.conversation_history] == ["3", "4"]
