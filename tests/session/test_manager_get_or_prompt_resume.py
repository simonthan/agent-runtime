"""Tests for SessionManager.get_or_prompt_resume and update_session."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_runtime.session.events import Active, NewSession, Resumable
from agent_runtime.session.manager import SessionManager
from agent_runtime.session.testing import FakeRedisClient, FakeSessionRepository


def _make_manager(
    idle_timeout: timedelta = timedelta(minutes=30),
    prefix: str = "session",
) -> tuple[SessionManager, FakeRedisClient, FakeSessionRepository]:
    redis = FakeRedisClient()
    repo = FakeSessionRepository()
    mgr = SessionManager(
        session_repo=repo,
        redis_client=redis,
        idle_timeout=idle_timeout,
        key_prefix=prefix,
    )
    return mgr, redis, repo


# ---------------------------------------------------------------------------
# NewSession: cold (Redis miss + DB miss)
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_cold_returns_new_session():
    mgr, _, _ = _make_manager()
    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")
    assert isinstance(decision, NewSession)


# ---------------------------------------------------------------------------
# Active: within idle window — lease extended atomically
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_active_within_window():
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")

    assert isinstance(decision, Active)
    assert decision.session_id == session.id

    # Verify the session key still exists (lease was extended via XX SET)
    raw_after = redis._store.get(f"session:{session.id}")
    assert raw_after is not None


# ---------------------------------------------------------------------------
# Resumable: after idle window lapse
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_lapsed_returns_resumable():
    """When session is older than idle_timeout, returns Resumable."""
    mgr, redis, _ = _make_manager(idle_timeout=timedelta(seconds=1))
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Backdate updated_at to force lapsed state
    raw = redis._store[f"session:{session.id}"]
    parsed = json.loads(raw)
    old_ts = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    parsed["updated_at"] = old_ts
    redis._store[f"session:{session.id}"] = json.dumps(parsed)

    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")
    assert isinstance(decision, Resumable)
    assert decision.session_id == session.id


# ---------------------------------------------------------------------------
# Cold-cache rehydration: Redis miss + DB hit
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_cold_cache_db_hit_returns_resumable():
    """Redis miss + DB hit → rehydrate, return Resumable."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Evict both the session key and the reverse index
    await redis.delete(f"session:{session.id}")
    await redis.delete("session:active:u1:b1")

    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")
    assert isinstance(decision, Resumable)
    assert decision.session_id == session.id


async def test_get_or_prompt_resume_rehydration_populates_redis():
    """After rehydration, get_session should hit Redis (not DB again)."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Evict from Redis
    await redis.delete(f"session:{session.id}")
    await redis.delete("session:active:u1:b1")

    # Rehydrate
    await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")

    # Subsequent get_session should hit Redis
    fetched = await mgr.get_session(session.id)
    assert fetched is not None
    assert fetched.id == session.id


# ---------------------------------------------------------------------------
# XX-SET failure fall-through (Opus review)
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_xx_set_failure_returns_resumable_with_same_session_id():
    """If the atomic XX set fails (key evicted between get and set), the
    result must be Resumable with the SAME original session_id — NOT a
    different ID from a DB lookup."""
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    original_sid = session.id

    # Subclass FakeRedisClient to return None once on xx=True for session key
    original_set = redis.set
    set_called = False

    async def patched_set(key, value, ex=None, px=None, nx=False, xx=False):
        nonlocal set_called
        if xx and key == f"session:{original_sid}" and not set_called:
            set_called = True
            # Simulate eviction: remove from store and return None
            redis._store.pop(key, None)
            return None
        return await original_set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    redis.set = patched_set  # type: ignore[method-assign]

    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")

    assert isinstance(decision, Resumable)
    # CRITICAL: same session_id, not a new one from DB
    assert decision.session_id == original_sid
    assert set_called  # confirm our mock was triggered


# ---------------------------------------------------------------------------
# update_session raw-dict pass-through (no ITSessionState filter)
# ---------------------------------------------------------------------------


async def test_update_session_raw_dict_pass_through():
    """update_session stores arbitrary keys without ITSessionState filtering."""
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    updated = await mgr.update_session(
        session.id,
        data={"arbitrary_key": "value", "unknown_field": 42},
    )
    assert updated is not None
    assert updated.data == {"arbitrary_key": "value", "unknown_field": 42}

    # Round-trip via get_session
    fetched = await mgr.get_session(session.id)
    assert fetched is not None
    assert fetched.data["arbitrary_key"] == "value"
    assert fetched.data["unknown_field"] == 42


async def test_update_session_replace_data():
    """replace_data=True replaces entire data dict."""
    mgr, _, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1", initial_context={"old": "data"})

    updated = await mgr.update_session(
        session.id,
        data={"new": "data"},
        replace_data=True,
    )
    assert updated is not None
    assert updated.data == {"new": "data"}
    assert "old" not in updated.data


async def test_update_session_add_message():
    mgr, _, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    updated = await mgr.update_session(
        session.id,
        add_message={"role": "user", "content": "hello"},
    )
    assert updated is not None
    assert len(updated.conversation_history) == 1
    assert updated.conversation_history[0]["role"] == "user"
    assert "timestamp" in updated.conversation_history[0]


# ---------------------------------------------------------------------------
# ConversationState extra="allow" round-trip
# ---------------------------------------------------------------------------


def test_conversation_state_extra_allow_round_trip():
    from agent_runtime.session.models import ConversationState

    cs = ConversationState(foo=1)  # type: ignore[call-arg]
    assert cs.model_dump()["foo"] == 1

    cs2 = ConversationState.model_validate({"foo": 1})
    assert cs2.foo == 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_empty_user_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="user_id required"):
        await mgr.get_or_prompt_resume(user_id="", bot_id="b1")


async def test_get_or_prompt_resume_empty_bot_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="bot_id required"):
        await mgr.get_or_prompt_resume(user_id="u1", bot_id="")


async def test_get_or_prompt_resume_whitespace_user_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="user_id required"):
        await mgr.get_or_prompt_resume(user_id="  ", bot_id="b1")


# ---------------------------------------------------------------------------
# end_session
# ---------------------------------------------------------------------------


async def test_end_session_marks_ended_and_clears_index():
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    result = await mgr.end_session(session.id)
    assert result is True

    # Index key cleared
    assert await redis.get("session:active:u1:b1") is None

    # Session status updated
    fetched = await mgr.get_session(session.id)
    assert fetched is not None
    assert fetched.status == "ended"


async def test_end_session_nonexistent_returns_false():
    mgr, _, _ = _make_manager()
    result = await mgr.end_session("nonexistent-session-id")
    assert result is False


async def test_end_session_decrements_counter():
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    assert await mgr.get_active_sessions_count("u1") == 1

    await mgr.end_session(session.id)

    # Counter should be cleaned up (0 or missing)
    count = await mgr.get_active_sessions_count("u1")
    assert count == 0


# ---------------------------------------------------------------------------
# touch_session
# ---------------------------------------------------------------------------


async def test_touch_session_existing_returns_true():
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    result = await mgr.touch_session(session.id)
    assert result is True


async def test_touch_session_nonexistent_returns_false():
    mgr, _, _ = _make_manager()
    result = await mgr.touch_session("ghost-session-id")
    assert result is False


# ---------------------------------------------------------------------------
# update_session: session-not-found returns None
# ---------------------------------------------------------------------------


async def test_update_session_nonexistent_returns_none():
    mgr, _, _ = _make_manager()
    result = await mgr.update_session("nonexistent-id", data={"x": 1})
    assert result is None


# ---------------------------------------------------------------------------
# resume_session: ended status returns None
# ---------------------------------------------------------------------------


async def test_resume_session_ended_status_returns_none():
    mgr, _, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.end_session(session.id)

    result = await mgr.resume_session(session.id, user_id="u1", bot_id="b1")
    assert result is None


# ---------------------------------------------------------------------------
# Exception paths
# ---------------------------------------------------------------------------


async def test_persist_resume_to_db_exception_does_not_raise():
    """If upsert_resume_data raises, create_session should still succeed
    (warning is logged, not re-raised)."""
    redis = FakeRedisClient()

    class BrokenRepo(FakeSessionRepository):
        async def upsert_resume_data(self, **kwargs):  # type: ignore[override]
            msg = "DB down"
            raise RuntimeError(msg)

    repo = BrokenRepo()
    mgr = SessionManager(
        session_repo=repo,
        redis_client=redis,
        idle_timeout=timedelta(minutes=30),
    )

    # Should NOT raise despite broken repo
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    assert session.id is not None


async def test_resume_from_db_exception_returns_none():
    """If get_session_for_resume raises, _resume_from_db returns None gracefully."""
    redis = FakeRedisClient()

    class BrokenRepo(FakeSessionRepository):
        async def get_session_for_resume(self, **kwargs):  # type: ignore[override]
            msg = "DB unavailable"
            raise RuntimeError(msg)

    repo = BrokenRepo()
    mgr = SessionManager(
        session_repo=repo,
        redis_client=redis,
        idle_timeout=timedelta(minutes=30),
    )

    # resume_session with a token will call _resume_from_db on Redis miss
    result = await mgr.resume_session("any-token", user_id="u1", bot_id="b1")
    assert result is None
