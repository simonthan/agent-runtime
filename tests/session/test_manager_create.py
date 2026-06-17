"""Tests for SessionManager.create_session and related atomic-claim logic."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from agent_runtime.session.events import NewSession, SessionAlreadyActive
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
# Cold start
# ---------------------------------------------------------------------------


async def test_create_session_cold_start():
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    assert session.user_id == "u1"
    assert session.bot_id == "b1"
    assert session.status == "active"

    # Session written to Redis
    raw = await redis.get(f"session:{session.id}")
    assert raw is not None

    # Reverse index written
    idx = await redis.get("session:active:u1:b1")
    assert idx == session.id

    # Counter incremented
    count = await mgr.get_active_sessions_count("u1")
    assert count == 1

    # Resume token persisted to DB
    assert session.id in repo._by_id


async def test_create_session_initial_context():
    mgr, redis, _ = _make_manager()
    session = await mgr.create_session(
        user_id="u1",
        bot_id="b1",
        initial_context={"foo": "bar"},
        client_context={"tenant": "t1"},
    )
    assert session.data == {"foo": "bar"}
    assert session.client_context == {"tenant": "t1"}


# ---------------------------------------------------------------------------
# Duplicate rejection
# ---------------------------------------------------------------------------


async def test_create_session_duplicate_raises_already_active():
    mgr, _, _ = _make_manager()
    first = await mgr.create_session(user_id="u1", bot_id="b1")

    with pytest.raises(SessionAlreadyActive) as exc_info:
        await mgr.create_session(user_id="u1", bot_id="b1")

    exc = exc_info.value
    assert exc.session_id == first.id
    assert isinstance(exc.last_activity_ts, datetime)


async def test_create_session_different_bot_does_not_raise():
    """Same user, different bot_id — NOT a duplicate."""
    mgr, _, _ = _make_manager()
    s1 = await mgr.create_session(user_id="u1", bot_id="b1")
    s2 = await mgr.create_session(user_id="u1", bot_id="b2")  # different bot
    assert s1.id != s2.id
    assert s2.bot_id == "b2"


async def test_create_session_different_user_does_not_raise():
    """Different users for same bot_id are independent."""
    mgr, _, _ = _make_manager()
    s1 = await mgr.create_session(user_id="u1", bot_id="b1")
    s2 = await mgr.create_session(user_id="u2", bot_id="b1")
    assert s1.id != s2.id


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


async def test_create_session_empty_bot_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="bot_id required"):
        await mgr.create_session(user_id="u1", bot_id="")


async def test_create_session_whitespace_bot_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="bot_id required"):
        await mgr.create_session(user_id="u1", bot_id="   ")


async def test_create_session_empty_user_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="user_id required"):
        await mgr.create_session(user_id="", bot_id="b1")


async def test_create_session_whitespace_user_id_raises():
    mgr, _, _ = _make_manager()
    with pytest.raises(ValueError, match="user_id required"):
        await mgr.create_session(user_id="  ", bot_id="b1")


# ---------------------------------------------------------------------------
# Atomic SET NX claim test
# ---------------------------------------------------------------------------


async def test_create_session_atomic_set_nx_pre_seeded_raises():
    """Pre-seed the active index key to simulate a race-won state.
    create_session must raise SessionAlreadyActive WITHOUT writing a new
    session record (verify no extra {prefix}:{uuid} keys appeared)."""
    mgr, redis, repo = _make_manager(prefix="sess")

    # Create a legitimate first session
    first = await mgr.create_session(user_id="u1", bot_id="b1")

    # Keys before the second attempt
    session_data_keys_before = [
        k for k in redis._store if k.startswith("sess:") and k.count(":") == 1
    ]

    with pytest.raises(SessionAlreadyActive) as exc_info:
        await mgr.create_session(user_id="u1", bot_id="b1")

    # No new session-data keys
    session_data_keys_after = [
        k for k in redis._store if k.startswith("sess:") and k.count(":") == 1
    ]
    assert session_data_keys_before == session_data_keys_after

    # The exception references the EXISTING session
    assert exc_info.value.session_id == first.id


# ---------------------------------------------------------------------------
# Dangling-pointer test
# ---------------------------------------------------------------------------


async def test_get_or_prompt_resume_dangling_pointer_returns_new_session_and_deletes_key():
    """Pre-seed reverse index pointing to a nonexistent session_id.
    get_or_prompt_resume must return NewSession() AND delete the dangling key."""
    mgr, redis, _ = _make_manager()

    # Pre-seed a dangling pointer
    await redis.set("session:active:u1:b1", "nonexistent-session-id")

    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")

    assert isinstance(decision, NewSession)
    # Key must be deleted
    assert await redis.get("session:active:u1:b1") is None


async def test_pre_seed_reverse_index_session_resumable_raises():
    """When pre-seeded index points to an existing but idle session,
    create_session raises SessionAlreadyActive via Resumable match arm."""
    mgr, redis, _ = _make_manager(idle_timeout=timedelta(seconds=1))
    # Create a session normally
    first = await mgr.create_session(user_id="u1", bot_id="b1")

    # Backdate updated_at to make session appear lapsed (Resumable path)
    raw = redis._store[f"session:{first.id}"]
    parsed = json.loads(raw)
    old_ts = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    parsed["updated_at"] = old_ts
    redis._store[f"session:{first.id}"] = json.dumps(parsed)

    # Force NX to fail by keeping the index key (it still points to a valid-but-lapsed session)
    # create_session NX will fail, get_or_prompt_resume returns Resumable, raises SessionAlreadyActive
    with pytest.raises(SessionAlreadyActive) as exc_info:
        await mgr.create_session(user_id="u1", bot_id="b1")
    # Raised with the lapsed session's id
    assert exc_info.value.session_id == first.id


async def test_create_session_succeeds_after_dangling_pointer_cleared():
    """After get_or_prompt_resume deletes the dangling key, create_session
    can atomically claim it again."""
    mgr, redis, _ = _make_manager()

    # Seed dangling pointer
    await redis.set("session:active:u1:b1", "ghost-id")

    # Trigger dangling-pointer cleanup
    decision = await mgr.get_or_prompt_resume(user_id="u1", bot_id="b1")
    assert isinstance(decision, NewSession)

    # Now create_session should succeed (NX can claim the clean key)
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    assert session.user_id == "u1"
    assert session.bot_id == "b1"


async def test_create_session_nx_fails_get_or_prompt_returns_new_session_defensive():
    """Covers the defensive NewSession arm in create_session (race: NX fails but
    get_or_prompt_resume returns NewSession — shouldn't happen normally but the
    code path must be executable for coverage)."""
    mgr, redis, _ = _make_manager()

    # First create sets the index key
    await mgr.create_session(user_id="u1", bot_id="b1")

    # Now delete the session record but keep the index — dangling ptr
    # When NX fails AND get_or_prompt_resume self-heals to NewSession, the
    # defensive arm raises SessionAlreadyActive with a synthetic ts
    await redis.delete("session:active:u1:b1")
    # Pre-seed the index to point to a nonexistent session (dangling)
    await redis.set("session:active:u1:b1", "dangling-id")

    # NX will fail (index exists), get_or_prompt_resume sees dangling ptr,
    # deletes it, returns NewSession → defensive SessionAlreadyActive
    with pytest.raises(SessionAlreadyActive):
        await mgr.create_session(user_id="u1", bot_id="b1")
