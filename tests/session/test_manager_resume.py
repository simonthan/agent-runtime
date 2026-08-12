"""Tests for SessionManager.resume_session."""

import json
from datetime import UTC, datetime, timedelta

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
# Happy path — Redis hit + atomic XX lease extension
# ---------------------------------------------------------------------------


async def test_resume_session_happy_path():
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Retrieve the resume token from the repository
    token = list(repo._by_token.keys())[0]

    resumed = await mgr.resume_session(token, user_id="u1", bot_id="b1")
    assert resumed is not None
    assert resumed.id == session.id
    assert resumed.user_id == "u1"
    assert resumed.bot_id == "b1"
    assert resumed.status == "active"


async def test_resume_session_by_session_id_directly():
    """resume_session also accepts a session_id directly (not just token)."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    resumed = await mgr.resume_session(session.id, user_id="u1", bot_id="b1")
    assert resumed is not None
    assert resumed.id == session.id


async def test_resume_session_extends_lease_atomically():
    """After resume, the session key should still exist in Redis (XX extended)."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    token = list(repo._by_token.keys())[0]

    raw_before = await redis.get(f"session:{session.id}")
    await mgr.resume_session(token, user_id="u1", bot_id="b1")
    raw_after = await redis.get(f"session:{session.id}")

    # Key still exists and was re-written (updated_at changed)
    assert raw_after is not None
    assert raw_before != raw_after  # updated_at in JSON changed


# ---------------------------------------------------------------------------
# Cold-cache rehydration: Redis miss + DB hit
# ---------------------------------------------------------------------------


async def test_resume_session_cold_cache_rehydration():
    """Redis miss + DB hit: session is restored from repo."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    token = list(repo._by_token.keys())[0]

    # Evict from Redis
    await redis.delete(f"session:{session.id}")
    assert await redis.get(f"session:{session.id}") is None

    # resume_session should fall back to DB
    resumed = await mgr.resume_session(token, user_id="u1", bot_id="b1")
    assert resumed is not None
    assert resumed.id == session.id

    # Session re-cached in Redis
    assert await redis.get(f"session:{session.id}") is not None


# ---------------------------------------------------------------------------
# Ownership rejection
# ---------------------------------------------------------------------------


async def test_resume_session_wrong_user_id_returns_none():
    mgr, _, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Pass the actual session_id directly — bypasses token lookup
    # This exercises the session.user_id != user_id ownership guard
    result = await mgr.resume_session(session.id, user_id="wrong-user", bot_id="b1")
    assert result is None


async def test_resume_session_wrong_bot_id_returns_none():
    """New dimension: bot_id ownership check."""
    mgr, _, repo = _make_manager()
    await mgr.create_session(user_id="u1", bot_id="b1")
    token = list(repo._by_token.keys())[0]

    result = await mgr.resume_session(token, user_id="u1", bot_id="different-bot")
    assert result is None


# ---------------------------------------------------------------------------
# Hard-expired: past idle_timeout
# ---------------------------------------------------------------------------


async def test_resume_session_hard_expired_returns_none():
    """Session older than idle_timeout returns None."""
    # Set a 1-second timeout for testing
    mgr, redis, repo = _make_manager(idle_timeout=timedelta(seconds=1))
    session = await mgr.create_session(user_id="u1", bot_id="b1")

    # Manually backdate updated_at to simulate expiry
    raw = await redis.get(f"session:{session.id}")
    assert raw is not None
    parsed = json.loads(raw)
    # Push updated_at 2 seconds into the past
    old_ts = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    parsed["updated_at"] = old_ts
    redis._store[f"session:{session.id}"] = json.dumps(parsed)

    result = await mgr.resume_session(session.id, user_id="u1", bot_id="b1")
    assert result is None


# ---------------------------------------------------------------------------
# Eviction during resume — XX-set returns None → falls through to DB
# ---------------------------------------------------------------------------


async def test_resume_session_unknown_session_id_returns_none():
    """Session ID that doesn't exist in Redis or DB returns None (covers
    _resume_from_db returning None when DB has no row)."""
    mgr, _, _ = _make_manager()
    result = await mgr.resume_session("completely-unknown-id", user_id="u1", bot_id="b1")
    assert result is None


async def test_resume_session_eviction_during_resume_falls_through_to_db():
    """Simulate key being evicted between get_session and the XX SET.
    The session should be re-fetched from DB."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    token = list(repo._by_token.keys())[0]

    # Intercept the XX set by subclassing FakeRedisClient to return None once on xx=True
    original_set = redis.set

    set_call_count = 0

    async def patched_set(key, value, ex=None, px=None, nx=False, xx=False):
        nonlocal set_call_count
        if xx and key == f"session:{session.id}":
            set_call_count += 1
            if set_call_count == 1:
                # Simulate eviction: remove key and return None
                redis._store.pop(key, None)
                return None
        return await original_set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    redis.set = patched_set  # type: ignore[method-assign]

    resumed = await mgr.resume_session(token, user_id="u1", bot_id="b1")
    assert set_call_count >= 1
    # T-133: `assert x is not None or x is None` asserted nothing. The token DOES resolve
    # in the repo, so the DB fallback must produce the same session, marked active.
    assert resumed is not None
    assert resumed.id == session.id
    assert resumed.status == "active"


# ---------------------------------------------------------------------------
# T-133: id-addressed resume survives a lease eviction
# ---------------------------------------------------------------------------


async def test_resume_by_session_id_survives_lease_eviction():
    """T-133 (3): the XX-failure path fed the raw argument to a repo seam that treats it
    STRICTLY as a resume token, so an id-addressed resume was denied ("session expired")
    even though a valid, ownership-checked SessionData was in hand."""
    mgr, redis, _repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    await mgr.update_session(session.id, add_message={"role": "user", "content": "keep me"})

    original_set = redis.set
    evicted = False

    async def patched_set(key, value, ex=None, px=None, nx=False, xx=False):
        nonlocal evicted
        if xx and key == f"session:{session.id}" and not evicted:
            evicted = True
            redis._store.pop(key, None)
            return None
        return await original_set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    redis.set = patched_set  # type: ignore[method-assign]

    resumed = await mgr.resume_session(session.id, user_id="u1", bot_id="b1")

    assert evicted
    assert resumed is not None, "id-addressed resume denied after a lease eviction"
    assert resumed.id == session.id
    assert resumed.status == "active"
    assert [m["content"] for m in resumed.conversation_history] == ["keep me"]
    assert await redis.get(f"session:{session.id}") is not None  # re-saved


async def test_resume_eviction_does_not_resurrect_an_ended_durable_session():
    """The recovery path must not re-save a session the durable store has retired.

    This is the test that only means something with C3b: a production repo filters
    `status='active'` inside get_session_for_resume, so BOTH the id-addressed case and
    the genuinely-ended case return None from _resume_from_db. The recovery path must
    therefore not treat "DB said no" as "must be the id-addressed bug" — it confirms
    liveness via get_active_session, which also returns None here."""
    mgr, redis, repo = _make_manager()
    session = await mgr.create_session(user_id="u1", bot_id="b1")
    token = next(iter(repo._by_token))
    repo._by_id[session.id].status = "ended"  # ended durably; the Redis blob still says active

    original_set = redis.set

    async def patched_set(key, value, ex=None, px=None, nx=False, xx=False):
        if xx and key == f"session:{session.id}":
            redis._store.pop(key, None)
            return None
        return await original_set(key, value, ex=ex, px=px, nx=nx, xx=xx)

    redis.set = patched_set  # type: ignore[method-assign]

    assert await mgr.resume_session(token, user_id="u1", bot_id="b1") is None
