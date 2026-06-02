"""T-494 invariant: Protocol conformance + static import guard for manager.py."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from agent_runtime.session.testing import FakeRedisClient, FakeSessionRepository


# ---------------------------------------------------------------------------
# FakeRedisClient structurally satisfies RedisClientProtocol
# ---------------------------------------------------------------------------


async def test_fake_redis_client_satisfies_protocol():
    """Exercise every method on FakeRedisClient to confirm structural conformance."""
    r = FakeRedisClient()

    # set + get
    result = await r.set("k", "v", ex=60)
    assert result is True
    val = await r.get("k")
    assert val == "v"

    # nx (set-if-not-exists)
    result_nx = await r.set("k", "v2", nx=True)
    assert result_nx is None  # key already exists
    assert await r.get("k") == "v"  # unchanged

    # xx (set-if-exists)
    result_xx = await r.set("k", "v3", xx=True)
    assert result_xx is True
    assert await r.get("k") == "v3"

    # xx on nonexistent key
    result_xx_miss = await r.set("nokey", "v", xx=True)
    assert result_xx_miss is None

    # incr + decr
    await r.set("counter", "0")
    v1 = await r.incr("counter")
    assert v1 == 1
    v2 = await r.incr("counter")
    assert v2 == 2
    v3 = await r.decr("counter")
    assert v3 == 1

    # expire
    exists = await r.expire("k", 30)
    assert exists is True
    missing = await r.expire("nonexistent", 30)
    assert missing is False

    # delete (single)
    deleted = await r.delete("k")
    assert deleted == 1
    assert await r.get("k") is None

    # delete (multiple)
    await r.set("a", "1")
    await r.set("b", "2")
    deleted_multi = await r.delete("a", "b", "nonexistent")
    assert deleted_multi == 2


async def test_fake_redis_incr_starts_from_zero_when_key_absent():
    r = FakeRedisClient()
    v = await r.incr("new_counter")
    assert v == 1
    assert await r.get("new_counter") == "1"


async def test_fake_redis_single_store_counter_visible_to_get():
    """Counter written by incr must be readable by get — single store invariant."""
    r = FakeRedisClient()
    await r.incr("cnt")
    await r.incr("cnt")
    raw = await r.get("cnt")
    assert raw == "2"


# ---------------------------------------------------------------------------
# FakeSessionRepository structurally satisfies SessionRepositoryProtocol
# ---------------------------------------------------------------------------


async def test_fake_session_repository_satisfies_protocol():
    """Exercise every method on FakeSessionRepository."""
    repo = FakeSessionRepository()
    now = datetime.now(UTC)
    sid = str(uuid4())

    # upsert_resume_data
    await repo.upsert_resume_data(
        session_id=sid,
        user_id="u1",
        bot_id="b1",
        resume_token="tok1",
        resume_expires_at=now + timedelta(minutes=30),
        client_context={"tenant": "t1"},
        channel="teams",
    )

    # get_session_for_resume — hit
    row = await repo.get_session_for_resume(
        resume_token="tok1", user_id="u1", bot_id="b1"
    )
    assert row is not None
    assert str(row.id) == sid
    assert row.bot_id == "b1"

    # get_session_for_resume — wrong bot_id
    row_wrong = await repo.get_session_for_resume(
        resume_token="tok1", user_id="u1", bot_id="wrong"
    )
    assert row_wrong is None

    # get_session_for_resume — wrong token
    row_none = await repo.get_session_for_resume(
        resume_token="bad-token", user_id="u1", bot_id="b1"
    )
    assert row_none is None

    # get_active_session — hit
    active = await repo.get_active_session(user_id="u1", bot_id="b1")
    assert active is not None
    assert str(active.id) == sid

    # get_active_session — miss
    miss = await repo.get_active_session(user_id="u1", bot_id="b99")
    assert miss is None


# ---------------------------------------------------------------------------
# Static import guard — T-494 invariant
# ---------------------------------------------------------------------------


def test_no_consumer_imports_in_manager():
    """manager.py must contain no references to ithelpdesk app module paths
    or concrete RedisClient construction (T-494 invariant)."""
    manager_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "agent_runtime"
        / "session"
        / "manager.py"
    )
    source = manager_path.read_text()

    forbidden = ["from app.", "RedisClient(", "ITSessionState", "app.config"]
    for fragment in forbidden:
        assert fragment not in source, (
            f"Forbidden import/reference found in manager.py: {fragment!r}"
        )
