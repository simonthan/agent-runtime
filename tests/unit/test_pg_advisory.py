from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from agent_runtime.pg_advisory import advisory_lock_key, pg_advisory_single_flight

ACQUIRE = "pg_try_advisory_lock"
RELEASE = "pg_advisory_unlock"
SOME_KEY = 42


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _FakeConn:
    def __init__(self, *, acquired=True, raise_on_release=None):
        self.acquired = acquired
        self.raise_on_release = raise_on_release
        self.statements: list[str] = []
        self.parameters: list[object] = []  # the key must reach the statement
        self.invalidated = 0

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        self.statements.append(sql)
        self.parameters.append(parameters)
        if RELEASE in sql:
            if self.raise_on_release is not None:
                raise self.raise_on_release
            return _FakeResult(value=True)
        return _FakeResult(value=self.acquired)

    async def invalidate(self):
        self.invalidated += 1


class _FakeSession:
    def __init__(self, conn):
        self.conn = conn
        self.execution_options = None
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed += 1

    async def connection(self, *, execution_options=None):
        self.execution_options = execution_options
        return self.conn

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


class _FakeFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


async def test_yields_true_and_releases_when_acquired():
    conn = _FakeConn(acquired=True)
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    async with pg_advisory_single_flight(factory, key=SOME_KEY) as acquired:
        assert acquired is True

    assert len(conn.statements) == 2
    assert ACQUIRE in conn.statements[0]
    assert RELEASE in conn.statements[1]
    assert session.commits == 0
    assert session.rollbacks == 0
    assert conn.invalidated == 0
    assert conn.parameters == [{"k": SOME_KEY}, {"k": SOME_KEY}]


async def test_yields_false_and_never_releases_when_not_acquired():
    conn = _FakeConn(acquired=False)
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    async with pg_advisory_single_flight(factory, key=SOME_KEY) as acquired:
        assert acquired is False

    assert len(conn.statements) == 1
    assert conn.parameters == [{"k": SOME_KEY}]
    assert not any(RELEASE in s for s in conn.statements)


async def test_connection_taken_in_autocommit():
    conn = _FakeConn(acquired=True)
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    async with pg_advisory_single_flight(factory, key=SOME_KEY):
        assert session.execution_options == {"isolation_level": "AUTOCOMMIT"}


async def test_body_exception_still_releases_and_propagates():
    conn = _FakeConn(acquired=True)
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    class _BodyError(Exception):
        pass

    with pytest.raises(_BodyError):
        async with pg_advisory_single_flight(factory, key=SOME_KEY):
            raise _BodyError("boom")

    assert any(RELEASE in s for s in conn.statements)


async def test_cancelled_unlock_invalidates_connection_and_reraises():
    conn = _FakeConn(acquired=True, raise_on_release=asyncio.CancelledError())
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    with pytest.raises(asyncio.CancelledError):
        async with pg_advisory_single_flight(factory, key=SOME_KEY):
            pass

    assert conn.invalidated == 1


async def test_failed_unlock_exception_invalidates_and_reraises():
    conn = _FakeConn(acquired=True, raise_on_release=RuntimeError("boom"))
    session = _FakeSession(conn)
    factory = _FakeFactory(session)

    with pytest.raises(RuntimeError):
        async with pg_advisory_single_flight(factory, key=SOME_KEY):
            pass

    assert conn.invalidated == 1


def test_advisory_lock_key_is_stable_and_pinned():
    key_1 = advisory_lock_key("tbp.briefing.schedule", "sched-001")
    key_2 = advisory_lock_key("tbp.briefing.schedule", "sched-002")

    assert key_1 == -2183477855666063756
    assert key_2 == -2928978304013417867
    assert -(2**63) <= key_1 < 2**63


def test_advisory_lock_key_encoding_is_injective():
    assert advisory_lock_key("a", "b:c") != advisory_lock_key("a:b", "c")


def test_module_has_no_top_level_sqlalchemy_import():
    module_path = (
        pathlib.Path(__file__).parent.parent.parent / "src" / "agent_runtime" / "pg_advisory.py"
    )
    tree = ast.parse(module_path.read_text())

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "sqlalchemy" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "sqlalchemy" not in module
