"""Postgres advisory-lock single-flight guard for cross-process scheduled work.

Deliberately dependency-free at import time and NOT under any existing subpackage.
The caller supplies a SQLAlchemy 2 async sessionmaker, so SQLAlchemy is necessarily
already installed in any process that can *call* this — but a consumer that merely
imports ``agent_runtime`` (or installs only ``[redis]`` / ``[teams]``) must not be made
to carry it. So: parameter types are structural Protocols rather than SQLAlchemy
imports, and the one symbol needed at runtime (``text``) is imported inside the guard.
An AST fence test (tests/unit/test_pg_advisory.py) pins that property.

Extracted verbatim from teams-bot-platform's
``NotificationEventRepository.digest_sweep_lock`` (TBP T-140), whose mechanics were
validated against a live Postgres 16 — including the pooled-connection still-locked
trap. Do not "simplify" the body; the invariants are enumerated on the guard below.
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

__all__ = ["advisory_lock_key", "pg_advisory_single_flight"]

# 8 bytes -> 64-bit signed, exactly Postgres' bigint advisory-key width.
_KEY_BYTES = 8


class AdvisoryLockConnection(Protocol):
    """The slice of SQLAlchemy's ``AsyncConnection`` this guard uses."""

    async def execute(self, statement: Any, parameters: Any = None) -> Any: ...

    async def invalidate(self) -> None: ...


class AdvisoryLockSession(Protocol):
    """The slice of SQLAlchemy's ``AsyncSession`` this guard uses (an async CM)."""

    async def connection(
        self, *, execution_options: Mapping[str, Any] | None = None
    ) -> AdvisoryLockConnection: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc_info: object) -> None: ...


class AdvisoryLockSessionFactory(Protocol):
    """A zero-arg callable returning a fresh session — e.g. ``async_sessionmaker``."""

    def __call__(self) -> AdvisoryLockSession: ...


def advisory_lock_key(namespace: str, ident: str) -> int:
    """Derive a STABLE 64-bit Postgres advisory-lock key from ``(namespace, ident)``.

    Stability across PROCESSES is the entire point: two workers exclude each other only
    if they compute the same key. ``hash()`` must never be used here — CPython salts str
    hashing with PYTHONHASHSEED, so two workers would derive DIFFERENT keys, both acquire,
    and both run, silently defeating the guard with no error anywhere. blake2b is seed-free
    and stable across processes, versions and machines.

    Returns a SIGNED value because Postgres' advisory-lock key is a signed bigint; the full
    64-bit range is used, which is why the pinned-value test exists (an algorithm change
    during a rolling deploy would leave old and new processes on different keys).

    The payload is LENGTH-PREFIXED, not separator-joined. A plain separator is not injective:
    with ``f"{namespace}\\x00{ident}"``, ("a", "b\\x00c") and ("a\\x00b", "c") hash the SAME
    bytes — i.e. two different features could collide by construction rather than by luck.
    Prefixing the namespace length makes the encoding unambiguous for every input.
    """
    payload = f"{len(namespace)}:{namespace}:{ident}".encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=_KEY_BYTES).digest(), "big", signed=True
    )


@asynccontextmanager
async def pg_advisory_single_flight(
    session_factory: AdvisoryLockSessionFactory, *, key: int
) -> AsyncIterator[bool]:
    """Single-flight guard. Yields True when THIS caller owns ``key``, False when another
    process already holds it (a False caller must then do nothing).

    MECHANICS ARE LOAD-BEARING (verified against Postgres 16 in TBP T-140):
      * The lock is SESSION-scoped, i.e. it lives on the CONNECTION, not the transaction.
      * The connection is taken in AUTOCOMMIT so no transaction is left open for the whole
        guarded operation, and this session MUST NOT commit or rollback: on commit
        SQLAlchemy returns the connection to the pool while it still holds the lock, which
        both leaks the lock and lets whoever is handed that pooled connection next
        re-acquire the same key REENTRANTLY — silently defeating the exclusion this guard
        exists to provide.
      * Release is therefore explicit, in a finally. Other sessions drawn from the same
        pool during the guarded operation do not disturb it.
      * CANCELLATION SAFETY: ``except BaseException``, not ``except Exception``. On task
        cancellation the unlock await re-raises CancelledError immediately and the unlock
        never reaches the server (schedulers shut down with ``wait=False``, which cancels
        in-flight jobs, so this is reachable). A connection returned to the pool still
        holding the lock would block every later run that draws a different connection —
        and be re-acquired REENTRANTLY by any run handed that same connection. Discard it.
    """
    from sqlalchemy import text  # noqa: PLC0415 — lazy on purpose, see module docstring

    async with session_factory() as s:
        conn = await s.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        acquired = bool(
            (await conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": key})).scalar_one()
        )
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
                except BaseException:
                    await conn.invalidate()
                    raise
