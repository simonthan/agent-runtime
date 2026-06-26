"""Test doubles for ``agent_runtime.session``. Import from this module in
consumer tests rather than mocking the Protocols by hand.

Convention matches ``agent_runtime.transport.teams.testing``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from agent_runtime.session.models import ResumeRow, SessionData, SessionSummaryRow


@dataclass
class FakeRedisClient:
    """In-memory dict-backed RedisClientProtocol implementation.

    Single store mirrors real Redis: ``incr``/``decr`` read/write the
    same keyspace as ``get``/``set`` (integer-strings). This means
    ``get_active_sessions_count`` exercises the same dict that the
    counter is written to — without this, the counter increments
    would be invisible to ``get`` and the test would silently pass
    with count=0 (Sonnet review).

    Honors ``nx`` and ``xx`` flags on ``set`` so atomic lease semantics
    are exercised.
    """

    _store: dict[str, str] = field(default_factory=dict)

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,  # noqa: ARG002
        px: int | None = None,  # noqa: ARG002
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None:
        if nx and key in self._store:
            return None
        if xx and key not in self._store:
            return None
        self._store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if self._store.pop(k, None) is not None:
                n += 1
        return n

    async def expire(self, key: str, seconds: int) -> bool:  # noqa: ARG002
        return key in self._store

    async def incr(self, key: str) -> int:
        current = int(self._store.get(key, "0"))
        new = current + 1
        self._store[key] = str(new)
        return new

    async def decr(self, key: str) -> int:
        current = int(self._store.get(key, "0"))
        new = current - 1
        self._store[key] = str(new)
        return new


@dataclass
class FakeSessionRepository:
    """In-memory SessionRepositoryProtocol implementation.

    Honors the Protocol's freshness contract: ``get_active_session`` filters
    rows where ``last_message_at > now() - idle_timeout`` (and treats
    ``status != 'active'`` as a miss). Without this, the fake would silently
    return stale rows that a correct production impl filters at the SQL
    layer, hiding consumer bugs that rely on the freshness invariant.
    """

    _by_id: dict[str, ResumeRow] = field(default_factory=dict)
    _by_token: dict[str, str] = field(default_factory=dict)
    _active_by_pair: dict[tuple[str, str], str] = field(default_factory=dict)
    idle_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    supports_durable_history: bool = True
    _messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    async def upsert_resume_data(
        self,
        *,
        session_id: str,
        user_id: str,
        bot_id: str,
        resume_token: str,
        resume_expires_at: datetime,  # noqa: ARG002
        client_context: dict[str, Any] | None = None,
        channel: str = "teams",  # noqa: ARG002
    ) -> None:
        """Persist a ResumeRow projection for resume lookups."""
        row = ResumeRow(
            id=UUID(session_id),
            user_id=user_id,
            bot_id=bot_id,
            created_at=datetime.now(UTC),
            client_context=client_context or {},
        )
        self._by_id[session_id] = row
        self._by_token[resume_token] = session_id
        self._active_by_pair[(user_id, bot_id)] = session_id

    async def get_session_for_resume(
        self,
        *,
        resume_token: str,
        user_id: str,
        bot_id: str,
    ) -> ResumeRow | None:
        sid = self._by_token.get(resume_token)
        if sid is None:
            return None
        row = self._by_id.get(sid)
        if row is None or row.user_id != user_id or row.bot_id != bot_id:
            return None
        return row

    async def get_active_session(
        self,
        *,
        user_id: str,
        bot_id: str,
    ) -> ResumeRow | None:
        sid = self._active_by_pair.get((user_id, bot_id))
        if sid is None:
            return None
        row = self._by_id.get(sid)
        if row is None or row.status != "active":
            return None
        last = row.last_message_at or row.created_at
        if (datetime.now(UTC) - last) > self.idle_timeout:
            return None
        return row

    # --- DurableHistoryRepository (T-036) ---------------------------------

    async def append_message(
        self,
        *,
        session_id: str,
        user_id: str,  # noqa: ARG002
        bot_id: str,  # noqa: ARG002
        message: dict[str, Any],
    ) -> None:
        self._messages.setdefault(session_id, []).append(message)

    async def get_conversation_history(
        self,
        *,
        session_id: str,
        user_id: str,
        bot_id: str,
    ) -> list[dict[str, Any]]:
        # Ownership guard: _by_id is populated by upsert_resume_data (called from
        # create_session); a row absent or owned by a different (user,bot) → [].
        row = self._by_id.get(session_id)
        if row is None or row.user_id != user_id or row.bot_id != bot_id:
            return []
        return list(self._messages.get(session_id, []))

    async def list_sessions(
        self,
        *,
        user_id: str,
        bot_id: str,
        limit: int,
        before: datetime | None = None,
    ) -> list[SessionSummaryRow]:
        def _sort_key(r: ResumeRow) -> datetime:
            return r.last_message_at or r.created_at

        # NB: the Fake does not mirror end_session's status mutation (that lives on the
        # Redis SessionData, not _by_id) — the history view intentionally lists ALL of a
        # (user,bot)'s sessions regardless of status, so this is by design.
        rows = [r for r in self._by_id.values() if r.user_id == user_id and r.bot_id == bot_id]
        rows.sort(key=_sort_key, reverse=True)
        if before is not None:
            rows = [r for r in rows if _sort_key(r) < before]
        return [
            SessionSummaryRow(
                id=r.id,
                title=None,
                status=r.status,
                created_at=r.created_at,
                last_message_at=r.last_message_at,
                message_count=len(self._messages.get(str(r.id), [])),
            )
            for r in rows[:limit]
        ]


def make_session_data(
    *,
    user_id: str = "u1",
    bot_id: str = "b1",
    **overrides: Any,
) -> SessionData:
    """Factory for ``SessionData`` with sensible defaults."""
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "id": str(uuid4()),
        "user_id": user_id,
        "bot_id": bot_id,
        "created_at": now,
        "updated_at": now,
    }
    return SessionData(**{**base, **overrides})
