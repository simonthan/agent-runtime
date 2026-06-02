"""Structural protocols for session subpackage dependencies (lift-ready).

Defined as typing.Protocol so concrete classes satisfy them without
inheritance. Implementations MUST scope all queries by user_oid + bot_id.
T-512 ownership-fix landed in v0.5.0; both the Redis layer (via
SessionManager) and the SQL layer (via this Protocol's concrete impl)
scope by user_oid + bot_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from agent_runtime.session.models import ResumeRow


class RedisClientProtocol(Protocol):
    """Minimal Redis surface used by SessionManager."""

    async def get(self, key: str) -> str | None: ...

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None: ...  # redis-py returns None when nx=True and key exists

    async def delete(self, *keys: str) -> int: ...

    async def expire(self, key: str, seconds: int) -> bool: ...

    async def incr(self, key: str) -> int: ...

    async def decr(self, key: str) -> int: ...


class SessionRepositoryProtocol(Protocol):
    """Minimal session-persistence surface used by SessionManager.

    Implementations MUST scope all queries by user_oid + bot_id. T-512
    ownership-fix landed in v0.5.0; both the Redis layer (via SessionManager)
    and the SQL layer (via this Protocol's concrete impl) scope by
    user_oid + bot_id.
    """

    async def upsert_resume_data(
        self,
        *,
        session_id: str,
        user_id: str,
        bot_id: str,
        resume_token: str,
        resume_expires_at: datetime,
        client_context: dict[str, Any] | None = None,
        channel: str = "teams",
    ) -> Any: ...

    async def get_session_for_resume(
        self,
        *,
        resume_token: str,
        user_id: str,
        bot_id: str,
    ) -> ResumeRow | None:
        """Return ResumeRow for the given resume token scoped to (user_id, bot_id).

        Returns ``None`` when the token is expired, unknown, or belongs to a
        different (user_id, bot_id) pair.
        """
        ...

    async def get_active_session(
        self,
        *,
        user_id: str,
        bot_id: str,
    ) -> ResumeRow | None:
        """Return the active session row for (user_id, bot_id), or None.

        Implementations MUST filter rows where status = 'active' AND
        last_message_at > now() - idle_timeout. SessionManager relies on this
        freshness invariant — it does NOT clean up hard-expired rows.

        The concrete implementation receives ``idle_timeout`` (e.g. as a
        constructor arg) to apply the window in SQL, avoiding stale
        ``status='active'`` rows surfacing after the idle window expires.
        """
        ...
