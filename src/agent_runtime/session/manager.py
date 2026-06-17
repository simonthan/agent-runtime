"""Redis-backed session state management with DB fallback for resume.

Lifted from ithelpdesk ``app.core.session_manager`` with these changes:
- (user_id, bot_id) keying replacing (user_id, intent) per ARCH §4 #4
- ``idle_timeout: timedelta`` replaces ``ttl: int``
- ``key_prefix`` instance arg replacing class-level PREFIX constants
- Atomic SET NX duplicate-create rejection (D4b)
- Atomic SET XX lease extension on resume (D4a)
- ``get_or_prompt_resume`` → ``ResumeDecision`` sealed union (D5a/b/c)
- ``touch_session`` atomic heartbeat
- ``_active_index_key`` reverse-index for (user, bot) → session_id
- ``save_feedback`` dropped (future ``FeedbackService`` per ARCH §2)
- consumer model filter dropped from ``update_session`` (raw dict pass-through)
- ``AuditLogger`` injected (``NullAuditLogger`` default)
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agent_runtime.logging import AuditLogger, NullAuditLogger
from agent_runtime.session.events import (
    Active,
    NewSession,
    Resumable,
    ResumeDecision,
    SessionAlreadyActive,
)
from agent_runtime.session.models import ResumeRow, SessionData

if TYPE_CHECKING:
    from agent_runtime.session.protocol import RedisClientProtocol, SessionRepositoryProtocol


def _utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(UTC)


def _ensure_utc(dt: datetime) -> datetime:
    # Defensive coercion at the ResumeRow boundary: asyncpg's default driver
    # returns naive UTC datetimes from TIMESTAMP columns. Naive values would
    # raise TypeError in any later `_utc_now() - dt` age comparison.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class SessionManager:
    """Manage conversation sessions with Redis backing and DB fallback for resume."""

    def __init__(
        self,
        *,
        session_repo: SessionRepositoryProtocol,
        redis_client: RedisClientProtocol,
        idle_timeout: timedelta = timedelta(minutes=30),
        key_prefix: str = "session",
        logger: AuditLogger | None = None,
    ) -> None:
        """Initialise SessionManager with injected dependencies.

        Args:
            session_repo: Session persistence repository (satisfies
                SessionRepositoryProtocol). Required — no fallback exists.
            redis_client: Redis client (satisfies RedisClientProtocol).
            idle_timeout: How long a session remains active without a message.
                Defaults to 30 minutes.
            key_prefix: Redis key namespace.  Defaults to ``"session"``.
            logger: Optional AuditLogger.  Defaults to ``NullAuditLogger()``.
        """
        self._session_repo = session_repo
        self.redis = redis_client
        self._idle_timeout = idle_timeout
        self._ttl_seconds = int(idle_timeout.total_seconds())
        self._prefix = key_prefix
        self._log: AuditLogger = logger or NullAuditLogger()

    # ------------------------------------------------------------------
    # Key builders
    # ------------------------------------------------------------------

    def _session_key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    def _resume_key(self, user_id: str, token: str) -> str:
        """OID-scoped resume token — prevents cross-user token reuse (T-512)."""
        return f"{self._prefix}:resume:{user_id}:{token}"

    def _active_index_key(self, user_id: str, bot_id: str) -> str:
        """Reverse index: (user_id, bot_id) → session_id."""
        return f"{self._prefix}:active:{user_id}:{bot_id}"

    def _count_key(self, user_id: str) -> str:
        return f"{self._prefix}:count:{user_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        user_id: str,
        bot_id: str,
        initial_context: dict[str, Any] | None = None,
        client_context: dict[str, Any] | None = None,
        channel: str = "teams",
    ) -> SessionData:
        """Create a new conversation session.

        Raises:
            ValueError: if ``user_id`` or ``bot_id`` is empty/whitespace.
            SessionAlreadyActive: if (user_id, bot_id) already has an active
                session within the idle window.
        """
        if not user_id or not user_id.strip():
            msg = "user_id required"
            raise ValueError(msg)
        if not bot_id or not bot_id.strip():
            msg = "bot_id required"
            raise ValueError(msg)

        session_id = str(uuid4())
        resume_token = secrets.token_urlsafe(24)
        now = _utc_now()
        resume_expires = now + self._idle_timeout

        session = SessionData(
            id=session_id,
            user_id=user_id,
            bot_id=bot_id,
            created_at=now,
            updated_at=now,
            data=initial_context or {},
            status="active",
            conversation_history=[],
            client_context=client_context or {},
        )

        # Atomic claim: SET NX prevents duplicate-session race (D4b)
        claimed = await self.redis.set(
            self._active_index_key(user_id, bot_id),
            session_id,
            ex=self._ttl_seconds,
            nx=True,
        )

        if not claimed:
            # Another session won the race — surface the existing session.
            decision = await self.get_or_prompt_resume(user_id=user_id, bot_id=bot_id)
            match decision:
                case Active(session_id=existing_sid):
                    existing = await self.get_session(existing_sid)
                    last_ts = existing.updated_at if existing else _utc_now()
                    raise SessionAlreadyActive(session_id=existing_sid, last_activity_ts=last_ts)
                case Resumable(session_id=existing_sid, last_activity_ts=last_ts):
                    raise SessionAlreadyActive(session_id=existing_sid, last_activity_ts=last_ts)
                case NewSession():
                    # Shouldn't happen here, but treat defensively as a race loss
                    raise SessionAlreadyActive(session_id=session_id, last_activity_ts=_utc_now())

        await self._save_session(session)

        # Increment per-user active session counter (O(1) vs O(N) scan)
        count_key = self._count_key(user_id)
        await self.redis.incr(count_key)
        await self.redis.expire(count_key, self._ttl_seconds)

        # Store resume token mapping in Redis (OID-scoped per T-512)
        await self.redis.set(
            self._resume_key(user_id, resume_token),
            session_id,
            ex=self._ttl_seconds,
        )

        # Persist resume token and client context to DB for durability
        await self._persist_resume_to_db(
            session_id,
            user_id,
            bot_id,
            resume_token,
            resume_expires,
            client_context,
            channel,
        )

        self._log.info("Session created", session_id=session_id, user_id=user_id)
        return session

    async def get_session(self, session_id: str) -> SessionData | None:
        """Retrieve a session by ID from Redis."""
        key = self._session_key(session_id)
        data = await self.redis.get(key)
        if not data:
            return None
        return self._deserialize_session(data)

    async def update_session(
        self,
        conversation_id: str,
        *,
        data: dict[str, Any] | None = None,
        add_message: dict | None = None,
        replace_data: bool = False,
    ) -> SessionData | None:
        """Update session data or add a message to history.

        ``data`` is stored as-is (raw dict).  The consumer model filter
        has been dropped; consumers that need validation should apply it
        upstream before calling this method.
        """
        session = await self.get_session(conversation_id)
        if not session:
            return None

        if data is not None:
            if replace_data:
                session.data = data
            else:
                session.data.update(data)

        if add_message:
            session.conversation_history.append(
                {
                    **add_message,
                    "timestamp": _utc_now().isoformat(),
                }
            )

        session.updated_at = _utc_now()
        await self._save_session(session)
        return session

    async def end_session(self, session_id: str) -> bool:
        """Mark a session as ended, clearing the active-index key."""
        session = await self.get_session(session_id)
        if not session:
            return False

        session.status = "ended"
        session.updated_at = _utc_now()
        await self._save_session(session)

        # Clear the reverse index so create_session may claim it again
        await self.redis.delete(self._active_index_key(session.user_id, session.bot_id))

        # Decrement per-user active session counter
        count_key = self._count_key(session.user_id)
        count = await self.redis.decr(count_key)
        if count is not None and count <= 0:
            await self.redis.delete(count_key)

        self._log.info("Session ended", session_id=session_id)
        return True

    async def resume_session(
        self,
        session_id_or_token: str,
        *,
        user_id: str,
        bot_id: str,
    ) -> SessionData | None:
        """Resume a session if it exists, belongs to the (user_id, bot_id) pair,
        and hasn't expired.

        Tries Redis first via token lookup or direct key, then falls back to DB
        if Redis data is missing.  Atomic SET XX re-extends the lease (D4a).

        Args:
            session_id_or_token: The session ID or resume token to resume.
            user_id: Owning user (for ownership validation).
            bot_id: Owning bot (for ownership validation — new dimension).
        """
        # Check if this is a resume token (OID-scoped per T-512)
        mapped_session_id = await self.redis.get(self._resume_key(user_id, session_id_or_token))
        actual_session_id = mapped_session_id or session_id_or_token

        session = await self.get_session(actual_session_id)

        # Fallback to DB if Redis miss
        if not session:
            session = await self._resume_from_db(session_id_or_token, user_id, bot_id)
            if session:
                # Re-cache in Redis
                await self._save_session(session)

        if not session:
            return None

        # Verify ownership to prevent session hijacking
        if session.user_id != user_id:
            return None

        # New dimension — reject if bot differs
        if session.bot_id != bot_id:
            return None

        if session.status == "ended":
            return None

        # Check if session is within resume window
        last_activity = session.updated_at or session.created_at
        age = (_utc_now() - last_activity).total_seconds()
        if age > self._ttl_seconds:
            return None

        session.status = "active"
        session.updated_at = _utc_now()

        # Atomic lease extension (D4a): SET ... EX ttl XX — only if key still exists
        serialized = self._serialize_session(session)
        extended = await self.redis.set(
            self._session_key(actual_session_id),
            serialized,
            ex=self._ttl_seconds,
            xx=True,
        )
        if not extended:
            # Key evicted between get_session and SET; rehydrate from DB
            session = await self._resume_from_db(session_id_or_token, user_id, bot_id)
            if session:
                await self._save_session(session)

        return session

    async def touch_session(self, session_id: str) -> bool:
        """Atomically extend session lease.

        Returns ``True`` if the key was extended, ``False`` if it had already
        been evicted (caller should treat the session as gone).
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        serialized = self._serialize_session(session)
        result = await self.redis.set(
            self._session_key(session_id),
            serialized,
            ex=self._ttl_seconds,
            xx=True,
        )
        return bool(result)

    async def get_or_prompt_resume(
        self,
        *,
        user_id: str,
        bot_id: str,
    ) -> ResumeDecision:
        """Determine the resume state for (user_id, bot_id).

        Returns:
            ``Active(session_id)`` — live session; lease atomically extended.
            ``Resumable(session_id, last_activity_ts)`` — session idle or not in
                Redis hot cache; caller should present the Resume card.
            ``NewSession()`` — no prior session; caller should create a fresh one.

        Raises:
            ValueError: if ``user_id`` or ``bot_id`` is empty/whitespace.
        """
        if not user_id or not user_id.strip():
            msg = "user_id required"
            raise ValueError(msg)
        if not bot_id or not bot_id.strip():
            msg = "bot_id required"
            raise ValueError(msg)

        # -- 1. Hot path: Redis reverse index ---------------------------------
        session_id = await self.redis.get(self._active_index_key(user_id, bot_id))

        if session_id:
            session = await self.get_session(session_id)

            if session is None:
                # Dangling reverse-index pointer (crash mid-create_session, or
                # session key evicted without index key expiry).  Delete the
                # dangler so the caller's next create_session can claim cleanly.
                await self.redis.delete(self._active_index_key(user_id, bot_id))
                return NewSession()

            if session.status == "active":
                age = _utc_now() - session.updated_at
                if age < self._idle_timeout:
                    # Within window — atomically extend lease (D5b)
                    serialized = self._serialize_session(session)
                    ok = await self.redis.set(
                        self._session_key(session_id),
                        serialized,
                        ex=self._ttl_seconds,
                        xx=True,
                    )
                    if ok:
                        return Active(session_id=session_id)
                    # XX-set failed → key evicted between get_session and SET.
                    # Fall through to Resumable — do NOT do a fresh DB lookup
                    # that could return a different concurrent session
                    # (Opus review finding).
                # Lapsed or XX-set failed — surface as Resumable
                return Resumable(
                    session_id=session_id,
                    last_activity_ts=session.updated_at,
                )

        # -- 2. Cold-cache rehydration (D5c) ----------------------------------
        # Protocol contract: get_active_session MUST filter rows where
        # status='active' AND last_message_at > now() - idle_timeout.
        row = await self._session_repo.get_active_session(user_id=user_id, bot_id=bot_id)
        if row is None:
            return NewSession()

        last = _ensure_utc(row.last_message_at or row.created_at)
        session = SessionData(
            id=str(row.id),
            user_id=row.user_id,
            bot_id=row.bot_id,
            created_at=_ensure_utc(row.created_at),
            updated_at=last,
            status="active",
            client_context=row.client_context,
        )
        await self._save_session(session)
        await self.redis.set(
            self._active_index_key(user_id, bot_id),
            str(row.id),
            ex=self._ttl_seconds,
        )
        return Resumable(session_id=str(row.id), last_activity_ts=last)

    async def get_active_sessions_count(self, user_id: str) -> int:
        """Get count of active sessions for a user via O(1) counter key."""
        count_key = self._count_key(user_id)
        value = await self.redis.get(count_key)
        return max(0, int(value)) if value is not None else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _persist_resume_to_db(
        self,
        session_id: str,
        user_id: str,
        bot_id: str,
        resume_token: str,
        resume_expires_at: datetime,
        client_context: dict[str, Any] | None = None,
        channel: str = "teams",
    ) -> None:
        """Persist resume token and client context via injected repository."""
        try:
            await self._session_repo.upsert_resume_data(
                session_id=session_id,
                user_id=user_id,
                bot_id=bot_id,
                resume_token=resume_token,
                resume_expires_at=resume_expires_at,
                client_context=client_context,
                channel=channel,
            )
        except Exception as e:  # noqa: BLE001
            self._log.warning("Failed to persist resume token via repo", error=str(e))

    async def _resume_from_db(
        self,
        resume_token: str,
        user_id: str,
        bot_id: str,
    ) -> SessionData | None:
        """Resume a session from durable storage via injected repository.

        ``SessionData.updated_at`` is populated from the real last-activity
        timestamp (``ResumeRow.last_message_at``) supplied by the repo, falling
        back to ``created_at`` for sessions that have not yet exchanged a
        message.  This ensures the TTL window check in ``resume_session`` ages
        the session from real user activity, not from session creation time.
        """
        try:
            row: ResumeRow | None = await self._session_repo.get_session_for_resume(
                resume_token=resume_token,
                user_id=user_id,
                bot_id=bot_id,
            )
            if not row:
                return None
            return SessionData(
                id=str(row.id),
                user_id=row.user_id,
                bot_id=row.bot_id,
                created_at=_ensure_utc(row.created_at),
                updated_at=_ensure_utc(row.last_message_at or row.created_at),
                data={},
                status=row.status,
                conversation_history=[],
                client_context=row.client_context,
            )
        except Exception as e:  # noqa: BLE001
            self._log.warning("Failed to resume from DB", error=str(e))
            return None

    async def _save_session(self, session: SessionData) -> None:
        """Save session to Redis."""
        key = self._session_key(session.id)
        data = self._serialize_session(session)
        await self.redis.set(key, data, ex=self._ttl_seconds)

    def _serialize_session(self, session: SessionData) -> str:
        """Serialize session to JSON."""
        return json.dumps(
            {
                "id": session.id,
                "user_id": session.user_id,
                "bot_id": session.bot_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "data": session.data,
                "status": session.status,
                "conversation_history": session.conversation_history,
                "client_context": session.client_context,
            }
        )

    def _deserialize_session(self, data: str) -> SessionData:
        """Deserialize session from JSON."""
        parsed = json.loads(data)
        return SessionData(
            id=parsed["id"],
            user_id=parsed["user_id"],
            bot_id=parsed["bot_id"],
            created_at=datetime.fromisoformat(parsed["created_at"]),
            updated_at=datetime.fromisoformat(parsed["updated_at"]),
            data=parsed["data"],
            status=parsed["status"],
            conversation_history=parsed["conversation_history"],
            client_context=parsed.get("client_context", {}),
        )
