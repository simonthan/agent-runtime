"""Conversation session management — Redis hot path, Postgres durable fallback.

Lift from ithelpdesk's ``app.core.session_*`` family. Public surface:
- ``SessionManager`` — orchestrator
- ``SessionData`` — runtime conversation snapshot
- ``ConversationState`` — Pydantic baseline (consumers subclass for domain fields)
- ``RedisClientProtocol``, ``SessionRepositoryProtocol`` — DI surfaces
- ``ResumeRow`` — repository payload (ORM-free)
- ``ResumeDecision`` = ``NewSession`` | ``Resumable`` | ``Active`` (sealed union)
- ``SessionAlreadyActive`` — raised by ``create_session`` per ARCH §4 #4 v1

Optional extras: ``[redis]`` (``redis>=5,<6``), ``[postgres]`` (``asyncpg>=0.29``).
Testing helpers in ``agent_runtime.session.testing``.
"""

from agent_runtime.session.events import (
    Active,
    NewSession,
    Resumable,
    ResumeDecision,
    SessionAlreadyActive,
)
from agent_runtime.session.manager import SessionManager
from agent_runtime.session.models import (
    ConversationState,
    ResumeRow,
    SessionData,
    SessionSummaryRow,
)
from agent_runtime.session.protocol import (
    DurableHistoryRepository,
    RedisClientProtocol,
    SessionRepositoryProtocol,
)

__all__ = [
    "Active",
    "ConversationState",
    "DurableHistoryRepository",
    "NewSession",
    "RedisClientProtocol",
    "Resumable",
    "ResumeDecision",
    "ResumeRow",
    "SessionAlreadyActive",
    "SessionData",
    "SessionManager",
    "SessionRepositoryProtocol",
    "SessionSummaryRow",
]
