"""ResumeDecision sealed union for T-008f Resume-card UX.

Pattern-match via ``match decision: case NewSession() | Resumable(...) | Active(...)``.
SessionAlreadyActive is raised by SessionManager.create_session when (user, bot)
already has an Active session within the idle window per ARCH §4 #4.
"""

# Note: `from __future__ import annotations` is intentionally omitted here.
# Frozen dataclasses use field defaults evaluated at class-creation time;
# TC003 (move datetime into TYPE_CHECKING) would cause NameError at runtime.
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True, slots=True)
class NewSession:
    """No existing session — caller should create a fresh one."""

    kind: Literal["new"] = "new"


@dataclass(frozen=True, slots=True)
class Resumable:
    """Session exists but is idle or not in Redis hot cache — caller should
    present Resume card."""

    session_id: str
    last_activity_ts: datetime
    kind: Literal["resumable"] = "resumable"


@dataclass(frozen=True, slots=True)
class Active:
    """Session is live within the idle window — caller can proceed directly."""

    session_id: str
    kind: Literal["active"] = "active"


ResumeDecision = NewSession | Resumable | Active


class SessionAlreadyActive(Exception):  # noqa: N818
    """Raised by SessionManager.create_session when (user_id, bot_id) already has
    an Active session within the idle window. Dispatcher pattern-matches this to
    surface the Resume card UX. Named without 'Error' suffix to communicate that
    it is a flow-control signal (not a fault), matching the plan's public API."""

    def __init__(self, session_id: str, last_activity_ts: datetime) -> None:
        super().__init__(f"Session {session_id} already active")
        self.session_id = session_id
        self.last_activity_ts = last_activity_ts
