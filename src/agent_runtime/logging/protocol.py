"""AuditLogger Protocol and a NullAuditLogger no-op default.

agent_runtime does not own audit-logging infrastructure. Each consumer
brings its own implementation. The Protocol below describes the structural
shape agent_runtime's lifted modules expect; ithelpdesk's
`app.utils.audit_logger.AuditLogger` satisfies it natively.
"""

from __future__ import annotations

from typing import Any, Protocol


class AuditLogger(Protocol):
    """Structural Protocol for audit logging."""

    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def security(self, message: str, **kwargs: Any) -> None: ...
    def action(
        self,
        action: str,
        result: str,
        user_id: str | None = None,
        session_id: str | None = None,
        details: dict | None = None,
        **kwargs: Any,
    ) -> None: ...


class NullAuditLogger:
    """No-op AuditLogger. Used as default when no consumer logger supplied."""

    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def security(self, message: str, **kwargs: Any) -> None: ...
    def action(
        self,
        action: str,
        result: str,
        user_id: str | None = None,
        session_id: str | None = None,
        details: dict | None = None,
        **kwargs: Any,
    ) -> None: ...
