"""Generic connector primitives for external systems.

Lifted from ithelpdesk during T-490a. See module docstring in ``base`` for usage.
"""

from agent_runtime.connectors.base import (
    NON_RETRYABLE_HTTP_STATUS_CODES,
    RETRYABLE_HTTP_STATUS_CODES,
    BaseConnector,
    ConnectorResult,
    RetryMixin,
    is_retryable_error,
    register_rate_limit,
    set_audit_logger,
)

__all__ = [
    "NON_RETRYABLE_HTTP_STATUS_CODES",
    "RETRYABLE_HTTP_STATUS_CODES",
    "BaseConnector",
    "ConnectorResult",
    "RetryMixin",
    "is_retryable_error",
    "register_rate_limit",
    "set_audit_logger",
]
