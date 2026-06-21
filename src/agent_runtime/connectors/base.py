"""Base connector interface for external systems."""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from agent_runtime.logging import AuditLogger, NullAuditLogger
from agent_runtime.safety import mask_string

_default_audit: AuditLogger = NullAuditLogger()

T = TypeVar("T")

# Errors that should be retried (transient failures)
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Errors that should NOT be retried (permanent failures)
NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 409, 422}

# Connector class name -> {"max_concurrent": int, "min_delay_seconds": float}.
# Populated by consumers via register_rate_limit() at import time.
# Use dict[str, dict[str, Any]] (NOT dict[str, dict[str, float]]) because values are
# heterogeneous: max_concurrent is int, min_delay_seconds is float.
_rate_limit_config: dict[str, dict[str, Any]] = {}


def register_rate_limit(
    class_name: str,
    max_concurrent: int,
    min_delay_seconds: float,
) -> None:
    """Register rate-limit config for a connector class.

    Call at module import time, before any connector of this class is instantiated.
    Idempotent — re-registering the same class_name overwrites the previous config.

    Args:
        class_name: ``type(connector).__name__`` — the class name used by
            ``BaseConnector.__init__`` to look up its throttle.
        max_concurrent: Maximum concurrent calls allowed for this class.
        min_delay_seconds: Minimum delay (seconds) between successive calls.
    """
    _rate_limit_config[class_name] = {
        "max_concurrent": int(max_concurrent),
        "min_delay_seconds": float(min_delay_seconds),
    }


def set_audit_logger(audit_logger: AuditLogger) -> None:
    """Override the module-level audit logger. Call once at consumer startup.

    Without this call, throttle/retry telemetry is routed through NullAuditLogger
    (silent). Consumers that want structured logging must wire their own
    AuditLogger Protocol implementation via this hook.
    """
    global _default_audit  # noqa: PLW0603 — module-level config hook by design
    _default_audit = audit_logger


class _ConnectorThrottle:
    """Rate limiter for a single connector class.

    Enforces max concurrency and minimum delay between calls.
    """

    def __init__(self, max_concurrent: int, min_delay_seconds: float):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._min_delay = min_delay_seconds
        self._last_call_time: float = 0.0
        self._delay_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        """Acquire throttle: wait for semaphore and enforce min delay."""
        async with self._semaphore:
            async with self._delay_lock:
                now = time.monotonic()
                elapsed = now - self._last_call_time
                if elapsed < self._min_delay:
                    await asyncio.sleep(self._min_delay - elapsed)
                self._last_call_time = time.monotonic()
            yield


# Singleton registry: one throttle per connector class name
_throttle_registry: dict[str, _ConnectorThrottle] = {}
_throttle_registry_lock = asyncio.Lock()


async def _get_throttle(connector_class_name: str) -> _ConnectorThrottle | None:
    """Get or create the throttle for a connector class."""
    config = _rate_limit_config.get(connector_class_name)
    if not config:
        return None

    async with _throttle_registry_lock:
        if connector_class_name not in _throttle_registry:
            _throttle_registry[connector_class_name] = _ConnectorThrottle(
                config["max_concurrent"],
                config["min_delay_seconds"],
            )
        return _throttle_registry[connector_class_name]


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable (transient) vs non-retryable (permanent)."""
    # Network/connection errors are retryable
    if isinstance(error, ConnectionError | TimeoutError | asyncio.TimeoutError):
        return True

    # httpx-specific errors
    if isinstance(error, httpx.TimeoutException):
        return True
    if isinstance(error, httpx.ConnectError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_HTTP_STATUS_CODES

    # OSError (network unreachable, etc.)
    if isinstance(error, OSError):
        return True

    # Non-retryable by default
    return False


@dataclass
class ConnectorResult:
    """Standard result from connector operations."""

    success: bool
    message: str
    data: dict[str, Any] | None = None
    error_code: str | None = None
    http_status: int | None = None


class BaseConnector(ABC):
    """Base class for all external system connectors."""

    def __init__(self):
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the connector (establish connections, validate credentials)."""
        pass

    @abstractmethod
    async def health_check(self) -> ConnectorResult:
        """Check if the external system is reachable and healthy."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources and close connections."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if connector is initialized."""
        return self._initialized

    def _handle_error(self, error: Exception, operation: str) -> ConnectorResult:
        """Standard error handling for connector operations.

        Returns a generic user-facing message — never exposes internal detail.
        Internal error info is preserved in data._internal_error for logging.

        SEC-2/SEC-3: driver/httpx exception text routinely embeds connection
        strings, bearer tokens, or PII. Both the audit line AND the
        ``_internal_error`` field are routed through ``mask_string`` so neither the
        consumer's audit sink nor a consumer that serializes ``.data`` to the
        channel leaks secrets — the "user sees .message, logs see .data" split is
        contract-only and unenforced, so we mask the value rather than rely on it.
        """
        masked_error = mask_string(str(error))
        _default_audit.error(f"Connector error during {operation}: {masked_error}")
        http_status = None
        if isinstance(error, httpx.HTTPStatusError):
            http_status = error.response.status_code
        return ConnectorResult(
            success=False,
            message="This service is temporarily unavailable. A support ticket will be created.",
            error_code=type(error).__name__,
            http_status=http_status,
            data={"_internal_error": f"Operation failed: {operation}: {masked_error}"},
        )


class RetryMixin:
    """Mixin for retry functionality with error classification."""

    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        min_wait: float = 1,
        max_wait: float = 10,
    ):
        """Decorator for operations that should retry on failure."""
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            reraise=True,
        )

    async def _execute_with_retry(
        self,
        operation: Callable,
        *args,
        max_attempts: int = 3,
        min_wait: float = 1.0,
        max_wait: float = 10.0,
        operation_name: str = "operation",
        **kwargs,
    ) -> Any:
        """Execute an async operation with exponential backoff retry on transient errors.

        Only retries on errors classified as retryable (network timeouts, 5xx, etc.).
        Non-retryable errors (4xx auth, not found) are raised immediately.
        Respects Retry-After headers on 429 responses.
        Applies per-connector rate limiting when configured.
        """
        last_error: Exception | None = None
        connector_class_name = type(self).__name__
        throttle = await _get_throttle(connector_class_name)

        for attempt in range(1, max_attempts + 1):
            try:
                if throttle:
                    async with throttle.acquire():
                        return await operation(*args, **kwargs)
                else:
                    return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e

                if not is_retryable_error(e):
                    # SEC-2: mask secrets/PII embedded in exception text before logging.
                    _default_audit.debug(f"{operation_name}: non-retryable error on attempt {attempt}: {mask_string(str(e))}")
                    raise

                if attempt == max_attempts:
                    # SEC-2: mask secrets/PII embedded in exception text before logging.
                    _default_audit.error(f"{operation_name}: failed after {max_attempts} attempts: {mask_string(str(e))}")
                    raise

                # Check for Retry-After header on 429 responses
                wait_time = min(min_wait * (2 ** (attempt - 1)), max_wait)
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (429, 503):
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = min(float(retry_after), max_wait)
                            _default_audit.warning(
                                f"{operation_name}: rate limited by {connector_class_name}, "
                                f"Retry-After: {retry_after}s"
                            )
                        except (ValueError, TypeError):
                            pass

                _default_audit.warning(
                    f"{operation_name}: attempt {attempt}/{max_attempts} failed "
                    f"({type(e).__name__}), retrying in {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)

        raise last_error  # Should not reach here, but safety net
