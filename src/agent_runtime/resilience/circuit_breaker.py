"""Circuit breaker pattern implementation for connector fault tolerance.

Provides automatic failure detection and recovery:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped after failures, requests fail fast
- HALF_OPEN: Testing recovery, limited requests allowed

Usage:
    breaker = CircuitBreaker("servicenow")

    async with breaker:
        result = await connector.some_operation()
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_runtime.logging import AuditLogger, NullAuditLogger

_default_audit: AuditLogger = NullAuditLogger()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Consecutive failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying recovery
    half_open_max_calls: int = 3  # Test calls allowed in half-open state
    success_threshold: int = 2  # Successes in half-open to close circuit


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_state_change: float
    total_failures: int
    total_successes: int


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is open. Retry after {retry_after:.1f} seconds.")


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures.

    State transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: "Callable[[str, CircuitState, CircuitState], Any] | None" = None,
        audit: AuditLogger | None = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._last_state_change = time.monotonic()
        self._total_failures = 0
        self._total_successes = 0
        self._lock = asyncio.Lock()
        self._on_state_change = on_state_change
        self._audit = audit if audit is not None else _default_audit

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _time_until_retry(self) -> float:
        """Get seconds until circuit can be retried."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        remaining = self.config.recovery_timeout - elapsed
        return max(0.0, remaining)

    def retry_after_seconds(self) -> float:
        """Seconds until the circuit will allow a probe call.

        Returns 0.0 when the circuit is CLOSED or HALF_OPEN (calls allowed
        immediately) or when the recovery_timeout has already elapsed for an
        OPEN circuit. Useful for structured log telemetry on the OPEN-state
        short-circuit path. See T-402c2.
        """
        if self._state != CircuitState.OPEN:
            return 0.0
        return self._time_until_retry()

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        self._audit.info(
            f"Circuit '{self.name}' transitioned from {old_state.value} to {new_state.value}"
        )

        if self._on_state_change:
            try:
                result = self._on_state_change(self.name, old_state, new_state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                self._audit.error(f"Circuit '{self.name}' state change callback failed: {e}")

    async def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            # HALF_OPEN state
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.CLOSED:
                self._failure_count = 0
                return

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0

    async def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if error:
                self._audit.warning(
                    f"Circuit '{self.name}' recorded failure: {type(error).__name__}: {error}"
                )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                await self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    async def __aenter__(self) -> "CircuitBreaker":
        """Context manager entry - check if execution allowed."""
        if not await self.can_execute():
            raise CircuitOpenError(self.name, self._time_until_retry())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - record success or failure."""
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def get_stats(self) -> CircuitStats:
        """Get current circuit breaker statistics."""
        return CircuitStats(
            name=self.name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_state_change=self._last_state_change,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
        )

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._audit.info(f"Circuit '{self.name}' manually reset")


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers per connector."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], Any] | None = None,
        audit: AuditLogger | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for the given name."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config, on_state_change, audit=audit)
            return self._breakers[name]

    async def get_all_stats(self) -> dict[str, CircuitStats]:
        """Get statistics for all circuit breakers."""
        async with self._lock:
            return {name: cb.get_stats() for name, cb in self._breakers.items()}

    async def reset(self, name: str) -> bool:
        """Reset a specific circuit breaker."""
        async with self._lock:
            if name in self._breakers:
                await self._breakers[name].reset()
                return True
            return False

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()


# Global registry singleton
circuit_breaker_registry = CircuitBreakerRegistry()

__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitStats",
    "CircuitOpenError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "circuit_breaker_registry",
]
