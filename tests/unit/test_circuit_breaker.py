"""Tests for circuit breaker implementation."""

import asyncio

import pytest

from agent_runtime.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.1,  # 100ms for fast tests
            half_open_max_calls=2,
            success_threshold=2,
        )
        return CircuitBreaker("test", config)

    async def test_initial_state_is_closed(self, breaker):
        """Test that circuit starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    async def test_can_execute_when_closed(self, breaker):
        """Test that execution is allowed when closed."""
        assert await breaker.can_execute()

    async def test_success_resets_failure_count(self, breaker):
        """Test that success resets failure count."""
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker._failure_count == 2

        await breaker.record_success()
        assert breaker._failure_count == 0

    async def test_opens_after_failure_threshold(self, breaker):
        """Test that circuit opens after threshold failures."""
        for _ in range(3):
            await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    async def test_rejects_calls_when_open(self, breaker):
        """Test that calls are rejected when circuit is open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        assert not await breaker.can_execute()

    async def test_transitions_to_half_open_after_timeout(self, breaker):
        """Test transition to half-open after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should allow one call (transitions to half-open)
        assert await breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

    async def test_closes_after_successes_in_half_open(self, breaker):
        """Test that circuit closes after successes in half-open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Transition to half-open
        await breaker.can_execute()

        # Record successes
        await breaker.record_success()
        await breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    async def test_reopens_on_failure_in_half_open(self, breaker):
        """Test that circuit reopens on failure in half-open."""
        # Open the circuit
        for _ in range(3):
            await breaker.record_failure()

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Transition to half-open
        await breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN

        # Fail
        await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerContextManager:
    """Tests for circuit breaker context manager."""

    async def test_context_manager_success(self):
        """Test context manager records success."""
        breaker = CircuitBreaker("test")

        async with breaker:
            pass  # Success

        assert breaker._total_successes == 1

    async def test_context_manager_failure(self):
        """Test context manager records failure."""
        breaker = CircuitBreaker("test")

        with pytest.raises(ValueError):
            async with breaker:
                raise ValueError("test error")

        assert breaker._failure_count == 1

    async def test_context_manager_raises_when_open(self):
        """Test context manager raises CircuitOpenError when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        breaker = CircuitBreaker("test", config)

        # Open the circuit
        await breaker.record_failure()

        with pytest.raises(CircuitOpenError) as exc_info:
            async with breaker:
                pass

        assert exc_info.value.name == "test"
        assert exc_info.value.retry_after > 0


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    async def test_creates_new_breaker(self):
        """Test registry creates new breaker."""
        registry = CircuitBreakerRegistry()
        breaker = await registry.get("test")

        assert breaker.name == "test"

    async def test_returns_same_breaker(self):
        """Test registry returns same breaker instance."""
        registry = CircuitBreakerRegistry()
        breaker1 = await registry.get("test")
        breaker2 = await registry.get("test")

        assert breaker1 is breaker2

    async def test_get_all_stats(self):
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry()
        await registry.get("breaker1")
        await registry.get("breaker2")

        stats = await registry.get_all_stats()

        assert "breaker1" in stats
        assert "breaker2" in stats

    async def test_reset_specific_breaker(self):
        """Test resetting a specific breaker."""
        registry = CircuitBreakerRegistry()
        breaker = await registry.get("test")

        # Open the circuit
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker.config = config
        await breaker.record_failure()
        assert breaker.is_open

        # Reset
        await registry.reset("test")
        assert breaker.is_closed

    async def test_reset_all_breakers(self):
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry()
        b1 = await registry.get("b1")
        b2 = await registry.get("b2")

        # Open both circuits
        b1.config = CircuitBreakerConfig(failure_threshold=1)
        b2.config = CircuitBreakerConfig(failure_threshold=1)
        await b1.record_failure()
        await b2.record_failure()

        assert b1.is_open
        assert b2.is_open

        # Reset all
        await registry.reset_all()

        assert b1.is_closed
        assert b2.is_closed


class TestRetryAfterSeconds:
    """Tests for retry_after_seconds() — added in T-402c2 for log telemetry."""

    @pytest.fixture
    def breaker(self):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            half_open_max_calls=1,
            success_threshold=1,
        )
        return CircuitBreaker("test", config)

    async def test_returns_zero_when_closed(self, breaker):
        assert breaker.retry_after_seconds() == 0.0

    async def test_returns_zero_when_half_open(self, breaker):
        # Trip
        for _ in range(2):
            await breaker.record_failure()
        # Wait for recovery
        await asyncio.sleep(0.55)
        # Transition to HALF_OPEN
        await breaker.can_execute()
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.retry_after_seconds() == 0.0

    async def test_positive_when_open_within_window(self, breaker):
        for _ in range(2):
            await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        retry = breaker.retry_after_seconds()
        assert 0.0 < retry <= 0.5

    async def test_decreases_monotonically_while_open(self, breaker):
        for _ in range(2):
            await breaker.record_failure()
        first = breaker.retry_after_seconds()
        await asyncio.sleep(0.05)
        second = breaker.retry_after_seconds()
        assert second < first

    async def test_returns_zero_after_recovery_window_elapsed(self, breaker):
        for _ in range(2):
            await breaker.record_failure()
        await asyncio.sleep(0.55)
        # State is still OPEN until can_execute() transitions it,
        # but the retry window has passed — value clamps to 0.0.
        assert breaker.state == CircuitState.OPEN
        assert breaker.retry_after_seconds() == 0.0
