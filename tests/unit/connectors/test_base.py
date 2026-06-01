"""Tests for agent_runtime.connectors.base — generic connector primitives."""

import asyncio

import httpx
import pytest

from agent_runtime.connectors.base import (
    NON_RETRYABLE_HTTP_STATUS_CODES,
    RETRYABLE_HTTP_STATUS_CODES,
    BaseConnector,
    ConnectorResult,
    RetryMixin,
    _get_throttle,
    _rate_limit_config,
    _throttle_registry,
    is_retryable_error,
    register_rate_limit,
    set_audit_logger,
)
from agent_runtime.logging import NullAuditLogger


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Clear module-level mutable state before AND after each test.

    Do NOT save/restore — old throttle instances may hold acquired semaphores
    from a failed prior test, and restoring them re-injects that broken state.
    Each test starts with empty registries; the original NullAuditLogger default
    is restored unconditionally.
    """
    import agent_runtime.connectors.base as base_mod
    base_mod._rate_limit_config.clear()
    base_mod._throttle_registry.clear()
    base_mod._default_audit = NullAuditLogger()
    yield
    base_mod._rate_limit_config.clear()
    base_mod._throttle_registry.clear()
    base_mod._default_audit = NullAuditLogger()


def _http_error(status_code: int) -> Exception:
    """Build a mock httpx.HTTPStatusError for a given status code."""
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


class TestRetryableErrorMatrix:
    def test_retryable_codes(self):
        for code in RETRYABLE_HTTP_STATUS_CODES:
            assert is_retryable_error(_http_error(code)) is True

    def test_non_retryable_codes(self):
        for code in NON_RETRYABLE_HTTP_STATUS_CODES:
            assert is_retryable_error(_http_error(code)) is False

    def test_connection_error_is_retryable(self):
        assert is_retryable_error(ConnectionError("dropped")) is True

    def test_timeout_error_is_retryable(self):
        assert is_retryable_error(TimeoutError("timed out")) is True

    def test_os_error_is_retryable(self):
        assert is_retryable_error(OSError("network unreachable")) is True

    def test_httpx_timeout_is_retryable(self):
        assert is_retryable_error(httpx.ConnectTimeout("timeout")) is True

    def test_httpx_connect_error_is_retryable(self):
        assert is_retryable_error(httpx.ConnectError("connect failed")) is True

    def test_value_error_is_not_retryable(self):
        assert is_retryable_error(ValueError("bad value")) is False


class TestConnectorResult:
    def test_default_factory_fields(self):
        result = ConnectorResult(success=True, message="ok")
        assert result.success is True
        assert result.message == "ok"
        assert result.data is None
        assert result.error_code is None
        assert result.http_status is None

    def test_full_construction(self):
        result = ConnectorResult(
            success=False,
            message="error",
            data={"key": "val"},
            error_code="SomeError",
            http_status=500,
        )
        assert result.success is False
        assert result.data == {"key": "val"}
        assert result.error_code == "SomeError"
        assert result.http_status == 500


class TestBaseConnectorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseConnector()  # abstract

    def test_subclass_with_required_methods_instantiates(self):
        # BaseConnector has 3 @abstractmethod methods: initialize, health_check, close.
        # All three must be overridden for the subclass to be instantiable.
        class _Stub(BaseConnector):
            async def initialize(self) -> bool: return True
            async def health_check(self) -> ConnectorResult: return ConnectorResult(success=True, message="ok")
            async def close(self) -> None: pass
        _Stub()  # no exception

    def test_is_initialized_starts_false(self):
        class _Stub(BaseConnector):
            async def initialize(self) -> bool: return True
            async def health_check(self) -> ConnectorResult: return ConnectorResult(success=True, message="ok")
            async def close(self) -> None: pass
        stub = _Stub()
        assert stub.is_initialized is False

    def test_handle_error_returns_structured_connector_result(self):
        """Covers the concrete inherited _handle_error path that all subclasses use."""
        class _Stub(BaseConnector):
            async def initialize(self) -> bool: return True
            async def health_check(self) -> ConnectorResult: return ConnectorResult(success=True, message="ok")
            async def close(self) -> None: pass
        stub = _Stub()
        # ihd source signature is `_handle_error(self, error: Exception, operation: str)` —
        # error FIRST, operation SECOND. Do NOT reverse these positional args (lift verbatim).
        result = stub._handle_error(ValueError("boom"), "test_op")
        assert isinstance(result, ConnectorResult)
        assert result.success is False
        assert result.error_code == "ValueError"
        assert "test_op" in result.data["_internal_error"]

    def test_handle_error_captures_http_status(self):
        class _Stub(BaseConnector):
            async def initialize(self) -> bool: return True
            async def health_check(self) -> ConnectorResult: return ConnectorResult(success=True, message="ok")
            async def close(self) -> None: pass
        stub = _Stub()
        err = _http_error(503)
        result = stub._handle_error(err, "http_op")
        assert result.http_status == 503
        assert result.success is False


class TestRetryMixinBehavior:
    async def test_retries_on_retryable_error_then_succeeds(self):
        """Operation fails twice with retryable error, then succeeds on 3rd attempt."""
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        class _Stub(RetryMixin):
            pass

        stub = _Stub()
        result = await stub._execute_with_retry(
            flaky_op,
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.05,
        )
        assert result == "success"
        assert call_count == 3

    async def test_no_retry_on_non_retryable_error(self):
        """Non-retryable error (401) should not trigger retries."""
        call_count = 0

        async def auth_fail():
            nonlocal call_count
            call_count += 1
            raise _http_error(401)

        class _Stub(RetryMixin):
            pass

        stub = _Stub()
        with pytest.raises(httpx.HTTPStatusError):
            await stub._execute_with_retry(
                auth_fail,
                max_attempts=3,
                min_wait=0.01,
                max_wait=0.05,
            )
        assert call_count == 1  # no retries

    async def test_raises_after_max_attempts_exhausted(self):
        """All retryable — should raise after max_attempts."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("always fails")

        class _Stub(RetryMixin):
            pass

        stub = _Stub()
        with pytest.raises(ConnectionError):
            await stub._execute_with_retry(
                always_fail,
                max_attempts=2,
                min_wait=0.01,
                max_wait=0.05,
            )
        assert call_count == 2

    async def test_throttle_applied_when_registered(self):
        """When a rate limit is registered for the connector class, throttle is invoked."""
        register_rate_limit("_ThrottledStub", max_concurrent=1, min_delay_seconds=0.0)
        call_count = 0

        async def op():
            nonlocal call_count
            call_count += 1
            return "ok"

        class _ThrottledStub(RetryMixin):
            pass

        stub = _ThrottledStub()
        result = await stub._execute_with_retry(op, max_attempts=1)
        assert result == "ok"
        assert call_count == 1

    async def test_throttle_min_delay_enforced(self):
        """Throttle min_delay path is exercised when two calls arrive in quick succession."""
        register_rate_limit("_DelayStub", max_concurrent=1, min_delay_seconds=0.05)

        results = []

        async def op():
            results.append(True)
            return True

        class _DelayStub(RetryMixin):
            pass

        stub = _DelayStub()
        # Call twice quickly — second call should hit the delay branch
        await stub._execute_with_retry(op, max_attempts=1)
        await stub._execute_with_retry(op, max_attempts=1)
        assert len(results) == 2

    async def test_retry_after_header_respected(self):
        """429 with Retry-After header: wait_time is capped from header, warning logged."""
        import agent_runtime.connectors.base as base_mod

        class _Spy:
            def __init__(self): self.calls = []
            def debug(self, m, **kw): self.calls.append(("debug", m))
            def info(self, m, **kw): self.calls.append(("info", m))
            def warning(self, m, **kw): self.calls.append(("warning", m))
            def error(self, m, **kw): self.calls.append(("error", m))
            def security(self, m, **kw): self.calls.append(("security", m))
            def action(self, a, r, **kw): self.calls.append(("action", a))

        spy = _Spy()
        set_audit_logger(spy)

        call_count = 0

        async def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request = httpx.Request("GET", "http://x")
                response = httpx.Response(
                    429,
                    request=request,
                    headers={"Retry-After": "0.01"},
                )
                raise httpx.HTTPStatusError("rate limited", request=request, response=response)
            return "recovered"

        class _Stub(RetryMixin):
            pass

        result = await _Stub()._execute_with_retry(
            rate_limited,
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
        )
        assert result == "recovered"
        warning_msgs = [c[1] for c in spy.calls if c[0] == "warning"]
        # Should have a Retry-After warning AND a retry warning
        assert any("Retry-After" in m for m in warning_msgs)

    def test_with_retry_decorator_returns_callable(self):
        """RetryMixin.with_retry() returns a tenacity retry decorator."""
        decorator = RetryMixin.with_retry(max_attempts=2, min_wait=0.01, max_wait=0.1)
        assert callable(decorator)

    async def test_retry_after_invalid_header_falls_back_to_exponential(self):
        """Invalid Retry-After header value is silently ignored (ValueError/TypeError path)."""
        call_count = 0

        async def rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request = httpx.Request("GET", "http://x")
                response = httpx.Response(
                    429,
                    request=request,
                    headers={"Retry-After": "not-a-number"},
                )
                raise httpx.HTTPStatusError("rate limited", request=request, response=response)
            return "recovered"

        class _Stub(RetryMixin):
            pass

        result = await _Stub()._execute_with_retry(
            rate_limited,
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
        )
        assert result == "recovered"
        assert call_count == 2


class TestRateLimitRegistration:
    async def test_get_throttle_returns_none_for_unregistered(self):
        throttle = await _get_throttle("UnregisteredConnector")
        assert throttle is None

    async def test_get_throttle_returns_instance_after_registration(self):
        register_rate_limit("TestConn", max_concurrent=2, min_delay_seconds=0.1)
        throttle = await _get_throttle("TestConn")
        assert throttle is not None

    def test_register_is_idempotent(self):
        register_rate_limit("X", max_concurrent=1, min_delay_seconds=0.5)
        register_rate_limit("X", max_concurrent=5, min_delay_seconds=0.1)
        assert _rate_limit_config["X"]["max_concurrent"] == 5  # int, not 5.0
        assert _rate_limit_config["X"]["min_delay_seconds"] == 0.1

    def test_max_concurrent_stored_as_int(self):
        register_rate_limit("IntCheck", max_concurrent=3, min_delay_seconds=1.0)
        assert isinstance(_rate_limit_config["IntCheck"]["max_concurrent"], int)

    def test_min_delay_stored_as_float(self):
        register_rate_limit("FloatCheck", max_concurrent=2, min_delay_seconds=0.5)
        assert isinstance(_rate_limit_config["FloatCheck"]["min_delay_seconds"], float)

    async def test_throttle_registry_caches_instance(self):
        """Second call to _get_throttle returns the same object."""
        register_rate_limit("CachedConn", max_concurrent=2, min_delay_seconds=0.1)
        t1 = await _get_throttle("CachedConn")
        t2 = await _get_throttle("CachedConn")
        assert t1 is t2

    async def test_rate_limit_config_starts_empty(self):
        """Acceptance criteria 3: _rate_limit_config == {} in unconfigured state."""
        # _reset_module_state fixture ensures clean state
        assert _rate_limit_config == {}


class TestSetAuditLogger:
    # NOTE: A "test the shipping default is NullAuditLogger" guard belongs in a
    # SEPARATE test module without the autouse _reset_module_state fixture (because
    # the fixture forces _default_audit = NullAuditLogger() before every test,
    # making any in-fixture assertion tautological). See test_protocol.py for
    # where to add such a guard if needed. The shipping default is also asserted
    # transitively by T-490b's test_audit_logger_is_wired (which would fail if
    # the shipping default were not NullAuditLogger AND the shim's set_audit_logger
    # call were removed).

    def test_set_audit_logger_replaces_module_default(self):
        class _Spy:
            def __init__(self): self.calls = []
            def debug(self, m, **kw): self.calls.append(("debug", m, kw))
            def info(self, m, **kw): self.calls.append(("info", m, kw))
            def warning(self, m, **kw): self.calls.append(("warning", m, kw))
            def error(self, m, **kw): self.calls.append(("error", m, kw))
            def security(self, m, **kw): self.calls.append(("security", m, kw))
            def action(self, action, result, **kw): self.calls.append(("action", action, result, kw))
        spy = _Spy()
        set_audit_logger(spy)
        import agent_runtime.connectors.base as base_mod
        assert base_mod._default_audit is spy

    def test_audit_logger_receives_error_calls(self):
        """_handle_error routes through _default_audit.error()."""
        class _Spy:
            def __init__(self): self.calls = []
            def debug(self, m, **kw): self.calls.append(("debug", m, kw))
            def info(self, m, **kw): self.calls.append(("info", m, kw))
            def warning(self, m, **kw): self.calls.append(("warning", m, kw))
            def error(self, m, **kw): self.calls.append(("error", m, kw))
            def security(self, m, **kw): self.calls.append(("security", m, kw))
            def action(self, action, result, **kw): self.calls.append(("action", action, result, kw))

        spy = _Spy()
        set_audit_logger(spy)

        class _Stub(BaseConnector):
            async def initialize(self) -> bool: return True
            async def health_check(self) -> ConnectorResult: return ConnectorResult(success=True, message="ok")
            async def close(self) -> None: pass

        _Stub()._handle_error(ValueError("boom"), "op")
        error_calls = [c for c in spy.calls if c[0] == "error"]
        assert len(error_calls) == 1
        assert "op" in error_calls[0][1]
