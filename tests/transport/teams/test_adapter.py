"""TeamsAdapter — construction + invoke return value + on_turn_error wiring."""

import threading
from unittest.mock import AsyncMock, patch

import pytest
from botframework.connector.auth import AuthenticationConstants, MicrosoftAppCredentials

from agent_runtime.transport.teams import TeamsAdapter, TeamsAdapterConfig
from agent_runtime.transport.teams import _msal as _msal_module
from agent_runtime.transport.teams._msal import BoundedAppCredentials


class _NoOpHandler:
    async def on_event(self, event, outbound):
        return None


@pytest.fixture(autouse=True)
def _mock_token():
    """Keep the T-119 pre-warm off the network (mirrors test_images.py's fixture).

    `_build_msal_app` must be patched or the warm makes a real discovery GET — MSAL's
    constructor does network I/O (`authority.py:96-99`). Patching the PARENT's
    `get_access_token` still intercepts: `BoundedAppCredentials` overrides it but
    delegates via `super()`, which resolves through the MRO to this patch.
    """
    with (
        patch.object(
            MicrosoftAppCredentials, "get_access_token", return_value="test-token"
        ) as mock,
        patch.object(_msal_module, "_build_msal_app", return_value=object()),
    ):
        yield mock


def test_adapter_constructs_settings_from_config():
    config = TeamsAdapterConfig(app_id="aid", app_password="pwd", tenant_id="tid")
    adapter = TeamsAdapter(config, _NoOpHandler())
    settings = adapter._adapter.settings
    assert settings.app_id == "aid"
    assert settings.app_password == "pwd"
    assert settings.channel_auth_tenant == "tid"


def test_adapter_uses_default_on_turn_error_when_none_provided():
    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    assert adapter._adapter.on_turn_error is not None


def test_adapter_uses_provided_on_turn_error():
    custom = AsyncMock()
    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t", on_turn_error=custom), _NoOpHandler())
    assert adapter._adapter.on_turn_error is custom


async def test_process_activity_raises_on_empty_auth_header():
    """Empty auth_header would bypass JWT validation in some botbuilder versions."""
    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    with pytest.raises(ValueError, match="auth_header is required"):
        await adapter.process_activity({"type": "message"}, auth_header="")


@pytest.mark.parametrize("header", ["   ", "\t", "\n", " \t\n "])
async def test_process_activity_raises_on_whitespace_auth_header(header):
    """SEC-5: a whitespace-only header is truthy but effectively empty downstream."""
    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    with pytest.raises(ValueError, match="auth_header is required"):
        await adapter.process_activity({"type": "message"}, auth_header=header)


def test_adapter_supplies_bounded_credentials_to_the_sdk():
    """T-115m: without app_credentials the SDK builds its own timeout-less
    MicrosoftAppCredentials and mints connector tokens on the event loop, unbounded, on
    every outbound activity. Construction must stay network-free — tbp builds this inside
    an @lru_cache'd provider (deps.py:488)."""
    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    creds = adapter._adapter.settings.app_credentials

    assert isinstance(creds, BoundedAppCredentials)
    assert creds.app is None  # nothing built, no network touched
    assert creds.microsoft_app_id == "a"


async def test_process_activity_warms_the_connector_token_before_dispatch():
    """T-119: msrest calls `signed_session` synchronously inside its async pipeline
    (`async_requests.py:99`), so the token must already be cached before the SDK runs."""
    calls: list[str] = []

    def _record_warm(_self=None, *_args, **_kwargs):
        calls.append("warm")
        return "test-token"

    async def _record_dispatch(*_args, **_kwargs):
        # No `return None` -- ruff RET501/PLR1711 fire on it and are NOT in the
        # `tests/**` per-file-ignore list. `process_activity` maps a None response
        # to (201, None), which is what the assertion below checks.
        calls.append("dispatch")

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = _record_dispatch
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record_warm):
        result = await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert calls == ["warm", "dispatch"]
    assert result == (201, None)


async def test_connector_token_warm_runs_off_the_event_loop():
    """The whole point: a cold mint does a discovery GET + token POST. On the loop that
    blocks the entire process; on a worker it blocks one turn."""
    threads: list[str] = []

    def _record(_self=None, *_args, **_kwargs):
        threads.append(threading.current_thread().name)
        return "test-token"

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = AsyncMock(return_value=None)
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert threads and all(name != threading.main_thread().name for name in threads)


async def test_warm_failure_does_not_fail_the_turn():
    """The warm is a cache-priming optimisation with a fallback, not a gate — the send
    still mints inline (bounded by T-115m)."""

    def _boom(_self=None, *_args, **_kwargs):
        msg = "Failed to get access token with error: invalid_client"
        raise PermissionError(msg)

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    dispatched = AsyncMock(return_value=None)
    adapter._adapter.process_activity = dispatched
    with patch.object(MicrosoftAppCredentials, "get_access_token", _boom):
        result = await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert result == (201, None)
    dispatched.assert_awaited_once()


async def test_warm_gate_mirrors_the_sdk_gate_literally(_mock_token):  # noqa: PT019
    """`AppCredentials._should_set_token` gates on the app id only — NOT the password.
    Gating wider would silently skip the warm for a blank-secret misconfiguration and
    suppress the one warning that diagnoses it."""
    for app_id in ("", AuthenticationConstants.ANONYMOUS_SKILL_APP_ID):
        adapter = TeamsAdapter(TeamsAdapterConfig(app_id, "p", "t"), _NoOpHandler())
        assert await adapter.warm_connector_token() is False
    _mock_token.assert_not_called()

    blank_secret = TeamsAdapter(TeamsAdapterConfig("a", "", "t"), _NoOpHandler())
    assert await blank_secret.warm_connector_token() is True
    _mock_token.assert_called_once()
