"""TeamsAdapter — construction + invoke return value + on_turn_error wiring."""

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from botbuilder.core import TurnContext
from botbuilder.schema import ActivityTypes
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


async def test_process_activity_warms_the_connector_token_inside_the_callback():
    """T-119: msrest calls `signed_session` synchronously inside its async pipeline
    (`async_requests.py:99`), so the token must already be cached before the SDK sends.
    T-134: the warm moved into the SDK callback -- which runs only after JWT validation --
    so it must still land BEFORE the handler's turn, just not before authentication."""
    calls: list[str] = []

    def _record_warm(_self=None, *_args, **_kwargs):
        calls.append("warm")
        return "test-token"

    async def _record_turn(_turn_context):
        calls.append("turn")

    async def _record_dispatch(_activity, _auth_header, callback):
        # Stands in for the SDK: authenticate, THEN invoke the callback. No `return None`
        # -- ruff RET501/PLR1711 fire on it and are NOT in the `tests/**` ignore list.
        calls.append("dispatch")
        await callback(MagicMock())

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = _record_dispatch
    adapter._handler.on_turn = _record_turn
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record_warm):
        result = await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert calls == ["dispatch", "warm", "turn"]
    assert result == (201, None)


async def test_warm_does_not_run_when_authentication_rejects_the_request():
    """T-134: the warm used to run before `Activity().deserialize` and before the SDK's JWT
    validation, gated on nothing but a non-empty header -- so a forged-token POST to the
    consumer's webhook dispatched a worker-thread AAD token operation. A request the SDK
    never authenticates must cost no token work at all."""
    calls: list[str] = []

    def _record_warm(_self=None, *_args, **_kwargs):
        calls.append("warm")
        return "test-token"

    async def _reject(_activity, _auth_header, _callback):
        # The SDK raises/returns without invoking the callback when auth fails.
        calls.append("auth-rejected")

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = _reject
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record_warm):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer forged")

    assert calls == ["auth-rejected"]


async def test_connector_token_warm_runs_off_the_event_loop():
    """The whole point: a cold mint does a discovery GET + token POST. On the loop that
    blocks the entire process; on a worker it blocks one turn."""
    threads: list[str] = []

    def _record(_self=None, *_args, **_kwargs):
        threads.append(threading.current_thread().name)
        return "test-token"

    async def _dispatch(_activity, _auth_header, callback):
        await callback(MagicMock())

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = _dispatch
    adapter._handler.on_turn = AsyncMock()
    with patch.object(MicrosoftAppCredentials, "get_access_token", _record):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert threads and all(name != threading.main_thread().name for name in threads)


async def test_warm_failure_does_not_fail_the_turn():
    """The warm is a cache-priming optimisation with a fallback, not a gate — the send
    still mints inline (bounded by T-115m). T-134: now asserted through the callback, so
    the PermissionError is raised on the path that actually runs it."""

    def _boom(_self=None, *_args, **_kwargs):
        msg = "Failed to get access token with error: invalid_client"
        raise PermissionError(msg)

    turned = AsyncMock()

    async def _dispatch(_activity, _auth_header, callback):
        await callback(MagicMock())

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _NoOpHandler())
    adapter._adapter.process_activity = _dispatch
    adapter._handler.on_turn = turned
    with patch.object(MicrosoftAppCredentials, "get_access_token", _boom):
        result = await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert result == (201, None)
    turned.assert_awaited_once()


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


async def test_on_message_sends_best_effort_reply_when_identity_fails():
    """T-286: a message whose identity can't be resolved gets a user-visible
    reply instead of being silently dropped."""
    sent: list[str] = []

    class _TrackingHandler:
        async def on_event(self, event, outbound):
            return None

    async def _dispatch(_activity, _auth_header, callback):
        ctx = MagicMock()
        ctx.activity = MagicMock()
        ctx.activity.type = ActivityTypes.message
        ctx.activity.from_property = MagicMock(id="29:test", aad_object_id="", name="Test")
        ctx.activity.value = None
        ctx.activity.attachments = None
        ctx.send_activity = AsyncMock(side_effect=lambda a: sent.append(a.text))
        await callback(ctx)

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _TrackingHandler())
    adapter._adapter.process_activity = _dispatch

    with (
        patch(
            "agent_runtime.transport.teams.adapter.resolve_identity",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(TurnContext, "remove_recipient_mention", return_value="hello"),
    ):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert len(sent) == 1
    assert "unable to process" in sent[0].lower()


async def test_on_message_calls_identity_failed_hook_when_identity_fails():
    """T-286: the adapter calls on_identity_failed on the handler if it exists."""
    hook_calls: list[dict] = []

    class _HookHandler:
        async def on_event(self, event, outbound):
            return None

        async def on_identity_failed(self, *, from_id, aad_object_id, conversation_type):
            hook_calls.append(
                {
                    "from_id": from_id,
                    "aad_object_id": aad_object_id,
                    "conversation_type": conversation_type,
                }
            )

    async def _dispatch(_activity, _auth_header, callback):
        ctx = MagicMock()
        ctx.activity = MagicMock()
        ctx.activity.type = ActivityTypes.message
        ctx.activity.from_property = MagicMock(id="29:test", aad_object_id="oid-123", name="Test")
        ctx.activity.conversation = MagicMock(conversation_type="groupChat")
        ctx.activity.value = None
        ctx.activity.attachments = None
        ctx.send_activity = AsyncMock()
        await callback(ctx)

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _HookHandler())
    adapter._adapter.process_activity = _dispatch

    with (
        patch(
            "agent_runtime.transport.teams.adapter.resolve_identity",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(TurnContext, "remove_recipient_mention", return_value="hello"),
    ):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert len(hook_calls) == 1
    assert hook_calls[0]["from_id"] == "29:test"
    assert hook_calls[0]["aad_object_id"] == "oid-123"
    assert hook_calls[0]["conversation_type"] == "groupChat"


async def test_on_message_swallows_reply_failure_and_still_calls_hook():
    """T-286: if the best-effort reply fails, the hook still fires."""
    hook_calls: list[dict] = []

    class _HookHandler:
        async def on_event(self, event, outbound):
            return None

        async def on_identity_failed(self, *, from_id, aad_object_id, conversation_type):
            hook_calls.append({"from_id": from_id})

    async def _dispatch(_activity, _auth_header, callback):
        ctx = MagicMock()
        ctx.activity = MagicMock()
        ctx.activity.type = ActivityTypes.message
        ctx.activity.from_property = MagicMock(id="29:test", aad_object_id="", name="Test")
        ctx.activity.conversation = MagicMock(conversation_type="personal")
        ctx.activity.value = None
        ctx.activity.attachments = None
        ctx.send_activity = AsyncMock(side_effect=RuntimeError("Connector refused"))
        await callback(ctx)

    adapter = TeamsAdapter(TeamsAdapterConfig("a", "p", "t"), _HookHandler())
    adapter._adapter.process_activity = _dispatch

    with (
        patch(
            "agent_runtime.transport.teams.adapter.resolve_identity",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(TurnContext, "remove_recipient_mention", return_value="hello"),
    ):
        await adapter.process_activity({"type": "message"}, auth_header="Bearer x")

    assert len(hook_calls) == 1  # hook fired despite reply failure
