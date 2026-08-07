"""Bounded MSAL credentials (T-115m)."""

from unittest.mock import patch

from botframework.connector.auth import MicrosoftAppCredentials

from agent_runtime.transport.teams import _msal
from agent_runtime.transport.teams._msal import BoundedAppCredentials, _build_msal_app


def test_build_msal_app_passes_an_http_timeout():
    """The bug itself: MSAL got no `timeout=`, so `requests` used timeout=None and a hung
    socket blocked token acquisition forever."""
    captured: dict[str, object] = {}

    class _FakeApp:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    creds = MicrosoftAppCredentials("aid", "pwd", channel_auth_tenant="tid")
    with patch.object(_msal, "ConfidentialClientApplication", _FakeApp):
        _build_msal_app(creds)

    assert captured["timeout"] == (5.0, 10.0)
    # Mirrors the SDK's own construction argument-for-argument, so the ONLY
    # behavioural difference is the timeout.
    assert captured["client_id"] == "aid"
    assert captured["client_credential"] == "pwd"
    assert captured["authority"] == creds.oauth_endpoint


def test_construction_is_network_free_and_seeding_is_lazy():
    """teams-bot-platform builds the adapter inside an @lru_cache'd provider documented as
    network-free (deps.py:488). MSAL must not be touched until the first token request."""
    with patch.object(_msal, "_build_msal_app") as build:
        creds = BoundedAppCredentials("aid", "pwd", channel_auth_tenant="tid")
        assert build.call_count == 0
        assert creds.app is None


def test_bounded_app_is_seeded_once_then_reused():
    """`self.app` is the public attribute the SDK checks before building its own
    timeout-less app — seeding it is what makes the timeout stick."""
    sentinel = object()
    creds = BoundedAppCredentials("aid", "pwd", channel_auth_tenant="tid")
    with (
        patch.object(_msal, "_build_msal_app", return_value=sentinel) as build,
        patch.object(MicrosoftAppCredentials, "get_access_token", return_value="tok"),
    ):
        assert creds.get_access_token() == "tok"
        assert creds.get_access_token() == "tok"

    assert build.call_count == 1
    assert creds.app is sentinel
