"""TeamsAdapter — construction + invoke return value + on_turn_error wiring."""

from unittest.mock import AsyncMock

import pytest

from agent_runtime.transport.teams import TeamsAdapter, TeamsAdapterConfig


class _NoOpHandler:
    async def on_event(self, event, outbound):
        return None


def test_adapter_constructs_settings_from_config():
    config = TeamsAdapterConfig(app_id="aid", app_password="pwd", tenant_id="tid")
    adapter = TeamsAdapter(config, _NoOpHandler())
    settings = adapter._adapter.settings
    assert settings.app_id == "aid"
    assert settings.app_password == "pwd"  # noqa: S105
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
