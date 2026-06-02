"""resolve_identity — Graph + fallback + fail-closed logging."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_runtime.transport.teams.identity import resolve_identity


def _turn_context(from_id="user-1", from_aad="", from_name="User One"):
    """Build a turn_context double mirroring botbuilder's ChannelAccount shape.

    Note: ChannelAccount has NO email field — only id, name, aad_object_id, role,
    properties. We don't pass from_email because identity.py never reads it from
    from_property; email comes exclusively from the TeamsInfo.get_member() path.
    """
    activity = SimpleNamespace(
        from_property=SimpleNamespace(id=from_id, aad_object_id=from_aad, name=from_name),
        conversation=SimpleNamespace(id="conv-1", tenant_id="tenant-test"),
        channel_id="msteams",
        service_url="https://smba.example/",
        id="activity-1",
        channel_data={"tenant": {"id": "tenant-test"}},
    )
    return MagicMock(activity=activity)


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_resolve_identity_happy_path(mock_get_member):
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User One"
    )
    ref = await resolve_identity(_turn_context())
    assert ref is not None
    assert ref.aad_object_id == "aad-1"
    assert ref.user_email == "u@example.com"
    assert ref.user_display_name == "User One"
    assert ref.conversation_id == "conv-1"


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_resolve_identity_drops_when_graph_raises_because_email_unavailable_in_fallback(
    mock_get_member, caplog
):
    """ChannelAccount.from_property has no email field, so Graph failure → activity dropped.

    The fallback path only populates aad_object_id from from_property; email cannot
    be recovered. Fail-closed policy then drops the activity.
    """
    mock_get_member.side_effect = RuntimeError("Graph unreachable")
    caplog.set_level(logging.WARNING)
    ref = await resolve_identity(_turn_context(from_aad="aad-fb"))
    assert ref is None
    messages = [r.message for r in caplog.records]
    assert any("TeamsInfo.get_member failed" in m for m in messages)
    assert any("Dropping inbound activity" in m for m in messages)


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_resolve_identity_fails_closed_when_graph_returns_no_email(mock_get_member, caplog):
    """TeamsInfo returns a member, but the email field is empty → drop."""
    mock_get_member.return_value = SimpleNamespace(aad_object_id="aad-x", email="", name=None)
    caplog.set_level(logging.WARNING)
    ref = await resolve_identity(_turn_context())
    assert ref is None
    assert any("Dropping inbound activity" in r.message for r in caplog.records)


def test_extract_tenant_id_prefers_conversation_tenant():
    """Tenant ID resolution: conversation.tenant_id wins when present."""
    from agent_runtime.transport.teams.identity import _extract_tenant_id

    activity = SimpleNamespace(
        conversation=SimpleNamespace(tenant_id="from-conv"),
        channel_data={"tenant": {"id": "from-channel"}},
    )
    assert _extract_tenant_id(activity) == "from-conv"


def test_extract_tenant_id_falls_back_to_channel_data():
    from agent_runtime.transport.teams.identity import _extract_tenant_id

    activity = SimpleNamespace(
        conversation=SimpleNamespace(tenant_id=""),
        channel_data={"tenant": {"id": "from-channel"}},
    )
    assert _extract_tenant_id(activity) == "from-channel"


def test_extract_tenant_id_returns_empty_when_channel_data_missing():
    from agent_runtime.transport.teams.identity import _extract_tenant_id

    activity = SimpleNamespace(
        conversation=SimpleNamespace(tenant_id=""),
        channel_data=None,
    )
    assert _extract_tenant_id(activity) == ""
