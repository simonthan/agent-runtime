"""OutboundChannel impl tests — assert correct Activity wire format."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from botbuilder.schema import ActivityTypes, ResourceResponse

from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel


@pytest.fixture
def turn_context():
    tc = MagicMock()
    tc.send_activity = AsyncMock()
    tc.update_activity = AsyncMock()
    return tc


async def test_send_text_sends_message_activity(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    await channel.send_text("hello")
    assert turn_context.send_activity.await_count == 1
    activity = turn_context.send_activity.await_args.args[0]
    assert activity.type == ActivityTypes.message
    assert activity.text == "hello"


async def test_send_card_wraps_in_adaptive_attachment(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    card = {"type": "AdaptiveCard", "version": "1.4", "body": []}
    await channel.send_card(card)
    activity = turn_context.send_activity.await_args.args[0]
    assert len(activity.attachments) == 1
    assert activity.attachments[0].content_type == "application/vnd.microsoft.card.adaptive"
    assert activity.attachments[0].content == card


async def test_send_typing_sends_typing_activity(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    await channel.send_typing()
    activity = turn_context.send_activity.await_args.args[0]
    assert activity.type == ActivityTypes.typing


async def test_send_oauth_card_wraps_in_oauth_attachment(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    card = {"text": "Sign in", "tokenExchangeResource": {"id": "x", "uri": "api://abc"}}
    await channel.send_oauth_card(card)
    activity = turn_context.send_activity.await_args.args[0]
    assert len(activity.attachments) == 1
    assert activity.attachments[0].content_type == "application/vnd.microsoft.card.oauth"
    assert activity.attachments[0].content == card


async def test_get_sign_in_resource_maps_response(turn_context):
    turn_context.activity.from_property.id = "29:user-abc"
    resp = MagicMock()
    resp.sign_in_link = "https://token.botframework.com/api/oauth/signin?signature=xyz"
    resp.token_exchange_resource = MagicMock(uri="api://obo-client-id")
    turn_context.adapter.get_sign_in_resource_from_user = AsyncMock(return_value=resp)
    channel = BotFrameworkOutboundChannel(turn_context)
    out = await channel.get_sign_in_resource(connection_name="tbp-sso-conn")
    assert out is not None
    assert out.sign_in_link == "https://token.botframework.com/api/oauth/signin?signature=xyz"
    assert out.token_exchange_uri == "api://obo-client-id"
    turn_context.adapter.get_sign_in_resource_from_user.assert_awaited_once_with(
        turn_context, "tbp-sso-conn", "29:user-abc"
    )


async def test_get_sign_in_resource_none_without_link(turn_context):
    turn_context.activity.from_property.id = "29:user-abc"
    resp = MagicMock()
    resp.sign_in_link = None
    turn_context.adapter.get_sign_in_resource_from_user = AsyncMock(return_value=resp)
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.get_sign_in_resource(connection_name="c") is None


async def test_get_sign_in_resource_none_without_connection(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    # Empty connection short-circuits BEFORE touching the adapter.
    assert await channel.get_sign_in_resource(connection_name="") is None


async def test_get_sign_in_resource_none_when_adapter_lacks_method(turn_context):
    turn_context.activity.from_property.id = "29:user-abc"
    turn_context.adapter = object()  # e.g. a continued/proactive context
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.get_sign_in_resource(connection_name="c") is None


async def test_get_sign_in_resource_none_on_token_service_error(turn_context):
    # Fail-safe (Opus R3 HIGH): a transient token-service raise must degrade to None,
    # NOT propagate — the dispatcher prompts SSO before answering the turn, so a raise
    # here would strand the user.
    turn_context.activity.from_property.id = "29:user-abc"
    turn_context.adapter.get_sign_in_resource_from_user = AsyncMock(
        side_effect=RuntimeError("token service 503")
    )
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.get_sign_in_resource(connection_name="c") is None


async def test_send_text_returns_activity_id(turn_context):
    turn_context.send_activity = AsyncMock(return_value=ResourceResponse(id="activity-abc"))
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.send_text("hello") == "activity-abc"


async def test_send_text_returns_none_when_no_resource_response(turn_context):
    turn_context.send_activity = AsyncMock(return_value=None)
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.send_text("hello") is None


async def test_send_text_normalises_empty_id_to_none(turn_context):
    """botbuilder substitutes ResourceResponse(id="") when the Connector response is
    falsy (bot_framework_adapter.py:722-723 + the validator nulling activity.id at
    turn_context.py:191). Empty string must normalise to None or a consumer's
    `activity_id is None` degrade check silently passes."""
    turn_context.send_activity = AsyncMock(return_value=ResourceResponse(id=""))
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.send_text("hello") is None


async def test_update_activity_sends_message_with_target_id(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.update_activity("activity-abc", "edited") is True
    assert turn_context.update_activity.await_count == 1
    activity = turn_context.update_activity.await_args.args[0]
    assert activity.type == ActivityTypes.message
    assert activity.id == "activity-abc"
    assert activity.text == "edited"


async def test_update_activity_returns_false_on_connector_error(turn_context):
    turn_context.update_activity = AsyncMock(side_effect=RuntimeError("connector 502"))
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.update_activity("activity-abc", "edited") is False


async def test_update_activity_returns_false_on_empty_id(turn_context):
    channel = BotFrameworkOutboundChannel(turn_context)
    assert await channel.update_activity("", "edited") is False
    assert turn_context.update_activity.await_count == 0
