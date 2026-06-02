"""OutboundChannel impl tests — assert correct Activity wire format."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from botbuilder.schema import ActivityTypes

from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel


@pytest.fixture
def turn_context():
    tc = MagicMock()
    tc.send_activity = AsyncMock()
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
