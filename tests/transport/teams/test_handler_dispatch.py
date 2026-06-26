"""Round-trip integration: Activity → adapter → handler with InboundEvent + outbound.

NOTE: every test in this module relies on the autouse `mock_send_activity` fixture
to patch `TurnContext.send_activity`. Without it, `BotFrameworkOutboundChannel.send_text`
(called by `_CapturingHandler.on_event`) raises KeyError because the real send path
needs `BOT_CONNECTOR_CLIENT_KEY` in `turn_state`, which is only populated by
`BotFrameworkAdapter.process_activity_with_identity` (which we bypass for unit testing).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from botbuilder.core import TurnContext
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ChannelAccount,
    ConversationAccount,
    InvokeResponse,
)

from agent_runtime.transport.teams import (
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
    TeamsAdapter,
    TeamsAdapterConfig,
)


class _CapturingHandler:
    def __init__(self, response: InvokeResponse | None = None):
        self.events: list = []
        self._response = response

    async def on_event(self, event, outbound):
        self.events.append(event)
        # Calls into BotFrameworkOutboundChannel — exercises the outbound code path.
        # The autouse fixture patches TurnContext.send_activity so this doesn't crash
        # on the missing BOT_CONNECTOR_CLIENT_KEY in turn_state.
        await outbound.send_text("hello")
        return self._response


def _make_activity(activity_type: str, **extra) -> Activity:
    base = {
        "type": activity_type,
        "id": "activity-1",
        "channel_id": "msteams",
        "service_url": "https://smba.example/",
        "conversation": ConversationAccount(id="conv-1", tenant_id="tenant-test"),
        "from_property": ChannelAccount(id="user-1", aad_object_id="aad-1", name="User One"),
        "recipient": ChannelAccount(id="bot-1", name="Bot"),
    }
    base.update(extra)
    return Activity(**base)


@pytest.fixture(autouse=True)
def mock_send_activity():
    """Patch TurnContext.send_activity for the whole module — see module docstring."""
    with patch.object(TurnContext, "send_activity", new_callable=AsyncMock) as spy:
        yield spy


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_round_trip_message_dispatches_inbound_message(mock_get_member, mock_send_activity):
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User One"
    )
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    activity = _make_activity(ActivityTypes.message, text="hi there")
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert mock_send_activity.await_count == 1
    assert mock_send_activity.await_args.args[0].text == "hello"
    assert len(handler.events) == 1
    assert isinstance(handler.events[0], InboundMessage)
    assert handler.events[0].text == "hi there"
    assert handler.events[0].conversation_ref.user_email == "u@example.com"


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_members_added_excludes_bot_and_drops_members_without_aad_object_id(mock_get_member):
    """Regression: (a) bot's `28:<guid>` ID NOT in added_aad_object_ids,
    (b) guest/federated members without aad_object_id silently dropped
    (NOT falling back to Bot Framework channel ID).
    """
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User"
    )
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    members = [
        ChannelAccount(id="28:bot-1", name="Bot"),
        ChannelAccount(id="user-2", aad_object_id="aad-u2", name="User Two"),
        ChannelAccount(id="29:guest-no-aad", aad_object_id="", name="Guest User"),  # dropped
    ]
    activity = _make_activity(ActivityTypes.conversation_update, members_added=members)
    activity.recipient = ChannelAccount(id="28:bot-1", name="Bot")
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert len(handler.events) == 1
    evt = handler.events[0]
    assert isinstance(evt, InboundMembersAdded)
    assert evt.bot_was_added is True
    assert evt.added_aad_object_ids == ("aad-u2",)  # neither bot nor guest


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_members_added_bot_compare_is_case_insensitive(mock_get_member):
    """Bot ID comparison handles Teams' occasional case-normalization quirks."""
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User"
    )
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    members = [ChannelAccount(id="28:Bot-Guid", name="Bot")]
    activity = _make_activity(ActivityTypes.conversation_update, members_added=members)
    activity.recipient = ChannelAccount(id="28:bot-guid", name="Bot")  # lowercase
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    evt = handler.events[0]
    assert evt.bot_was_added is True


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_invoke_returns_handler_invoke_response(mock_get_member):
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User"
    )
    handler = _CapturingHandler(response=InvokeResponse(status=200, body={"x": 1}))
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    activity = _make_activity(
        ActivityTypes.invoke, name="adaptiveCard/action", value={"action": "submit"}
    )
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert len(handler.events) == 1
    assert isinstance(handler.events[0], InboundInvoke)
    assert handler.events[0].name == "adaptiveCard/action"
    # Verify the handler's InvokeResponse propagated through ActivityHandler's invoke flow.
    # botbuilder stores it on the turn_state — we assert on the handler's return-side instead,
    # which is more direct and decouples from botbuilder's internal storage key.
    assert handler._response.status == 200
    assert handler._response.body == {"x": 1}


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_fail_closed_drops_event_when_no_email(mock_get_member):
    mock_get_member.return_value = SimpleNamespace(aad_object_id="aad-1", email="", name=None)
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    activity = _make_activity(ActivityTypes.message, text="hi")
    activity.from_property = ChannelAccount(id="user-1", aad_object_id="", name="No Email")
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert handler.events == []  # handler never called


def _deserialize_message_activity(text: str, entities: list[dict] | None = None) -> Activity:
    """Build a message Activity via the wire (dict) path so entity.additional_properties is
    populated — remove_recipient_mention reads mentions from additional_properties, not typed attrs."""
    body: dict = {
        "type": "message",
        "id": "activity-1",
        "channelId": "msteams",
        "serviceUrl": "https://smba.example/",
        "conversation": {"id": "conv-1", "tenantId": "tenant-test"},
        "from": {"id": "user-1", "aadObjectId": "aad-1", "name": "User One"},
        "recipient": {"id": "bot-1", "name": "Bot"},
        "text": text,
    }
    if entities:
        body["entities"] = entities
    return Activity().deserialize(body)


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_message_strips_bot_recipient_mention(mock_get_member):
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User One"
    )
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    # The mention must target recipient.id ("bot-1") to be removed; another user's mention stays.
    activity = _deserialize_message_activity(
        "<at>Knowledge Bot</at> summarize this thread",
        entities=[
            {
                "type": "mention",
                "text": "<at>Knowledge Bot</at>",
                "mentioned": {"id": "bot-1", "name": "Knowledge Bot"},
            }
        ],
    )
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert len(handler.events) == 1
    assert isinstance(handler.events[0], InboundMessage)
    assert handler.events[0].text == "summarize this thread"


@patch("agent_runtime.transport.teams.identity.TeamsInfo.get_member", new_callable=AsyncMock)
async def test_dm_message_without_mention_unchanged(mock_get_member):
    """Personal-chat path is byte-equivalent: no recipient-mention entity → text verbatim."""
    mock_get_member.return_value = SimpleNamespace(
        aad_object_id="aad-1", email="u@example.com", name="User One"
    )
    handler = _CapturingHandler()
    adapter = TeamsAdapter(TeamsAdapterConfig("aid", "pwd", "tid"), handler)
    activity = _deserialize_message_activity("plain dm question")
    tc = TurnContext(adapter._adapter, activity)
    await adapter._handler.on_turn(tc)

    assert handler.events[0].text == "plain dm question"
