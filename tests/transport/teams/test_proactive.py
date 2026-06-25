"""TeamsAdapter.send_proactive + ConversationRef (de)serialization."""

from unittest.mock import AsyncMock

import pytest

from agent_runtime.transport.teams import (
    TeamsAdapter,
    TeamsAdapterConfig,
    conversation_ref_from_dict,
    conversation_ref_to_dict,
)
from agent_runtime.transport.teams.testing import make_conversation_ref


class _NoOpHandler:
    async def on_event(self, event, outbound):
        return None


def _adapter() -> TeamsAdapter:
    return TeamsAdapter(TeamsAdapterConfig("app-123", "pwd", "tid"), _NoOpHandler())


def test_conversation_ref_round_trips():
    ref = make_conversation_ref(service_url="https://smba.example/x/")
    assert conversation_ref_from_dict(conversation_ref_to_dict(ref)) == ref


def test_to_dict_includes_channel_ids():
    d = conversation_ref_to_dict(make_conversation_ref())
    assert d["user_channel_id"] == "29:user-1"
    assert d["recipient_id"] == "28:bot-1"


def test_from_dict_defaults_missing_keys():
    # A dict persisted before a field existed still loads (no KeyError).
    ref = conversation_ref_from_dict({"conversation_id": "c1"})
    assert ref.conversation_id == "c1"
    assert ref.channel_id == "msteams"  # default applied
    assert ref.user_channel_id == ""  # forward-compat default
    assert ref.recipient_id == ""


def test_from_dict_ignores_unknown_keys():
    ref = conversation_ref_from_dict({"conversation_id": "c1", "bogus": "x"})
    assert ref.conversation_id == "c1"


async def test_send_proactive_requires_text_or_card():
    with pytest.raises(ValueError, match="text and/or card"):
        await _adapter().send_proactive(make_conversation_ref(), bot_app_id="app-123")


async def test_send_proactive_builds_canonical_reference():
    adapter = _adapter()
    adapter._adapter.continue_conversation = AsyncMock()
    ref = make_conversation_ref(
        conversation_id="conv-9",
        service_url="https://smba.example/v3/",
        channel_id="msteams",
        user_channel_id="29:alice",
        recipient_id="28:app-123",
    )
    await adapter.send_proactive(ref, bot_app_id="app-123", text="hi")

    adapter._adapter.continue_conversation.assert_awaited_once()
    reference, callback, bot_id = adapter._adapter.continue_conversation.call_args.args
    assert reference.conversation.id == "conv-9"
    assert reference.service_url == "https://smba.example/v3/"
    assert reference.channel_id == "msteams"
    assert reference.user.id == "29:alice"  # canonical sender channel id
    assert reference.bot.id == "28:app-123"  # canonical bot channel id
    assert bot_id == "app-123"  # raw GUID for token mint (correct)
    # Drive the callback against a fake TurnContext → asserts the send surface.
    fake_ctx = AsyncMock()
    await callback(fake_ctx)
    fake_ctx.send_activity.assert_awaited_once()
    assert fake_ctx.send_activity.call_args.args[0].text == "hi"


async def test_send_proactive_falls_back_when_channel_ids_empty():
    # A reference persisted before user_channel_id/recipient_id existed.
    adapter = _adapter()
    adapter._adapter.continue_conversation = AsyncMock()
    ref = make_conversation_ref(
        aad_object_id="00000000-0000-0000-0000-000000000001",
        user_channel_id="",
        recipient_id="",
    )
    await adapter.send_proactive(ref, bot_app_id="app-123", text="hi")
    reference, _, _ = adapter._adapter.continue_conversation.call_args.args
    assert reference.user.id == "00000000-0000-0000-0000-000000000001"  # OID fallback
    assert reference.bot.id == "28:app-123"  # synthesized bot fallback


async def test_send_proactive_sends_text_then_card_order():
    adapter = _adapter()
    adapter._adapter.continue_conversation = AsyncMock()
    await adapter.send_proactive(
        make_conversation_ref(), bot_app_id="app-123", text="t", card={"type": "AdaptiveCard"}
    )
    _, callback, _ = adapter._adapter.continue_conversation.call_args.args
    fake_ctx = AsyncMock()
    await callback(fake_ctx)
    assert fake_ctx.send_activity.await_count == 2  # text first, then card
    assert fake_ctx.send_activity.call_args_list[0].args[0].text == "t"
    assert fake_ctx.send_activity.call_args_list[1].args[0].attachments[0].content == {
        "type": "AdaptiveCard"
    }
