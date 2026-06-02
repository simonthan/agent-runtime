"""ConversationRef + InboundEvent dataclass tests."""
import pytest
from agent_runtime.transport.teams.events import (
    ConversationRef,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
)
from agent_runtime.transport.teams.testing import make_conversation_ref


def test_conversation_ref_is_frozen():
    ref = make_conversation_ref()
    with pytest.raises(AttributeError):
        ref.user_email = "other@example.com"  # type: ignore[misc]


def test_conversation_ref_is_hashable():
    {make_conversation_ref()}  # constructing a set proves __hash__ works


def test_inbound_message_kind_discriminator():
    msg = InboundMessage(conversation_ref=make_conversation_ref(), text="hi")
    assert msg.kind == "message"
    match msg:
        case InboundMessage():
            pass
        case _:
            pytest.fail("InboundMessage didn't match")


def test_inbound_members_added_defaults():
    evt = InboundMembersAdded(conversation_ref=make_conversation_ref())
    assert evt.added_aad_object_ids == ()
    assert evt.bot_was_added is False
    assert evt.kind == "members_added"


def test_inbound_invoke_carries_name_and_value():
    evt = InboundInvoke(
        conversation_ref=make_conversation_ref(),
        name="adaptiveCard/action",
        value={"x": 1},
    )
    assert evt.kind == "invoke"
    assert evt.name == "adaptiveCard/action"
    assert evt.value == {"x": 1}
