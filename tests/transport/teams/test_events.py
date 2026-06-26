"""ConversationRef + InboundEvent dataclass tests."""

import pytest

from agent_runtime.transport.teams.events import (
    ConversationRef,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
    conversation_ref_from_dict,
    conversation_ref_to_dict,
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


def test_conversation_type_defaults_to_personal():
    assert make_conversation_ref().conversation_type == "personal"


def test_conversation_ref_serializes_conversation_type_round_trip():
    ref = make_conversation_ref(conversation_type="channel")
    rebuilt = conversation_ref_from_dict(conversation_ref_to_dict(ref))
    assert rebuilt.conversation_type == "channel"
    assert rebuilt == ref  # full equality — no field dropped in the round-trip


def test_conversation_ref_from_dict_missing_conversation_type_coerces_personal():
    """A dict persisted before the field existed (T-029a-b rows) loads as 'personal'."""
    legacy = conversation_ref_to_dict(make_conversation_ref())
    del legacy["conversation_type"]
    assert conversation_ref_from_dict(legacy).conversation_type == "personal"


def test_conversation_ref_from_dict_empty_conversation_type_coerces_personal():
    legacy = conversation_ref_to_dict(make_conversation_ref())
    legacy["conversation_type"] = ""
    assert conversation_ref_from_dict(legacy).conversation_type == "personal"
