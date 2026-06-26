"""Test doubles and factory helpers for consumers of the Teams transport.

Public surface: ``FakeOutboundChannel``, ``make_conversation_ref``,
``make_inbound_message``, ``make_inbound_members_added``, ``make_inbound_invoke``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_runtime.transport.teams.events import (
    ConversationRef,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
)


@dataclass
class FakeOutboundChannel:
    """In-memory OutboundChannel that records sends in order."""

    sent_texts: list[str] = field(default_factory=list)
    sent_cards: list[dict] = field(default_factory=list)
    sent_oauth_cards: list[dict] = field(default_factory=list)
    sent_typing_count: int = 0

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)

    async def send_card(self, card: dict) -> None:
        self.sent_cards.append(card)

    async def send_oauth_card(self, card: dict) -> None:
        self.sent_oauth_cards.append(card)

    async def send_typing(self) -> None:
        self.sent_typing_count += 1

    def clear(self) -> None:
        self.sent_texts.clear()
        self.sent_cards.clear()
        self.sent_oauth_cards.clear()
        self.sent_typing_count = 0


def make_conversation_ref(**overrides: str) -> ConversationRef:
    return ConversationRef(
        **{
            "aad_object_id": "aad-test-user-1",
            "user_email": "user1@example.com",
            "user_display_name": "Test User",
            "conversation_id": "conv-1",
            "channel_id": "msteams",
            "tenant_id": "tenant-test",
            "service_url": "https://smba.trafficmanager.net/test/",
            "activity_id": "activity-1",
            "user_channel_id": "29:user-1",
            "recipient_id": "28:bot-1",
            "conversation_type": "personal",
            **overrides,
        }
    )


def make_inbound_message(
    text: str = "hello",
    value: dict | None = None,
    **ref_overrides: str,
) -> InboundMessage:
    return InboundMessage(
        conversation_ref=make_conversation_ref(**ref_overrides),
        text=text,
        value=value,
    )


def make_inbound_members_added(
    added_aad_object_ids: tuple[str, ...] = ("aad-new-user",),
    *,
    bot_was_added: bool = False,
    **ref_overrides: str,
) -> InboundMembersAdded:
    return InboundMembersAdded(
        conversation_ref=make_conversation_ref(**ref_overrides),
        added_aad_object_ids=added_aad_object_ids,
        bot_was_added=bot_was_added,
    )


def make_inbound_invoke(
    name: str = "adaptiveCard/action",
    value: dict | None = None,
    **ref_overrides: str,
) -> InboundInvoke:
    # Propagate value as-is — including None. The adapter sets value=None when
    # activity.value is not a dict (e.g. typed MessagingExtensionQuery, missing),
    # so test factories must be able to reproduce that branch.
    return InboundInvoke(
        conversation_ref=make_conversation_ref(**ref_overrides),
        name=name,
        value=value,
    )
