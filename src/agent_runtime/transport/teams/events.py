"""Framework-agnostic inbound event dataclasses.

Consumers receive these instead of botbuilder Activity objects, keeping
the boundary clean (see __init__ module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ConversationRef:
    """Identifies who sent an event, from where, and where to reply."""

    aad_object_id: str        # Entra Object ID — primary key for ACL + group lookup
    user_email: str           # human-readable identifier — audit log key
    user_display_name: str
    conversation_id: str      # Teams conversation ID — session key
    channel_id: str           # "msteams" for v1; explicit for future channels
    tenant_id: str            # Entra tenant ID
    service_url: str          # Bot Framework Service base URL for outbound routing
    activity_id: str          # Inbound activity ID — reserved for future reply_to_id


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """User text message or Adaptive Card Action.Submit payload."""

    conversation_ref: ConversationRef
    text: str = ""                       # may be empty when value is set
    value: dict | None = None            # Adaptive Card Action.Submit data
    kind: Literal["message"] = "message"


@dataclass(frozen=True, slots=True)
class InboundMembersAdded:
    """One or more members joined the conversation (possibly including the bot).

    ``added_aad_object_ids`` contains only Entra Object IDs. Members lacking
    an ``aad_object_id`` (rare: guests, federated accounts) are silently
    dropped — ACL systems such as Microsoft Graph ``transitiveMemberOf`` cannot
    use Bot Framework channel IDs. The bot's own entry is never included; its
    presence is signalled via ``bot_was_added``.
    """

    conversation_ref: ConversationRef
    added_aad_object_ids: tuple[str, ...] = field(default_factory=tuple)
    bot_was_added: bool = False
    kind: Literal["members_added"] = "members_added"


@dataclass(frozen=True, slots=True)
class InboundInvoke:
    """Adaptive Card Action.Execute, sign-in verifyState, or messaging-extension invoke."""

    conversation_ref: ConversationRef
    name: str = ""                       # e.g. "adaptiveCard/action"
    value: dict | None = None
    kind: Literal["invoke"] = "invoke"


InboundEvent = InboundMessage | InboundMembersAdded | InboundInvoke
