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

    aad_object_id: str  # Entra Object ID — primary key for ACL + group lookup
    user_email: str  # human-readable identifier — audit log key
    user_display_name: str
    conversation_id: str  # Teams conversation ID — session key
    channel_id: str  # "msteams" for v1; explicit for future channels
    tenant_id: str  # Entra tenant ID
    service_url: str  # Bot Framework Service base URL for outbound routing
    activity_id: str  # Inbound activity ID — reserved for future reply_to_id
    user_channel_id: str = ""  # "29:…" sender channel id → proactive ConversationReference.user
    recipient_id: str = ""  # "28:<appid>" bot channel id → proactive ConversationReference.bot
    conversation_type: str = "personal"  # "personal" | "channel" | "groupChat" (T-031a)
    # Channel turns get stateless single-shot handling; DM turns use the persona/session flow.


def conversation_ref_to_dict(ref: ConversationRef) -> dict[str, str]:
    """Serialize a ConversationRef to a flat str->str dict for durable storage.

    Consumers persist this (e.g. a Postgres row) to send a proactive message
    later via ``TeamsAdapter.send_proactive``. Rebuild with
    ``conversation_ref_from_dict``, which tolerates missing keys so a row
    written by an older schema still loads.
    """
    return {
        "aad_object_id": ref.aad_object_id,
        "user_email": ref.user_email,
        "user_display_name": ref.user_display_name,
        "conversation_id": ref.conversation_id,
        "channel_id": ref.channel_id,
        "tenant_id": ref.tenant_id,
        "service_url": ref.service_url,
        "activity_id": ref.activity_id,
        "user_channel_id": ref.user_channel_id,
        "recipient_id": ref.recipient_id,
        "conversation_type": ref.conversation_type,
    }


def conversation_ref_from_dict(data: dict[str, str]) -> ConversationRef:
    """Rebuild a ConversationRef from ``conversation_ref_to_dict`` output.

    Missing keys default to "" (``channel_id`` to "msteams", ``conversation_type``
    to "personal") so a dict persisted before a field existed still loads —
    forward-compat for schema evolution. Unknown keys are ignored for the same reason.
    """
    return ConversationRef(
        aad_object_id=data.get("aad_object_id", ""),
        user_email=data.get("user_email", ""),
        user_display_name=data.get("user_display_name", ""),
        conversation_id=data.get("conversation_id", ""),
        channel_id=data.get("channel_id", "") or "msteams",
        tenant_id=data.get("tenant_id", ""),
        service_url=data.get("service_url", ""),
        activity_id=data.get("activity_id", ""),
        user_channel_id=data.get("user_channel_id", ""),
        recipient_id=data.get("recipient_id", ""),
        conversation_type=data.get("conversation_type", "") or "personal",
    )


@dataclass(frozen=True, slots=True)
class FileAttachment:
    """A Teams file upload surfaced on an inbound message.

    Teams stores a 1:1 chat file upload in the *sender's* OneDrive and delivers it
    as an activity attachment whose ``contentType`` is
    ``application/vnd.microsoft.teams.file.download.info``. ``item_id`` is the
    OneDrive driveItem id (``content.uniqueId``) — the ``resource_id`` a consumer
    passes to a drive read (e.g. eyeglass ``read_resource(type="drive")``).
    ``file_type`` is ``content.fileType`` (e.g. ``"docx"``, lowercase, no dot; may
    be ``""``). ``download_url`` is a pre-authenticated, time-limited URL — NOT used
    for durable read-on-demand, surfaced only for completeness."""

    item_id: str
    name: str
    file_type: str = ""
    download_url: str = ""


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """User text message or Adaptive Card Action.Submit payload."""

    conversation_ref: ConversationRef
    text: str = ""  # may be empty when value is set
    value: dict | None = None  # Adaptive Card Action.Submit data
    attachments: tuple[FileAttachment, ...] = field(default_factory=tuple)  # T-037c
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
    name: str = ""  # e.g. "adaptiveCard/action"
    value: dict | None = None
    kind: Literal["invoke"] = "invoke"


InboundEvent = InboundMessage | InboundMembersAdded | InboundInvoke
