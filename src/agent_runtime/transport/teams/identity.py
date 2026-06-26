"""Identity resolution for inbound Teams activities.

Strategy:
- Call ``TeamsInfo.get_member()`` (Graph) for the canonical Entra identity —
  this is the only path that yields an email, since ``activity.from_property``
  is a plain ``ChannelAccount`` (no ``email`` field).
- On Graph failure, populate ``aad_object_id`` from ``from_property`` as best
  effort, but ``email`` remains empty.
- Fail closed: drop the activity (return None + structured WARNING) if no
  email can be resolved. The handler is not invoked for unidentifiable
  users — they cannot be ACL-checked, billed, or audited.

WARNING — Graph rate limits. Every inbound activity makes one Graph call.
Microsoft caps ``/teams/{id}/members/{userId}`` at ~10k req / 10 min per tenant
for app-only auth, with per-app-id throttling that can engage sooner. A
sustained throughput of more than a few messages per second per tenant will
hit throttling. Consumers scaling beyond a single department MUST layer a
Redis cache keyed on ``(tenant_id, from_property.id)`` with ~15-minute TTL
in front of ``resolve_identity``; see T-008e Open follow-ups.

PII note: the structured WARNING on the drop path logs ``from_id`` (opaque
``29:<base64>`` Bot Framework identifier) and ``aad_object_id`` (Entra GUID).
Neither is direct PII per Microsoft's Teams audit guidance — both are
operational identifiers, not personal data. Email is intentionally NOT logged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from botbuilder.core.teams import TeamsInfo

if TYPE_CHECKING:
    from botbuilder.core import TurnContext

from agent_runtime.safety import mask_telemetry
from agent_runtime.transport.teams.events import ConversationRef

logger = logging.getLogger(__name__)


def _extract_tenant_id(activity: Any) -> str:
    """Pull tenant ID from conversation.tenant_id or fall back to channel_data.tenant.id.

    Two real sources in Teams activities:
    - ``conversation.tenant_id`` — present on Teams activities (set by Bot Connector)
    - ``channel_data.tenant.id`` — older shape, sometimes present on conversation-update
    """
    conv_tenant = getattr(activity.conversation, "tenant_id", "") or ""
    if conv_tenant:
        return conv_tenant
    channel_data = getattr(activity, "channel_data", None)
    if isinstance(channel_data, dict):
        return channel_data.get("tenant", {}).get("id", "") or ""
    return ""


async def resolve_identity(turn_context: TurnContext) -> ConversationRef | None:
    """Return a populated ConversationRef or None if the user cannot be identified."""
    activity = turn_context.activity
    from_info = activity.from_property

    aad_object_id = ""
    email = ""
    display_name = from_info.name or "Teams User"

    try:
        member = await TeamsInfo.get_member(turn_context, from_info.id)
        # member is a TeamsChannelAccount (botbuilder.schema.teams) — has email + aad_object_id.
        aad_object_id = getattr(member, "aad_object_id", "") or ""
        email = getattr(member, "email", "") or ""
        display_name = member.name or display_name
    except Exception as exc:  # noqa: BLE001 — Graph call has no narrow exception class
        logger.warning(
            "TeamsInfo.get_member failed for %s; falling back to from_property "
            "(email is NOT available in fallback — ChannelAccount has no email field): %s",
            from_info.id,
            mask_telemetry(str(exc)),
        )
        # Only aad_object_id can come from from_property; email cannot.
        aad_object_id = getattr(from_info, "aad_object_id", "") or ""

    if not email:
        logger.warning(
            "Dropping inbound activity — no email resolved for Teams user "
            "(from_id=%s aad_object_id=%s). Either TeamsInfo.get_member failed "
            "or returned an empty email.",
            from_info.id,
            aad_object_id,
        )
        return None

    return ConversationRef(
        aad_object_id=aad_object_id,
        user_email=email,
        user_display_name=display_name,
        conversation_id=activity.conversation.id,
        channel_id=activity.channel_id or "msteams",
        tenant_id=_extract_tenant_id(activity),
        service_url=activity.service_url or "",
        activity_id=activity.id or "",
        user_channel_id=from_info.id or "",
        recipient_id=getattr(activity.recipient, "id", "") or "",
        conversation_type=getattr(activity.conversation, "conversation_type", "") or "personal",
    )
