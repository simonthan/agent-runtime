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

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any

from botbuilder.core.teams import TeamsInfo

if TYPE_CHECKING:
    from botbuilder.core import TurnContext

from agent_runtime.safety import mask_telemetry
from agent_runtime.transport.teams.events import ConversationRef

logger = logging.getLogger(__name__)

# T-134 -- retry policy for the single Graph retry in `_get_member_with_retry`. Named
# constants because `src/` runs ruff `select = ["ALL"]`, where bare 400/500 comparisons
# trip PLR2004.
_HTTP_CLIENT_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500
_RETRY_BASE_DELAY_SECONDS = 0.25
_RETRY_JITTER_SECONDS = 0.25


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


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a botbuilder/msrest Graph error, or None.

    ``TeamsInfo.get_member`` surfaces ``botbuilder.schema.ErrorResponseException``, a
    subclass of msrest's ``HttpOperationError``, which stores the underlying response on
    ``.response`` (``msrest/exceptions.py:150-158``). That object is ``requests``-shaped
    (``.status_code``) or ``aiohttp``-shaped (``.status``) depending on which sender msrest
    selected, so both are probed -- on the exception itself as well as on ``.response``,
    because a bare ``aiohttp.ClientResponseError`` carries ``.status`` directly and has no
    ``.response`` at all. ``None`` means no status was recoverable -- a connection reset, DNS
    or TLS failure that never produced a response, which is exactly the transient case the
    retry exists for, so ``None`` retries.
    """
    for source in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
    return None


async def _get_member_with_retry(turn_context: TurnContext, member_id: str) -> Any:
    """One retry (T-084b) for TRANSIENT Graph failures only, after a jittered pause (T-134).

    The failure T-084b observed is a transient Graph/Connector error that silently costs
    the user their message. A 4xx is not that. It is either permanent for this member (403
    missing admin consent, 404 unknown member, 401 bad credentials, 400 malformed) or a
    throttle (429) -- and the module header above documents Graph's ~10k req / 10 min
    per-tenant cap on this exact route. The old unconditional retry therefore fired a
    second call on a 429 at the precise moment the tenant's budget was exhausted, doubling
    the request rate that caused the throttle, and re-tried permanent 403s that could never
    succeed. Both now re-raise immediately; ``resolve_identity`` catches, falls back to
    ``from_property``, and drops exactly as before -- same user-visible outcome, one Graph
    call instead of two. 408 is folded into the 4xx skip: Graph does not issue it on this
    route in practice, and one clean inequality is worth more than the special case.

    5xx and status-less errors (connection reset, DNS, TLS -- the T-084b case) keep their
    one retry, now after ~0.25-0.5s. The jitter keeps a tenant's concurrent turns from
    re-hitting Graph in lockstep. Worst case is still 2 Graph calls per activity, and the
    added delay sits far inside the Connector's ~15s redelivery window. Do NOT turn this
    into a loop.

    This deliberately diverges from ``connectors/base.py``'s generic
    ``RETRYABLE_HTTP_STATUS_CODES`` (which treats 408 and 429 as retryable): that policy is for
    httpx-shaped connector calls with their own throttle accounting, whereas this one guards a
    msrest/botbuilder call against a documented per-tenant Graph quota, where retrying a 429 is
    the failure being fixed. There is intentionally no shared retry policy here -- pulling in
    the connector machinery for one optional retry would cost more than it saves.
    """
    try:
        return await TeamsInfo.get_member(turn_context, member_id)
    except Exception as exc:
        status = _http_status(exc)
        if status is not None and _HTTP_CLIENT_ERROR_MIN <= status < _HTTP_SERVER_ERROR_MIN:
            raise
        await asyncio.sleep(
            _RETRY_BASE_DELAY_SECONDS + random.uniform(0, _RETRY_JITTER_SECONDS)  # noqa: S311
        )
        return await TeamsInfo.get_member(turn_context, member_id)


async def resolve_identity(turn_context: TurnContext) -> ConversationRef | None:
    """Return a populated ConversationRef or None if the user cannot be identified."""
    activity = turn_context.activity
    from_info = activity.from_property

    aad_object_id = ""
    email = ""
    display_name = from_info.name or "Teams User"

    try:
        member = await _get_member_with_retry(turn_context, from_info.id)
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
