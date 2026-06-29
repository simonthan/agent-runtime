"""TeamsAdapter — BotFrameworkAdapter wrapper + Activity → InboundEvent dispatch."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from botbuilder.core import (
    ActivityHandler,
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import (
    Activity,
    ChannelAccount,
    ConversationAccount,
    ConversationReference,
    InvokeResponse,
)

from agent_runtime.transport.teams.events import (
    ConversationRef,
    FileAttachment,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
)
from agent_runtime.transport.teams.identity import resolve_identity
from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_runtime.transport.teams.protocol import TeamsHandler

logger = logging.getLogger(__name__)

# Teams delivers a 1:1 chat file upload as an attachment with this contentType;
# its `content.uniqueId` is the OneDrive driveItem id (the read-on-demand key).
_TEAMS_FILE_DOWNLOAD_INFO = "application/vnd.microsoft.teams.file.download.info"


def _extract_file_attachments(raw: list | None) -> tuple[FileAttachment, ...]:
    """Pull Teams file uploads off an inbound activity.

    Surfaces ONLY attachments that (a) carry the Teams file-download contentType,
    (b) have a parseable ``content`` (dict, or a JSON-string an upstream serializer
    left unparsed), and (c) expose a non-empty ``uniqueId`` (the OneDrive item id
    required to read the file back). Inline images, Adaptive Cards, and link
    unfurls are ignored, so a message with no readable file attachment yields an
    empty tuple — byte-identical to the prior behaviour. A file-download attachment
    whose content can't be parsed is logged at debug and skipped (observable, not a
    silent vanish)."""
    if not raw:
        return ()
    out: list[FileAttachment] = []
    for a in raw:
        if getattr(a, "content_type", None) != _TEAMS_FILE_DOWNLOAD_INFO:
            continue
        content = getattr(a, "content", None)
        if isinstance(content, str):  # botbuilder does not recursively parse string content
            try:
                content = json.loads(content)
            except (ValueError, TypeError):
                logger.debug("Unparseable file.download.info content; skipping attachment")
                continue
        if not isinstance(content, dict):
            continue
        item_id = content.get("uniqueId")
        if not isinstance(item_id, str) or not item_id:
            # A non-string uniqueId (adversarial JSON) is not a readable driveItem
            # id; coercing it would smuggle junk into FileAttachment.item_id.
            continue
        file_type = content.get("fileType")
        download_url = content.get("downloadUrl")
        out.append(
            FileAttachment(
                item_id=item_id,
                name=getattr(a, "name", None) or "",
                file_type=file_type.lower() if isinstance(file_type, str) else "",
                download_url=download_url if isinstance(download_url, str) else "",
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class TeamsAdapterConfig:
    app_id: str
    app_password: str
    tenant_id: str
    on_turn_error: Callable[[TurnContext, Exception], Awaitable[None]] | None = None


class _EventDispatchingHandler(ActivityHandler):
    """Internal ActivityHandler that converts botbuilder activities to InboundEvents."""

    def __init__(self, handler: TeamsHandler) -> None:
        super().__init__()
        self._handler = handler
        self._invoke_response: InvokeResponse | None = None

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        ref = await resolve_identity(turn_context)
        if ref is None:
            return
        # Non-dict activity.value (e.g. typed MessagingExtensionQuery objects in
        # botbuilder ≥4.16) is intentionally coerced to None until a richer event
        # type is added — v0.4.0 scope is dict-shaped Action.Submit payloads only.
        raw_value = turn_context.activity.value
        # Strip the bot's own @mention so channel text arrives clean ("@Bot hi" -> "hi").
        # remove_recipient_mention is a no-op in 1:1 DMs (no recipient-mention entity), so the
        # personal-chat path is byte-equivalent. Guard None: it returns activity.text verbatim,
        # which may be None.
        mention_stripped = TurnContext.remove_recipient_mention(turn_context.activity) or ""
        event = InboundMessage(
            conversation_ref=ref,
            text=mention_stripped.strip(),
            value=raw_value if isinstance(raw_value, dict) else None,
            attachments=_extract_file_attachments(turn_context.activity.attachments),
        )
        await self._handler.on_event(event, BotFrameworkOutboundChannel(turn_context))

    async def on_members_added_activity(
        self, members_added: list, turn_context: TurnContext
    ) -> None:
        ref = await resolve_identity(turn_context)
        if ref is None:
            return
        bot_id = (turn_context.activity.recipient.id or "").lower()
        # Exclude the bot's own entry — its `m.id` is a Bot Framework app-ID
        # ("28:<guid>"), not an Entra Object ID. Case-insensitive compare because
        # Teams occasionally normalizes one side differently than the other.
        # Bot presence is signalled via bot_was_added instead.
        human_members = [m for m in members_added if (m.id or "").lower() != bot_id]
        # Drop members without aad_object_id (rare: guests, federated accounts) —
        # falling back to m.id would emit Bot Framework channel IDs (e.g. "29:<base64>")
        # which T-008g consumers can't pass to transitiveMemberOf without 404ing.
        added_ids = tuple(m.aad_object_id for m in human_members if m.aad_object_id)
        event = InboundMembersAdded(
            conversation_ref=ref,
            added_aad_object_ids=added_ids,
            bot_was_added=any((m.id or "").lower() == bot_id for m in members_added),
        )
        await self._handler.on_event(event, BotFrameworkOutboundChannel(turn_context))

    async def on_invoke_activity(self, turn_context: TurnContext) -> InvokeResponse:
        ref = await resolve_identity(turn_context)
        if ref is None:
            return InvokeResponse(status=401)
        event = InboundInvoke(
            conversation_ref=ref,
            name=turn_context.activity.name or "",
            value=(
                turn_context.activity.value
                if isinstance(turn_context.activity.value, dict)
                else None
            ),
        )
        result = await self._handler.on_event(event, BotFrameworkOutboundChannel(turn_context))
        return result if isinstance(result, InvokeResponse) else InvokeResponse(status=200)


class TeamsAdapter:
    """Wraps BotFrameworkAdapter; exposes a HTTP-framework-agnostic entry point."""

    def __init__(self, config: TeamsAdapterConfig, handler: TeamsHandler) -> None:
        self._adapter = BotFrameworkAdapter(
            BotFrameworkAdapterSettings(
                app_id=config.app_id,
                app_password=config.app_password,
                channel_auth_tenant=config.tenant_id,
            )
        )
        self._adapter.on_turn_error = config.on_turn_error or self._default_on_turn_error
        self._handler = _EventDispatchingHandler(handler)

    @staticmethod
    async def _default_on_turn_error(_context: TurnContext, error: Exception) -> None:
        logger.exception("Unhandled error in Teams handler", exc_info=error)

    async def process_activity(
        self,
        activity_body: dict[str, Any],
        auth_header: str,
    ) -> tuple[int, dict[str, Any] | None]:
        """Entry point invoked by the consumer's HTTP webhook route.

        Returns ``(status_code, response_body_or_None)`` — translate to the
        consumer's HTTP framework's response object.

        Raises ``ValueError`` if ``auth_header`` is empty or whitespace-only.
        Some botbuilder versions silently skip JWT validation when given an empty
        header; a whitespace-only header (`" "`) is truthy but effectively empty
        downstream, so we strip-check too (SEC-5). We fail loudly to catch consumer
        HTTP routes that forget to forward the inbound ``Authorization`` header.
        """
        if not auth_header or not auth_header.strip():
            msg = (
                "auth_header is required; pass the inbound Authorization header verbatim "
                "(including the 'Bearer ' prefix). An empty header would bypass JWT validation."
            )
            raise ValueError(msg)
        activity = Activity().deserialize(activity_body)
        response = await self._adapter.process_activity(
            activity, auth_header, self._handler.on_turn
        )
        if response is None:
            return (201, None)
        return (response.status, response.body)

    async def send_proactive(
        self,
        ref: ConversationRef,
        *,
        bot_app_id: str,
        text: str | None = None,
        card: dict[str, Any] | None = None,
    ) -> None:
        """Send an unsolicited (proactive) message into an existing 1:1 Teams chat.

        Reconstructs a canonical botbuilder ``ConversationReference`` from the
        stored ``ref`` and drives ``BotFrameworkAdapter.continue_conversation``,
        which manufactures a synthetic ``TurnContext`` routed to
        ``ref.service_url``. The callback reuses the same
        ``BotFrameworkOutboundChannel`` surface as the inbound path, so text and
        Adaptive Cards render identically whether solicited or proactive.

        ``ref.user_channel_id`` / ``ref.recipient_id`` are the Bot Framework
        channel-account ids captured on inbound; they fill the reference's
        ``user`` / ``bot`` per botbuilder's continuation contract
        (``get_continuation_activity`` maps user->from_property, bot->recipient).
        References persisted before those fields existed deserialize with empty
        ids — we fall back to the Entra OID / ``28:<bot_app_id>`` so an upgraded
        deploy can still message pre-existing users (1:1 routing keys off
        ``conversation.id`` + ``service_url`` regardless).

        Consent is structural: a proactive message is only deliverable when the
        caller already holds a ``ConversationRef`` — Teams grants one only after
        the user has messaged the bot. ``bot_app_id`` is the bot's Entra app
        (client) ID — botbuilder uses it to mint the outbound Connector token. At
        least one of ``text`` / ``card`` must be provided; passing both sends two
        activities in order (text first).
        """
        if text is None and card is None:
            msg = "send_proactive requires text and/or card"
            raise ValueError(msg)

        reference = ConversationReference(
            channel_id=ref.channel_id or "msteams",
            service_url=ref.service_url,
            conversation=ConversationAccount(id=ref.conversation_id),
            user=ChannelAccount(
                id=ref.user_channel_id or ref.aad_object_id,
                name=ref.user_display_name,
            ),
            bot=ChannelAccount(id=ref.recipient_id or f"28:{bot_app_id}"),
        )

        async def _callback(turn_context: TurnContext) -> None:
            channel = BotFrameworkOutboundChannel(turn_context)
            if text is not None:
                await channel.send_text(text)
            if card is not None:
                await channel.send_card(card)

        await self._adapter.continue_conversation(reference, _callback, bot_app_id)
