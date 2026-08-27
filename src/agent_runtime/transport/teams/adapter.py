"""TeamsAdapter — BotFrameworkAdapter wrapper + Activity → InboundEvent dispatch."""

from __future__ import annotations

import asyncio
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
    ActivityTypes,  # T-286: used in best-effort reply
    ChannelAccount,
    ConversationAccount,
    ConversationReference,
    InvokeResponse,
)
from botframework.connector.auth import AuthenticationConstants

from agent_runtime.safety import mask_telemetry
from agent_runtime.transport.teams._msal import BoundedAppCredentials
from agent_runtime.transport.teams.events import (
    ConversationRef,
    FileAttachment,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
    InlineImageAttachment,
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

# T-286: best-effort reply when identity resolution fails.
_IDENTITY_FAIL_REPLY = (
    "I'm unable to process messages in this conversation. "
    "Please try messaging me in a direct chat."
)


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


def _extract_inline_images(raw: list | None) -> tuple[InlineImageAttachment, ...]:
    """Pull Teams inline images (camera captures, pasted photos) off an inbound activity.

    Surfaces attachments whose declared ``content_type`` is the literal
    ``"image/*"`` Teams sends for a camera capture, or starts with ``"image/"``
    for clients that send a concrete mime, AND expose a non-empty, ``https``
    ``content_url``. File-download-info attachments, Adaptive Cards, and link
    unfurls are ignored — the two extractors are disjoint by contentType, same
    as ``_extract_file_attachments``. A missing, non-string, or non-``https``
    ``content_url`` is logged at debug and skipped (observable, not a silent
    vanish) — the download helper later enforces the stricter Bot Framework
    host allowlist; this extractor only enforces the transport scheme."""
    if not raw:
        return ()
    out: list[InlineImageAttachment] = []
    for a in raw:
        # Teams sends the literal "image/*" for camera captures; some clients send
        # a concrete mime (e.g. "image/png") — "image/*".startswith("image/") is
        # True, so this single check covers both per the plan's matching rule.
        content_type = getattr(a, "content_type", None)
        if not isinstance(content_type, str) or not content_type.startswith("image/"):
            continue
        content_url = getattr(a, "content_url", None)
        if not isinstance(content_url, str) or not content_url.startswith("https://"):
            logger.debug("Inline image missing a https content_url; skipping attachment")
            continue
        out.append(
            InlineImageAttachment(
                content_url=content_url,
                content_type=content_type,
                name=getattr(a, "name", None) or "",
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
            # T-286: best-effort reply — user must never lose a message silently.
            try:
                await turn_context.send_activity(
                    Activity(type=ActivityTypes.message, text=_IDENTITY_FAIL_REPLY)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "identity_fail_reply_send_failed from_id=%s error=%s",
                    turn_context.activity.from_property.id,
                    mask_telemetry(str(exc)),
                )
            # Notify consumer handler if it implements the optional audit hook.
            hook = getattr(self._handler, "on_identity_failed", None)
            if hook is not None:
                activity = turn_context.activity
                try:
                    await hook(
                        from_id=getattr(activity.from_property, "id", "") or "",
                        aad_object_id=getattr(activity.from_property, "aad_object_id", "") or "",
                        conversation_type=getattr(
                            activity.conversation, "conversation_type", ""
                        ) or "unknown",
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("on_identity_failed_hook_raised")
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
            images=_extract_inline_images(turn_context.activity.attachments),
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
        # T-115m -- without this the SDK builds its own timeout-less
        # MicrosoftAppCredentials (`bot_framework_adapter.py:1352-1380`) and
        # mints the connector token synchronously inside msrest's async
        # pipeline (`msrest/pipeline/async_requests.py:99`) -- i.e. on the
        # event loop, unbounded, on EVERY outbound activity. A caller-supplied
        # AppCredentials is returned verbatim ahead of that build path (`:1364`).
        # Construction stays network-free; the MSAL app is built on first send.
        # T-119 -- held on the instance so `warm_connector_token` can prime the very
        # object every outbound path mints from (`bot_framework_adapter.py:199`, `:1365`).
        self._credentials = BoundedAppCredentials(
            config.app_id,
            config.app_password,
            channel_auth_tenant=config.tenant_id,
        )
        self._adapter = BotFrameworkAdapter(
            BotFrameworkAdapterSettings(
                app_id=config.app_id,
                app_password=config.app_password,
                channel_auth_tenant=config.tenant_id,
                app_credentials=self._credentials,
            )
        )
        self._adapter.on_turn_error = config.on_turn_error or self._default_on_turn_error
        self._handler = _EventDispatchingHandler(handler)

    @staticmethod
    async def _default_on_turn_error(_context: TurnContext, error: Exception) -> None:
        logger.exception("Unhandled error in Teams handler", exc_info=error)

    async def warm_connector_token(self) -> bool:
        """Mint or refresh the outbound connector token on a worker thread (T-119).

        T-115m bounded this token acquisition; it did NOT move it off the event loop.
        ``msrest``'s ``AsyncRequestsCredentialsPolicy.send`` calls
        ``signed_session`` synchronously inside an ``async def``
        (``msrest/pipeline/async_requests.py:99``), so every outbound activity can
        block the loop for as long as MSAL's HTTP timeout allows. MSAL reaches the
        network more often than "once per process": it stops serving a cached token
        5 minutes before expiry (``msal/application.py:1652``) and refreshes
        proactively once AAD's ``refresh_in`` elapses (``:1662-1665``, typically half
        the token lifetime).

        Calling this first makes the SDK's on-loop call an in-memory cache lookup.
        It is a cache-priming optimisation with a fallback, NOT a gate: on failure the
        send still mints inline (bounded by T-115m), so a warm failure must never fail
        the turn. Returns True when the token call completed without raising.

        KNOWN RESIDUAL: when AAD is failing but a valid-but-aging token is still cached,
        msal swallows the refresh error and returns the cached token
        (``msal/application.py:1721-1727``) without clearing ``refresh_on``. This method
        then returns True having logged nothing, and the SDK's on-loop call repeats the
        same failing request. Loop-blocking is no worse than before this change, but it
        is not fixed either — see the T-119 plan, Design section 2.

        Public so a consumer may additionally call it from its startup lifespan; the
        two entry points below already cover every outbound path, because
        ``BotFrameworkAdapter`` returns caller-supplied credentials for every connector
        client it builds (``bot_framework_adapter.py:1359-1367``).
        """
        app_id = self._credentials.microsoft_app_id
        if not app_id or app_id == AuthenticationConstants.ANONYMOUS_SKILL_APP_ID:
            # Mirrors the SDK's own gate LITERALLY (`AppCredentials._should_set_token`,
            # `app_credentials.py:95-102`) -- note it does NOT consult the password.
            # Gating on the password too would silently skip the warm for a blank-secret
            # misconfiguration, suppressing the one warning that would diagnose it.
            return False
        try:
            await asyncio.to_thread(self._credentials.get_access_token)
        except Exception:  # noqa: BLE001 -- a pre-warm failure must never fail the turn
            logger.warning(
                "Connector token pre-warm failed; the outbound send will mint inline",
                exc_info=True,
            )
            return False
        return True

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

        Pre-warms the outbound connector token on a worker thread (T-119) from INSIDE the
        SDK callback, which runs only after JWT validation has accepted ``auth_header``
        (T-134). The warm never raises, so a token failure degrades to an inline mint
        rather than failing the turn.
        """
        if not auth_header or not auth_header.strip():
            msg = (
                "auth_header is required; pass the inbound Authorization header verbatim "
                "(including the 'Bearer ' prefix). An empty header would bypass JWT validation."
            )
            raise ValueError(msg)
        activity = Activity().deserialize(activity_body)

        async def _warm_then_turn(turn_context: TurnContext) -> Any:
            # T-134 -- the warm lives INSIDE the callback, which the SDK invokes only after
            # `JwtTokenValidation.authenticate_request` has accepted `auth_header`. It used
            # to run before both the deserialize and that validation, gated on nothing but a
            # non-empty header string -- so any POST carrying `Authorization: Bearer
            # <garbage>` spent a worker thread and, on a cold or aging MSAL cache, a real
            # round trip to login.microsoftonline.com, entirely unauthenticated.
            # T-119's property is preserved: the SDK sends no activity before invoking this
            # callback, so the token is still cached before msrest's on-loop `signed_session`
            # call. Never raises. The callback's return value is passed straight through --
            # the SDK reads the invoke response off turn_state, but returning it costs
            # nothing and keeps this a pure interposition.
            await self.warm_connector_token()
            return await self._handler.on_turn(turn_context)

        response = await self._adapter.process_activity(activity, auth_header, _warm_then_turn)
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

        # T-119 -- same reason as `process_activity`: `continue_conversation` mints the
        # connector token on the event loop via msrest's sync `signed_session` call.
        await self.warm_connector_token()

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
