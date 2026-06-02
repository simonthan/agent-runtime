"""TeamsAdapter — BotFrameworkAdapter wrapper + Activity → InboundEvent dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from botbuilder.core import (
    ActivityHandler,
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)
from botbuilder.schema import Activity, InvokeResponse

from agent_runtime.transport.teams.events import InboundInvoke, InboundMembersAdded, InboundMessage
from agent_runtime.transport.teams.identity import resolve_identity
from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_runtime.transport.teams.protocol import TeamsHandler

logger = logging.getLogger(__name__)


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
        event = InboundMessage(
            conversation_ref=ref,
            text=(turn_context.activity.text or "").strip(),
            value=raw_value if isinstance(raw_value, dict) else None,
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

        Raises ``ValueError`` if ``auth_header`` is empty. Some botbuilder
        versions silently skip JWT validation when given an empty header;
        we fail loudly to catch consumer HTTP routes that forget to forward
        the inbound ``Authorization`` header.
        """
        if not auth_header:
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
