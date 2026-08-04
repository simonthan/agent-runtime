"""OutboundChannel Protocol + Bot Framework implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from botbuilder.schema import Activity, ActivityTypes, Attachment

from agent_runtime.safety import mask_telemetry

if TYPE_CHECKING:
    from botbuilder.core import TurnContext

logger = logging.getLogger(__name__)

_ADAPTIVE_CARD_CONTENT_TYPE = "application/vnd.microsoft.card.adaptive"
_OAUTH_CARD_CONTENT_TYPE = "application/vnd.microsoft.card.oauth"


@dataclass(frozen=True)
class SignInResource:
    """Channel-agnostic view of a Bot Framework token-service sign-in resource.

    ``sign_in_link`` is the signed token-service URL
    (``https://token.botframework.com/api/oauth/signin?signature=…``) a consumer places
    in an OAuthCard ``signin`` button. ``token_exchange_uri`` is the resource the silent
    Teams token exchange targets (may be ``None`` when the connection defines none).
    A ``None`` return from ``get_sign_in_resource`` — not this dataclass — signals "no
    resource available"; when returned, ``sign_in_link`` is always non-empty.
    """

    sign_in_link: str | None
    token_exchange_uri: str | None


class OutboundChannel(Protocol):
    """Minimal outbound surface — text, Adaptive Card, OAuth Card, typing, edit-in-place."""

    async def send_text(self, text: str) -> str | None: ...
    async def send_card(self, card: dict) -> None: ...
    async def send_oauth_card(self, card: dict) -> None: ...
    async def send_typing(self) -> None: ...
    async def update_activity(self, activity_id: str, text: str) -> bool: ...
    async def get_sign_in_resource(self, *, connection_name: str) -> SignInResource | None: ...


class BotFrameworkOutboundChannel:
    """Production implementation backed by a botbuilder TurnContext."""

    def __init__(self, turn_context: TurnContext) -> None:
        self._turn_context = turn_context

    async def send_text(self, text: str) -> str | None:
        """Send a message activity; return the channel-assigned activity id.

        The id is what ``update_activity`` targets (T-107). Returns ``None`` when the
        message cannot be edited later, so a consumer's ``if not activity_id`` degrade
        check is sufficient.

        The trailing ``or None`` is load-bearing, not defensive noise: when the Connector
        returns a falsy response, botbuilder substitutes
        ``ResourceResponse(id=activity.id or "")`` (bot_framework_adapter.py:722-723) — and
        ``activity.id`` was already hard-nulled by the outbound validator
        (turn_context.py:191). That path therefore yields the EMPTY STRING, not ``None``.
        Without ``or None`` a caller testing ``activity_id is None`` would sail past the
        guard and edit-loop against an id that can never match.

        Send failures still RAISE — T-089's ``dispatcher._safe_send`` catches them.
        """
        response = await self._turn_context.send_activity(
            Activity(type=ActivityTypes.message, text=text)
        )
        return getattr(response, "id", None) or None

    async def update_activity(self, activity_id: str, text: str) -> bool:
        """Replace the text of a previously sent bot message. Returns True on success.

        Returns ``False`` — never raises — when the id is empty or the Connector edit
        fails (channel quirk, expired/deleted message, 4xx/5xx). The caller's degrade
        signal is the return value, not an exception: T-107's consumer (T-109's progress
        ticker) must fall back to a one-shot notice without killing the turn.

        ``TurnContext.update_activity`` applies the conversation reference to the activity
        before dispatching and does NOT clobber ``id`` (only ``reply_to_id``), so setting
        type/id/text here is sufficient. The adapter resolves the SAME cached
        ``ConnectorClient`` from ``turn_state`` that ``send_activities`` uses, so this works
        identically on the T-089 detached-turn path.

        ``CancelledError`` is a ``BaseException`` and deliberately propagates — T-109
        cancels the progress ticker in a ``finally``.
        """
        if not activity_id:
            return False
        try:
            await self._turn_context.update_activity(
                Activity(type=ActivityTypes.message, id=activity_id, text=text)
            )
        except Exception as exc:  # noqa: BLE001 — a failed edit degrades the caller, never the turn
            # mask_telemetry (not exc_info=True): the msrest/Connector error can embed the
            # service URL, which carries tenant/conversation segments (T-021a precedent,
            # same treatment as get_sign_in_resource below).
            logger.warning(
                "update_activity failed (%s): %s",
                type(exc).__name__,
                mask_telemetry(str(exc)),
            )
            return False
        return True

    async def send_card(self, card: dict) -> None:
        attachment = Attachment(content_type=_ADAPTIVE_CARD_CONTENT_TYPE, content=card)
        await self._turn_context.send_activity(
            Activity(type=ActivityTypes.message, attachments=[attachment])
        )

    async def send_oauth_card(self, card: dict) -> None:
        attachment = Attachment(content_type=_OAUTH_CARD_CONTENT_TYPE, content=card)
        await self._turn_context.send_activity(
            Activity(type=ActivityTypes.message, attachments=[attachment])
        )

    async def send_typing(self) -> None:
        await self._turn_context.send_activity(Activity(type=ActivityTypes.typing))

    async def get_sign_in_resource(self, *, connection_name: str) -> SignInResource | None:
        """Fetch a token-service sign-in resource for the current turn's user via the
        BotFrameworkAdapter. Returns ``None`` (so the caller sends NO OAuth card) when a
        resource can't be obtained — an OAuth card without a valid signin link renders
        as an unsupported card in Teams and blocks the silent token exchange (T-075).

        The user id passed to the token service is the Bot Framework channel-account id
        (``activity.from_property.id``), not the Entra OID. Accessed defensively so a
        proactive/continued turn context (no ``get_sign_in_resource_from_user``) yields
        ``None`` instead of raising. The live token-service call is wrapped so a
        transient failure (5xx/network) ALSO yields ``None`` — the caller must be free to
        skip the card and still answer the turn (the eager SSO prompt runs before
        ``process_turn`` in the dispatcher, so a raise here would strand the user)."""
        if not connection_name:
            return None
        activity = self._turn_context.activity
        user_id = getattr(getattr(activity, "from_property", None), "id", None)
        adapter = getattr(self._turn_context, "adapter", None)
        get = getattr(adapter, "get_sign_in_resource_from_user", None)
        if not user_id or get is None:
            return None
        try:
            resp = await get(self._turn_context, connection_name, user_id)
        except Exception as exc:  # noqa: BLE001 — token-service blip must not strand the turn (fail-safe)
            # mask_telemetry (not exc_info=True): the underlying msrest/Graph HTTP error can
            # embed the token-service request URL, which may carry an Entra OID or tenant
            # GUID segment (T-021a telemetry-leak precedent — see identity.py, connectors/base.py).
            logger.warning(
                "Token-service get_sign_in_resource failed (connection=%s): %s",
                connection_name,
                mask_telemetry(str(exc)),
            )
            return None
        sign_in_link = getattr(resp, "sign_in_link", None)
        if not sign_in_link:
            return None
        ter = getattr(resp, "token_exchange_resource", None)
        return SignInResource(
            sign_in_link=sign_in_link,
            token_exchange_uri=getattr(ter, "uri", None) if ter is not None else None,
        )
