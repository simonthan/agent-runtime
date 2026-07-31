"""OutboundChannel Protocol + Bot Framework implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from botbuilder.schema import Activity, ActivityTypes, Attachment

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
    """Minimal outbound surface — text, Adaptive Card, OAuth Card, typing indicator."""

    async def send_text(self, text: str) -> None: ...
    async def send_card(self, card: dict) -> None: ...
    async def send_oauth_card(self, card: dict) -> None: ...
    async def send_typing(self) -> None: ...
    async def get_sign_in_resource(self, *, connection_name: str) -> SignInResource | None: ...


class BotFrameworkOutboundChannel:
    """Production implementation backed by a botbuilder TurnContext."""

    def __init__(self, turn_context: TurnContext) -> None:
        self._turn_context = turn_context

    async def send_text(self, text: str) -> None:
        await self._turn_context.send_activity(Activity(type=ActivityTypes.message, text=text))

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
        except Exception:  # noqa: BLE001 — token-service blip must not strand the turn (fail-safe)
            logger.warning(
                "Token-service get_sign_in_resource failed (connection=%s)",
                connection_name,
                exc_info=True,
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
