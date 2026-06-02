"""OutboundChannel Protocol + Bot Framework implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from botbuilder.schema import Activity, ActivityTypes, Attachment

if TYPE_CHECKING:
    from botbuilder.core import TurnContext

_ADAPTIVE_CARD_CONTENT_TYPE = "application/vnd.microsoft.card.adaptive"


class OutboundChannel(Protocol):
    """Minimal outbound surface — text, Adaptive Card, typing indicator."""

    async def send_text(self, text: str) -> None: ...
    async def send_card(self, card: dict) -> None: ...
    async def send_typing(self) -> None: ...


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

    async def send_typing(self) -> None:
        await self._turn_context.send_activity(Activity(type=ActivityTypes.typing))
