"""TeamsHandler Protocol — implemented by consumers (e.g. T-008e dispatcher)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from botbuilder.schema import InvokeResponse

    from agent_runtime.transport.teams.events import InboundEvent
    from agent_runtime.transport.teams.outbound import OutboundChannel


class TeamsHandler(Protocol):
    """Consumer-implemented handler invoked once per inbound event.

    Return ``InvokeResponse`` for ``InboundInvoke`` events; return ``None``
    for ``InboundMessage`` and ``InboundMembersAdded``.
    """

    async def on_event(
        self,
        event: InboundEvent,
        outbound: OutboundChannel,
    ) -> InvokeResponse | None: ...
