"""Microsoft Teams transport for agent-runtime.

Fresh-write subpackage (not a verbatim lift from ithelpdesk). Code passes
the project's ``select = ["ALL"]`` ruff config without per-file-ignores;
keep it that way unless an exception is documented here.

Public surface:
- ``TeamsAdapter`` + ``TeamsAdapterConfig`` — wraps BotFrameworkAdapter
- ``TeamsHandler`` Protocol — consumer implements ``on_event``
- ``OutboundChannel`` Protocol + ``BotFrameworkOutboundChannel`` impl
- ``ConversationRef`` + ``InboundMessage`` / ``InboundMembersAdded`` / ``InboundInvoke``
- ``InvokeResponse`` (re-exported from botbuilder.schema for invoke return values)

Testing helpers in ``agent_runtime.transport.teams.testing``:
- ``FakeOutboundChannel``, ``make_inbound_message``, ``make_inbound_members_added``,
  ``make_inbound_invoke``
"""

from botbuilder.schema import InvokeResponse

from agent_runtime.transport.teams.adapter import TeamsAdapter, TeamsAdapterConfig
from agent_runtime.transport.teams.events import (
    ConversationRef,
    InboundEvent,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
    conversation_ref_from_dict,
    conversation_ref_to_dict,
)
from agent_runtime.transport.teams.outbound import BotFrameworkOutboundChannel, OutboundChannel
from agent_runtime.transport.teams.protocol import TeamsHandler

__all__ = [
    "BotFrameworkOutboundChannel",
    "ConversationRef",
    "InboundEvent",
    "InboundInvoke",
    "InboundMembersAdded",
    "InboundMessage",
    "InvokeResponse",
    "OutboundChannel",
    "TeamsAdapter",
    "TeamsAdapterConfig",
    "TeamsHandler",
    "conversation_ref_from_dict",
    "conversation_ref_to_dict",
]
