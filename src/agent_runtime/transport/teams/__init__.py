"""Microsoft Teams transport for agent-runtime.

Fresh-write subpackage (not a verbatim lift from ithelpdesk). Code passes
the project's ``select = ["ALL"]`` ruff config without per-file-ignores;
keep it that way unless an exception is documented here.

Public surface:
- ``TeamsAdapter`` + ``TeamsAdapterConfig`` — wraps BotFrameworkAdapter
- ``TeamsHandler`` Protocol — consumer implements ``on_event``
- ``OutboundChannel`` Protocol + ``BotFrameworkOutboundChannel`` impl
- ``ConversationRef`` + ``FileAttachment`` + ``InlineImageAttachment`` +
  ``InboundMessage`` / ``InboundMembersAdded`` / ``InboundInvoke``
- ``InvokeResponse`` (re-exported from botbuilder.schema for invoke return values)
- ``download_inline_image`` + ``BotFrameworkCredentials`` + ``DownloadedImage`` +
  ``InlineImageDownloadError`` — authenticated inline-image download (T-067a)

Testing helpers in ``agent_runtime.transport.teams.testing``:
- ``FakeOutboundChannel``, ``make_file_attachment``, ``make_inline_image``,
  ``make_inbound_message``, ``make_inbound_members_added``, ``make_inbound_invoke``
"""

from botbuilder.schema import InvokeResponse

from agent_runtime.transport.teams.adapter import TeamsAdapter, TeamsAdapterConfig
from agent_runtime.transport.teams.events import (
    ConversationRef,
    FileAttachment,
    InboundEvent,
    InboundInvoke,
    InboundMembersAdded,
    InboundMessage,
    InlineImageAttachment,
    conversation_ref_from_dict,
    conversation_ref_to_dict,
)
from agent_runtime.transport.teams.images import (
    BotFrameworkCredentials,
    DownloadedImage,
    InlineImageDownloadError,
    download_inline_image,
)
from agent_runtime.transport.teams.outbound import (
    BotFrameworkOutboundChannel,
    OutboundChannel,
    SignInResource,
)
from agent_runtime.transport.teams.protocol import TeamsHandler

__all__ = [
    "BotFrameworkCredentials",
    "BotFrameworkOutboundChannel",
    "ConversationRef",
    "DownloadedImage",
    "FileAttachment",
    "InboundEvent",
    "InboundInvoke",
    "InboundMembersAdded",
    "InboundMessage",
    "InlineImageAttachment",
    "InlineImageDownloadError",
    "InvokeResponse",
    "OutboundChannel",
    "SignInResource",
    "TeamsAdapter",
    "TeamsAdapterConfig",
    "TeamsHandler",
    "conversation_ref_from_dict",
    "conversation_ref_to_dict",
    "download_inline_image",
]
