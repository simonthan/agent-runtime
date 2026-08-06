"""Anthropic API wrapper with opinionated two-cache-breakpoint contract.

Public surface:

- ``AnthropicClient`` — async client with ``complete()`` / ``complete_messages()`` methods
- ``build_anthropic_sdk_client`` — provider factory (public Anthropic API vs Azure AI Foundry)
- ``ClaudeResponse`` — frozen dataclass with token-usage + cache-stats
- ``Message`` / ``History`` — conversation history types
- ``LLMImage`` / ``ANTHROPIC_IMAGE_MEDIA_TYPES`` — base64 image content blocks for the
  first user message (T-067d vision passthrough)
- ``LLMError``, ``LLMRateLimitError``, ``LLMAPIError``, ``LLMResponseError`` — exception hierarchy
- ``ToolUseBlock`` — one tool call the model requested (parsed from content blocks)
- ``ToolUseLoop`` — generic fenced tool-use loop primitive
- ``ToolResult``, ``ToolCall``, ``ToolLoopStep``, ``ToolLoopResult``, ``ToolExecutor`` — loop types
- ``PendingConfirmation``, ``ExecuteDecision``, ``InjectResultDecision``,
  ``ResumeDecision``, ``ConfirmPredicate`` — confirm-before-dispatch (T-025a)
- ``ToolRoundContext``, ``current_tool_round``, ``bind_tool_round`` — the executing round's
  index / remaining tool-round budget, readable by an executor with no signature change (T-115j)

See ``agent_runtime.llm.client.AnthropicClient.complete`` docstring for the
two-breakpoint cache contract (static system prefix + per-turn retrieval block).
"""

from agent_runtime.llm.client import AnthropicClient
from agent_runtime.llm.compaction import (
    CompactionConfig,
    CompactionEngine,
    CompactionResult,
    WorkingMemory,
    estimate_tokens,
)
from agent_runtime.llm.errors import (
    LLMAPIError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from agent_runtime.llm.factory import build_anthropic_sdk_client
from agent_runtime.llm.models import (
    ANTHROPIC_IMAGE_MEDIA_TYPES,
    ClaudeResponse,
    History,
    LLMImage,
    Message,
    ToolUseBlock,
)
from agent_runtime.llm.round_context import (
    ToolRoundContext,
    bind_tool_round,
    current_tool_round,
)
from agent_runtime.llm.tool_loop import (
    ConfirmPredicate,
    ExecuteDecision,
    InjectResultDecision,
    PendingConfirmation,
    ResumeDecision,
    ToolCall,
    ToolExecutor,
    ToolLoopResult,
    ToolLoopStep,
    ToolResult,
    ToolUseLoop,
)

__all__ = [
    "ANTHROPIC_IMAGE_MEDIA_TYPES",
    "AnthropicClient",
    "ClaudeResponse",
    "CompactionConfig",
    "CompactionEngine",
    "CompactionResult",
    "ConfirmPredicate",
    "ExecuteDecision",
    "History",
    "InjectResultDecision",
    "LLMAPIError",
    "LLMError",
    "LLMImage",
    "LLMRateLimitError",
    "LLMResponseError",
    "Message",
    "PendingConfirmation",
    "ResumeDecision",
    "ToolCall",
    "ToolExecutor",
    "ToolLoopResult",
    "ToolLoopStep",
    "ToolResult",
    "ToolRoundContext",
    "ToolUseBlock",
    "ToolUseLoop",
    "WorkingMemory",
    "bind_tool_round",
    "build_anthropic_sdk_client",
    "current_tool_round",
    "estimate_tokens",
]
