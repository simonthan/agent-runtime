"""Anthropic API wrapper with opinionated two-cache-breakpoint contract.

Public surface:

- ``AnthropicClient`` — async client with ``complete()`` / ``complete_messages()`` methods
- ``ClaudeResponse`` — frozen dataclass with token-usage + cache-stats
- ``Message`` / ``History`` — conversation history types
- ``LLMError``, ``LLMRateLimitError``, ``LLMAPIError``, ``LLMResponseError`` — exception hierarchy
- ``ToolUseBlock`` — one tool call the model requested (parsed from content blocks)
- ``ToolUseLoop`` — generic fenced tool-use loop primitive
- ``ToolResult``, ``ToolCall``, ``ToolLoopStep``, ``ToolLoopResult``, ``ToolExecutor`` — loop types

See ``agent_runtime.llm.client.AnthropicClient.complete`` docstring for the
two-breakpoint cache contract (static system prefix + per-turn retrieval block).
"""

from agent_runtime.llm.client import AnthropicClient
from agent_runtime.llm.errors import (
    LLMAPIError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
)
from agent_runtime.llm.models import ClaudeResponse, History, Message, ToolUseBlock
from agent_runtime.llm.tool_loop import (
    ToolCall,
    ToolExecutor,
    ToolLoopResult,
    ToolLoopStep,
    ToolResult,
    ToolUseLoop,
)

__all__ = [
    "AnthropicClient",
    "ClaudeResponse",
    "History",
    "LLMAPIError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseError",
    "Message",
    "ToolCall",
    "ToolExecutor",
    "ToolLoopResult",
    "ToolLoopStep",
    "ToolResult",
    "ToolUseBlock",
    "ToolUseLoop",
]
