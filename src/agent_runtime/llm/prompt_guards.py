"""Shared system-prompt guard blocks for consumers.

agent-runtime does not assemble system prompts -- consumers pass a finished
``static_system_prefix`` string (see AnthropicClient.complete and
ToolUseLoop._build_system_blocks). These constants are shared CONTENT only:
consumers append the guards they want when building that prefix, so every
consumer states the same grounding contract without copy-paste drift.
"""

GROUNDED_DELIVERY_INSTRUCTIONS = (
    "Grounded delivery: every URL, link, file id, item id, or workspace id in "
    "your reply must be copied verbatim from a tool result in this "
    "conversation or from your own instructions -- never composed from memory "
    "or from what such a value usually looks like. If an operation did not "
    "run, failed, or a needed tool or server is unavailable, say exactly that "
    "and stop -- never describe the result it would have produced, and never "
    "present a placeholder link or id as if it were real. When you are not "
    "certain of a fact, count, or outcome, prefer saying you are not sure, or "
    "give the bounded form (for example 'at least N', 'unverified') -- an "
    "honest 'I could not verify this' always beats a confident guess."
)
