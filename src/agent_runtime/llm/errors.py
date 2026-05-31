"""Exception hierarchy for the LLM wrapper.

All exceptions raised by ``AnthropicClient`` derive from ``LLMError``.
Consumers never import from ``anthropic`` to catch wrapper errors —
``__cause__`` chains to the original SDK exception when applicable.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for the LLM wrapper."""


class LLMRateLimitError(LLMError):
    """Raised when the SDK exhausts its retry budget on 429 responses."""


class LLMAPIError(LLMError):
    """Raised on non-retryable SDK API errors (4xx other than 429, 5xx after retries)."""


class LLMResponseError(LLMError):
    """Raised when the SDK returns a response with no content or a non-text first block."""
