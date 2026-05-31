"""``AnthropicClient`` — opinionated wrapper around the Anthropic SDK.

Encodes ARCHITECTURE.md §4 decision #5: every request carries exactly two
``cache_control`` ephemeral breakpoints — one on the static system prefix,
one on the per-turn retrieval block. Prompt caching is GA on the Anthropic
API (since SDK 0.39) — no ``anthropic-beta`` header is sent.

The SDK client is dependency-injected via constructor (real ``AsyncAnthropic``
or a fake satisfying the same structural Protocol). This enables (a) shared
``httpx`` connection pooling across many wrapper instances and (b) deterministic
tests without monkeypatching the SDK.

The wrapper is bot-agnostic — it has no knowledge of ``bot_id`` or ``user_id``.
Per-tenant budget enforcement happens at the service-layer call site, not here.

Cache hits require **byte-for-byte identical** ``static_system_prefix`` AND
identical ``model`` across turns. A per-call ``model=`` override (e.g.,
switching mid-conversation from Sonnet to Haiku) **fragments** the cache and
fires ``llm_cache_not_written`` on the first call to the new model — that's
expected, not a bug.

Callers are responsible for sanitizing ``user_message`` and ``retrieval_block``
against prompt injection. Use ``agent_runtime.safety.sanitize_for_llm_prompt``
and/or ``agent_runtime.safety.InjectionDetector`` at the call site; the wrapper
intentionally does NOT sanitize so the layer of defense is auditable.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from anthropic import APIError, RateLimitError

from agent_runtime.llm.errors import (
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
)
from agent_runtime.llm.models import ClaudeResponse, History
from agent_runtime.logging import AuditLogger, NullAuditLogger

__all__ = ["AnthropicClient"]


_CACHE_MIN_TOKENS: dict[str, int] = {
    # Trailing hyphen disambiguates future variants — ``claude-sonnet`` would
    # match e.g. a hypothetical ``claude-sonnet-small`` with a different threshold;
    # ``claude-sonnet-`` only matches versioned ids like ``claude-sonnet-4-6``.
    "claude-sonnet-": 1024,
    "claude-opus-": 4096,
    "claude-haiku-": 2048,
}
_DEFAULT_CACHE_HINT = 1024


def _cache_threshold_hint(model: str) -> tuple[int, bool]:
    """Return ``(min_tokens, model_unknown)`` for the given model name.

    Uses longest-prefix match against ``_CACHE_MIN_TOKENS``. Unknown models
    fall back to ``_DEFAULT_CACHE_HINT`` with ``model_unknown=True``.
    """
    matches = [(prefix, n) for prefix, n in _CACHE_MIN_TOKENS.items() if model.startswith(prefix)]
    if not matches:
        return _DEFAULT_CACHE_HINT, True
    _, n = max(matches, key=lambda pair: len(pair[0]))
    return n, False


class _AnthropicMessages(Protocol):
    """Structural Protocol for the SDK's ``client.messages`` attribute."""

    async def create(self, **kwargs: Any) -> Any: ...


class _AnthropicAPI(Protocol):
    """Structural Protocol for an Anthropic SDK client.

    Both real ``anthropic.AsyncAnthropic`` and ``tests/unit/llm/fakes.FakeAsyncAnthropic``
    satisfy this Protocol structurally.

    Note: ``ty`` (the type checker) may flag call-sites that pass a real
    ``AsyncAnthropic`` instance because the SDK's ``messages.create`` uses
    explicit keyword params rather than ``**kwargs``. The runtime contract holds;
    add ``# type: ignore[arg-type]`` at the call site if ``ty`` complains.
    """

    messages: _AnthropicMessages


class AnthropicClient:
    """Opinionated async wrapper around the Anthropic SDK.

    See module docstring for the two-cache-breakpoint contract.
    """

    def __init__(
        self,
        *,
        client: _AnthropicAPI,
        default_model: str = "claude-sonnet-4-6",
        default_max_tokens: int = 4096,
        default_temperature: float = 0.0,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._audit: AuditLogger = audit_logger or NullAuditLogger()

    async def complete(
        self,
        *,
        static_system_prefix: str,
        user_message: str,
        dynamic_system_suffix: str | None = None,
        history: History = (),
        retrieval_block: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
    ) -> ClaudeResponse:
        """Send a completion request with two ``cache_control`` ephemeral breakpoints.

        Breakpoint #1: ``static_system_prefix`` (always cached).
        Breakpoint #2: ``retrieval_block`` (cached when present; prepended to user_message).

        ``dynamic_system_suffix`` and ``history`` are passed through uncached.
        """
        chosen_model = model or self._default_model
        chosen_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        chosen_temperature = temperature if temperature is not None else self._default_temperature

        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": static_system_prefix,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if dynamic_system_suffix:
            # Truthy check (not ``is not None``) so empty string is skipped —
            # Anthropic API rejects ``{"type":"text","text":""}`` with 400.
            system_blocks.append({"type": "text", "text": dynamic_system_suffix})

        user_content: list[dict[str, Any]] = []
        if retrieval_block:
            # Truthy check — see ``dynamic_system_suffix`` comment above.
            user_content.append(
                {
                    "type": "text",
                    "text": retrieval_block,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        user_content.append({"type": "text", "text": user_message})

        messages: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
        messages.append({"role": "user", "content": user_content})

        self._audit.debug(
            "llm_request_start",
            model=chosen_model,
            has_retrieval_block=retrieval_block is not None,
            history_len=len(history),
        )

        start = time.monotonic()
        try:
            raw = await self._client.messages.create(
                model=chosen_model,
                max_tokens=chosen_max_tokens,
                temperature=chosen_temperature,
                system=system_blocks,
                messages=messages,
            )
        except RateLimitError as exc:
            status_code = getattr(exc, "status_code", None)
            self._audit.warning("llm_rate_limited", model=chosen_model, status_code=status_code)
            # Use type+status only; SDK exception ``__str__`` can include the
            # response body (potentially echoing prompt fragments) — keep
            # diagnostic chain via ``from exc`` instead.
            raise LLMRateLimitError(f"{type(exc).__name__}: status={status_code}") from exc
        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            error_type = type(exc).__name__
            self._audit.error(
                "llm_api_error",
                model=chosen_model,
                error_type=error_type,
                status_code=status_code,
            )
            # PII discipline: do NOT include ``str(exc)`` in the wrapped
            # message — see RateLimitError branch above.
            raise LLMAPIError(f"{error_type}: status={status_code}") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        response = self._parse_response(raw)  # may emit llm_unexpected_extra_blocks

        self._audit.info(
            "llm_response_success",
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
            cache_read_input_tokens=response.cache_read_input_tokens,
            stop_reason=response.stop_reason,
            duration_ms=duration_ms,
        )

        if (
            response.cache_creation_input_tokens == 0
            and response.cache_read_input_tokens == 0
            and response.input_tokens > 0
        ):
            threshold_hint, model_unknown = _cache_threshold_hint(chosen_model)
            self._audit.warning(
                "llm_cache_not_written",
                model=chosen_model,
                input_tokens=response.input_tokens,
                threshold_hint=threshold_hint,
                model_unknown=model_unknown,
            )

        return response

    def _parse_response(self, raw: Any) -> ClaudeResponse:
        content_blocks = getattr(raw, "content", None)
        if not content_blocks:
            raise LLMResponseError("response has no content blocks")
        first = content_blocks[0]
        text = getattr(first, "text", None)
        if text is None:
            raise LLMResponseError(f"first content block is not text (type={type(first).__name__})")

        # Surface (rather than silently truncate) responses with extra blocks
        # we don't currently handle — e.g., extended-thinking ``thinking_block``
        # at index 0+ or future ``tool_use_block`` at index 1+. v0.2.0 only
        # extracts the first text block; ops should know when this drops data.
        if len(content_blocks) > 1:
            self._audit.warning(
                "llm_unexpected_extra_blocks",
                model=raw.model,
                count=len(content_blocks),
            )

        usage = raw.usage
        return ClaudeResponse(
            content=text,
            model=raw.model,
            stop_reason=raw.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
