"""``AnthropicClient`` — opinionated wrapper around the Anthropic SDK.

Encodes ARCHITECTURE.md §4 decision #5 (reopened 2026-06-27): every request carries a
``cache_control`` ephemeral breakpoint on the static system prefix and one on the
per-turn retrieval block, plus an OPTIONAL third on the conversation-history prefix
(``cache_history=True``). Prompt caching is GA on the Anthropic API (since SDK 0.39) —
no ``anthropic-beta`` header is sent.

Every breakpoint is a 5-minute entry unless the caller asks for a 1-hour TTL on the
static-prefix breakpoint (``prefix_cache_ttl="1h"``, v0.45.0, TBP T-7683; ARCHITECTURE.md
§4 #5 re-opened narrowly 2026-10-21). Only breakpoint 1 can be 1-hour: the API requires a
longer-TTL entry to come before any shorter one, and breakpoint 1 (on ``system``) is the
first marker in the request (tools carry none). The caller must not mark a ``tools`` entry
with a 5-minute ``cache_control`` while asking for a 1-hour prefix.

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
at the call site; the wrapper intentionally does NOT sanitize so the layer of
defense is auditable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal, Protocol

import httpx
from anthropic import APIError, APIStatusError, RateLimitError

from agent_runtime.llm.errors import (
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseError,
)
from agent_runtime.llm.models import ClaudeResponse, History, LLMImage, ToolUseBlock
from agent_runtime.llm.streaming import (
    PromptUsage,
    StreamListener,
    TextDelta,
    ToolUseStarted,
    UsageListener,
    guard_listener,
)
from agent_runtime.logging import AuditLogger, NullAuditLogger

__all__ = ["AnthropicClient", "assemble_history_messages"]

logger = logging.getLogger(__name__)


_CACHE_MIN_TOKENS: dict[str, int] = {
    # Trailing hyphen disambiguates future variants — ``claude-sonnet`` would
    # match e.g. a hypothetical ``claude-sonnet-small`` with a different threshold;
    # ``claude-sonnet-`` only matches versioned ids like ``claude-sonnet-4-6``.
    "claude-sonnet-": 1024,
    "claude-opus-": 4096,
    "claude-haiku-": 2048,
}
_DEFAULT_CACHE_HINT = 1024

# v0.45.0 (TBP T-7683) -- the TTLs a caller may ask for on breakpoint 1. None (the default)
# sends the bare 5-minute marker, byte-for-byte the pre-v0.45.0 request.
_PREFIX_CACHE_TTLS = frozenset({"1h"})

# T-316b -- HTTP statuses on which a STREAMING request is retried once without streaming.
# If that retry succeeds, the endpoint takes the request but not the stream (a proxy or a
# passthrough without SSE), so the client stops streaming for its lifetime
# (`llm_stream_unsupported`). Never 429 / 5xx / timeouts: those are not about streaming.
_STREAM_FALLBACK_STATUSES = frozenset({400, 404, 405, 406, 415, 501})


def _prefix_cache_control(ttl: str | None) -> dict[str, str]:
    """Breakpoint 1's ``cache_control`` marker (v0.45.0, TBP T-7683).

    ``None`` -> ``{"type": "ephemeral"}`` (5 minutes; byte-for-byte the marker every earlier
    version sent). ``"1h"`` -> ``{"type": "ephemeral", "ttl": "1h"}``. Anything else raises
    ``ValueError`` before any model call.
    """
    if ttl is None:
        return {"type": "ephemeral"}
    if ttl not in _PREFIX_CACHE_TTLS:
        raise ValueError(f"prefix_cache_ttl must be None or '1h', got {ttl!r}")
    return {"type": "ephemeral", "ttl": ttl}


def _creation_split(usage: Any) -> tuple[int, int]:
    """``(1-hour, 5-minute)`` cache-write tokens from ``usage.cache_creation`` (v0.45.0).

    The endpoint breaks ``cache_creation_input_tokens`` down by TTL in
    ``usage.cache_creation.ephemeral_1h_input_tokens`` / ``ephemeral_5m_input_tokens``. An
    endpoint (or SDK model) without the breakdown yields ``(0, 0)``: the consumer can tell
    a missing split from a real one because the two then do not add up to
    ``cache_creation_input_tokens``. A plain dict (an SDK too old to type the field) is
    read the same way; a value that is not an int reads as 0.
    """
    split = getattr(usage, "cache_creation", None)
    if split is None:
        return 0, 0
    if isinstance(split, dict):
        one_hour = split.get("ephemeral_1h_input_tokens", 0)
        five_min = split.get("ephemeral_5m_input_tokens", 0)
    else:
        one_hour = getattr(split, "ephemeral_1h_input_tokens", 0)
        five_min = getattr(split, "ephemeral_5m_input_tokens", 0)
    # a non-int (None, a test double's Mock) reads as 0 -- never fails the parse
    return (
        one_hour if isinstance(one_hour, int) else 0,
        five_min if isinstance(five_min, int) else 0,
    )


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


def assemble_history_messages(
    history: History | tuple[Any, ...], *, cache_history: bool = False
) -> list[dict[str, Any]]:
    """Project conversation ``history`` into SDK message dicts.

    With ``cache_history=True`` AND non-empty history, a single ``cache_control``
    ephemeral breakpoint is placed on the LAST block of the LAST history message — the
    stable conversation prefix. Anthropic then incrementally caches the growing history
    across turns (turn N's prefix = turn N-1's cached prefix + one new exchange).

    With ``cache_history=False`` (the default) the output is byte-identical to the prior
    ``[{"role": m["role"], "content": m["content"]} for m in history]`` projection — the
    regression guarantee for existing callers (ithelpdesk). See ARCHITECTURE.md §4 #5
    (reopened 2026-06-27 to permit a third, opt-in breakpoint).
    """
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    if cache_history and messages:
        messages[-1] = {
            "role": messages[-1]["role"],
            "content": _mark_cache_control(messages[-1]["content"]),
        }
    return messages


def _mark_cache_control(content: Any) -> Any:
    """Return ``content`` with a ``cache_control`` ephemeral marker on its last block.

    ``str`` content becomes a one-element text block list; a block list is copied (the
    caller's list is never mutated) with the marker merged onto the trailing dict block.
    Empty lists or a non-dict trailing block are returned unchanged — nothing to mark.
    """
    cc = {"type": "ephemeral"}
    if isinstance(content, str):
        return [{"type": "text", "text": content, "cache_control": cc}]
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        return [*content[:-1], {**content[-1], "cache_control": cc}]
    return content


async def _forward_event(event: Any, on_stream: StreamListener) -> bool:
    """T-316b -- one raw SDK stream event as a listener event: a text delta, or the start of
    a tool_use block. Every other event (message/usage/stop, thinking, json deltas) is not
    reported. True = something was handed to the listener (it may be on a screen now)."""
    etype = getattr(event, "type", None)
    if etype == "content_block_delta":
        delta = getattr(event, "delta", None)
        if getattr(delta, "type", None) == "text_delta":
            text = getattr(delta, "text", "") or ""
            if text:
                await on_stream(TextDelta(text=text))
                return True
    elif etype == "content_block_start":
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) == "tool_use":
            await on_stream(ToolUseStarted(name=getattr(block, "name", "") or ""))
            return True
    return False


def _report_prompt_usage(event: Any, on_usage: UsageListener, audit: AuditLogger) -> None:
    """v0.44.0 (TBP T-7624) -- hand a `message_start` event's prompt usage to `on_usage`. A
    missing usage reports nothing; a raising callback is logged (`llm_usage_listener_failed`,
    warning; audit and stdlib) and never fails the call. Not a stream event: the client's
    one retry after the stream started still counts only what the stream listener was given."""
    usage = getattr(getattr(event, "message", None), "usage", None)
    if usage is None:
        return
    one_hour, five_min = _creation_split(usage)
    reported = PromptUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_1h_input_tokens=one_hour,
        cache_creation_5m_input_tokens=five_min,
    )
    try:
        on_usage(reported)
    except Exception as exc:  # noqa: BLE001 -- a consumer bug must never fail the call
        audit.warning("llm_usage_listener_failed", error_type=type(exc).__name__)
        logger.warning("llm_usage_listener_failed error_type=%s", type(exc).__name__)


class _IncompleteStreamError(Exception):
    """T-316b -- the stream ended (cleanly, from the transport's view) before `message_stop`:
    a proxy cut it. `messages.create` can never return such a body, so neither may we."""


def _is_event_stream(stream: Any) -> bool:
    """T-316b -- whether the streamed response really is SSE (the SDK's `AsyncMessageStream`
    exposes its `httpx.Response`). Unknown counts as not SSE (the fallback)."""
    response = getattr(stream, "response", None)
    headers = getattr(response, "headers", None)
    content_type = headers.get("content-type", "") if headers is not None else ""
    return "text/event-stream" in str(content_type).lower()


class _NoStreamError(Exception):
    """T-316b -- this endpoint does not stream: refused with `status` before any event, or
    answered with no stream event at all (status None)."""

    def __init__(self, status: int | None) -> None:
        super().__init__(f"no stream (status={status})")
        self.status = status


def _stream_error_type(exc: APIError) -> str | None:
    """The `error.type` of an error the API sent INSIDE a stream (`event: error`), e.g.
    "overloaded_error" / "rate_limit_error". Such errors arrive on a 200 response, so their
    HTTP status no longer says what they are."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    etype = error.get("type") if isinstance(error, dict) else None
    return etype if isinstance(etype, str) else None


class _AnthropicMessages(Protocol):
    """Structural Protocol for the SDK's ``client.messages`` attribute.

    ``stream`` (T-316b) is used only when the client streams (``stream_transport=True`` or
    an ``on_stream`` listener); it returns the SDK's async context manager whose stream
    yields raw events and ends with ``get_final_message()``."""

    async def create(self, **kwargs: Any) -> Any: ...

    def stream(self, **kwargs: Any) -> Any: ...


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
        default_temperature: float | None = 0.0,
        audit_logger: AuditLogger | None = None,
        stream_transport: bool = False,
    ) -> None:
        """`stream_transport` (v0.43.0, T-316b): send EVERY call as a streaming request and
        assemble the response from the stream, so a long output never waits on one HTTP
        read (the SDK's read timeout then bounds the gap between events, not the whole
        response). The returned `ClaudeResponse` is the same as a non-streaming call's.
        Default False = byte-for-byte unchanged (`messages.create`). A call that passes an
        `on_stream` listener streams either way."""
        self._client = client
        self._default_model = default_model
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature
        self._audit: AuditLogger = audit_logger or NullAuditLogger()
        self._stream_transport = stream_transport
        # T-316b -- latched True when a streaming request failed and the same request then
        # succeeded without streaming: this endpoint cannot stream, stop trying.
        self._stream_unsupported = False

    async def complete_messages(
        self,
        *,
        system_blocks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        tool_choice: dict[str, Any] | None = None,
        on_stream: StreamListener | None = None,
        on_usage: UsageListener | None = None,
    ) -> ClaudeResponse:
        """Low-level single SDK round over a caller-assembled message list.

        `system_blocks` already carry their own cache_control. `messages` is the
        full list (history + tool rounds) with raw content blocks. `tools`, when
        present, is passed verbatim to the SDK. Does error-mapping + parse +
        cache-not-written hint. `ToolUseLoop` drives this; `complete()` wraps it.

        `tool_choice` (v0.39.0, TBP T-7254a) is passed verbatim to the SDK, and only
        together with a non-empty `tools` (with no tools there is nothing to choose;
        the forced-final call passes none). None omits the param: byte-for-byte
        unchanged. A changed tool_choice keeps the tools + system cache and misses the
        messages cache for that call; the next call with a different choice cannot read
        the messages cache this one wrote.
        Some models reject a forced choice (`{"type": "tool"}` / `{"type": "any"}`)
        with a 400, which surfaces as `LLMAPIError` -- the consumer gates by model.

        `on_stream` (v0.43.0, T-316b) receives `TextDelta` / `ToolUseStarted` events while the
        response is generated (see `agent_runtime.llm.streaming` for the contract). Passing
        it makes this call a streaming request; the return value and error mapping are the
        same as without it. None (default) with `stream_transport=False` = byte-for-byte
        unchanged.

        `on_usage` (v0.44.0, TBP T-7624) is called synchronously with the `PromptUsage` the
        endpoint reports when a STREAMED response starts (once per attempt), so a consumer
        can account for a call that is cancelled or fails before it returns. It never makes
        a call stream, never changes the request, and a raising callback never fails the
        call. A call that is not streamed never calls it.
        """
        chosen_model = model or self._default_model
        chosen_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens
        chosen_temperature = temperature if temperature is not None else self._default_temperature

        create_kwargs: dict[str, Any] = {
            "model": chosen_model,
            "max_tokens": chosen_max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        if chosen_temperature is not None:
            # T-190: None ⇒ omit — newer model families hard-400 on an explicit
            # temperature; omission defers to the API default. Default 0.0 keeps
            # every existing caller byte-identical.
            create_kwargs["temperature"] = chosen_temperature
        if tools:
            create_kwargs["tools"] = tools
            if tool_choice is not None:
                create_kwargs["tool_choice"] = tool_choice

        # Request-start audit lives HERE (not in complete()) so every loop round
        # emits a paired start/response event — complete() must NOT also log it,
        # or single-shot calls double-log (Sonnet F4).
        self._audit.debug(
            "llm_request_start", model=chosen_model, has_tools=bool(tools), n_messages=len(messages)
        )
        start = time.monotonic()
        streaming = (self._stream_transport or on_stream is not None) and (
            not self._stream_unsupported
        )
        try:
            if streaming:
                raw = await self._create_streamed(
                    create_kwargs,
                    guard_listener(on_stream, self._audit),
                    chosen_model,
                    on_usage=on_usage,
                )
            else:
                raw = await self._client.messages.create(**create_kwargs)
        except RateLimitError as exc:
            status_code = getattr(exc, "status_code", None)
            self._audit.warning("llm_rate_limited", model=chosen_model, status_code=status_code)
            raise LLMRateLimitError(f"{type(exc).__name__}: status={status_code}") from exc
        except APIError as exc:
            status_code = getattr(exc, "status_code", None)
            if _stream_error_type(exc) == "rate_limit_error":  # T-316b: an in-stream 429
                self._audit.warning("llm_rate_limited", model=chosen_model, status_code=429)
                raise LLMRateLimitError(f"{type(exc).__name__}: status=429") from exc
            error_type = type(exc).__name__
            self._audit.error(
                "llm_api_error", model=chosen_model, error_type=error_type, status_code=status_code
            )
            raise LLMAPIError(f"{error_type}: status={status_code}") from exc
        except (httpx.TransportError, httpx.StreamError, _IncompleteStreamError) as exc:
            # T-316b -- a streamed body is read AFTER the SDK's send(), so a connection that
            # drops or stalls mid-stream reaches us as a raw httpx error (the SDK maps only
            # errors inside send()); and a stream cut before message_stop is no answer at all.
            # Both map exactly like the API errors above.
            error_type = (
                "IncompleteStream"
                if isinstance(exc, _IncompleteStreamError)
                else type(exc).__name__
            )
            self._audit.error(
                "llm_api_error", model=chosen_model, error_type=error_type, status_code=None
            )
            raise LLMAPIError(f"{error_type}: status=None") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        response = self._parse_response(raw)
        self._audit.info(
            "llm_response_success",
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
            cache_read_input_tokens=response.cache_read_input_tokens,
            cache_creation_1h_input_tokens=response.cache_creation_1h_input_tokens,
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

    async def _create_streamed(
        self,
        create_kwargs: dict[str, Any],
        on_stream: StreamListener | None,
        model: str,
        *,
        on_usage: UsageListener | None = None,
    ) -> Any:
        """T-316b -- one streaming request, assembled into the SDK's final Message (the same
        object `messages.create` returns, so `_parse_response` is shared). Forwards text
        deltas and tool_use starts to `on_stream` (already guarded).

        - This endpoint does not stream (a `_STREAM_FALLBACK_STATUSES` error before any event,
          or a 2xx with no stream event: a proxy that ignored `stream: true`): ONE retry
          without streaming; a successful retry latches streaming off for this client.
        - A failure AFTER the stream started (a dropped connection, a stall past the read
          timeout, an error event such as overloaded_error) used to be retried by the SDK
          inside send(); the streamed body is read after send(), so the SDK no longer does.
          It is retried here ONCE -- only when the client allows retries (max_retries > 0)
          and nothing has been handed to the listener yet (text on a screen is never
          re-generated). Otherwise it propagates to `complete_messages`' mapping.
        - A stream that ends before `message_stop` raises (mapped to LLMAPIError): a cut
          stream is not an answer."""
        retry_left = (getattr(self._client, "max_retries", 0) or 0) > 0
        while True:
            progress = {"started": False, "forwarded": False}
            try:
                return await self._stream_once(create_kwargs, on_stream, progress, on_usage)
            except _NoStreamError as exc:
                raw = await self._client.messages.create(**create_kwargs)
                self._stream_unsupported = True
                self._audit.error("llm_stream_unsupported", model=model, status_code=exc.status)
                # The stdlib line too: a consumer that wires no audit logger still sees it.
                logger.error(  # noqa: TRY400 -- an expected degrade, not a crash: no traceback
                    "llm_stream_unsupported model=%s status_code=%s", model, exc.status
                )
                return raw
            except (APIStatusError, httpx.TransportError, httpx.StreamError) as exc:
                # before the stream opened the SDK's own retry loop already ran (the error
                # came out of send()); after a forwarded event the text may be on a screen
                if not retry_left or not progress["started"] or progress["forwarded"]:
                    raise
                retry_left = False
                self._audit.warning("llm_stream_retry", model=model, error_type=type(exc).__name__)

    async def _stream_once(
        self,
        create_kwargs: dict[str, Any],
        on_stream: StreamListener | None,
        progress: dict[str, bool],
        on_usage: UsageListener | None = None,
    ) -> Any:
        """One streaming attempt (see `_create_streamed`). `progress["started"]` turns True once
        the stream is open (headers received: anything after this is read from the body, which
        the SDK's retry never covered), `progress["forwarded"]` once the listener has been
        handed anything.

        No event at all: a body that is not an event stream (a proxy answered plain JSON)
        means this endpoint cannot stream (`_NoStreamError`, fallback + latch); an event
        stream with no event (cut before its first event, or pings only) is a transient cut
        (`_IncompleteStreamError`), never a latch."""
        seen_event = False
        saw_stop = False
        try:
            async with self._client.messages.stream(**create_kwargs) as stream:
                progress["started"] = True
                async for event in stream:
                    seen_event = True
                    etype = getattr(event, "type", None)
                    if etype == "message_stop":
                        saw_stop = True
                    elif etype == "message_start" and on_usage is not None:
                        _report_prompt_usage(event, on_usage, self._audit)  # T-7624
                    if on_stream is not None and await _forward_event(event, on_stream):
                        progress["forwarded"] = True
                if not seen_event:
                    if _is_event_stream(stream):
                        raise _IncompleteStreamError
                    raise _NoStreamError(None)
                if not saw_stop:
                    raise _IncompleteStreamError
                return await stream.get_final_message()
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if not seen_event and status in _STREAM_FALLBACK_STATUSES:
                raise _NoStreamError(status) from exc
            raise

    async def complete(
        self,
        *,
        static_system_prefix: str,
        user_message: str,
        dynamic_system_suffix: str | None = None,
        history: History = (),
        retrieval_block: str | None = None,
        images: tuple[LLMImage, ...] = (),
        tools: list[dict[str, Any]] | None = None,
        cache_history: bool = False,
        cache_retrieval: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        on_stream: StreamListener | None = None,
        prefix_cache_ttl: Literal["1h"] | None = None,
    ) -> ClaudeResponse:
        """Send a completion request with ``cache_control`` ephemeral breakpoints.

        Breakpoint #1: ``static_system_prefix`` (always cached).
        Breakpoint #2: ``retrieval_block`` (cached when present and
        ``cache_retrieval=True``; prepended to user_message). Pass
        ``cache_retrieval=False`` to skip the BP2 write for single-shot callers
        whose retrieval block is never read back (T-190).
        Breakpoint #3 (opt-in): the last history message, marked when
        ``cache_history=True`` and ``history`` is non-empty — caches the stable
        conversation prefix so multi-turn sessions read it from cache (T-038a).

        ``dynamic_system_suffix`` is passed through uncached.

        ``images`` (T-067d) inserts base64 image content blocks between the
        cached retrieval block and the user text (vision passthrough). They sit
        AFTER the breakpoint-#2 marker, so per-turn images never join (or
        invalidate) the cached retrieval prefix. Default ``()`` is byte-for-byte
        unchanged — the regression guarantee for existing callers.

        Delegates to ``complete_messages`` after assembling blocks. ``tools`` is
        passed verbatim; ``None`` omits the param entirely (D5 — byte-identical
        to v0.5.0 behavior when no tools supplied).

        NOTE: llm_request_start is emitted inside complete_messages (Sonnet F4) —
        do NOT re-add it here or single-shot calls double-log.

        ``on_stream`` (v0.43.0, T-316b) is forwarded to ``complete_messages`` -- see there.

        ``prefix_cache_ttl`` (v0.45.0, TBP T-7683): ``"1h"`` makes breakpoint #1 a 1-hour
        entry (``{"type": "ephemeral", "ttl": "1h"}``; written at 2x base input instead of
        1.25x, see ``ClaudeResponse.cache_creation_1h_input_tokens``). Every other marker
        stays 5 minutes. ``None`` (default) is byte-for-byte unchanged; any other value
        raises ``ValueError`` before the call.
        """
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": static_system_prefix,
                "cache_control": _prefix_cache_control(prefix_cache_ttl),
            }
        ]
        if dynamic_system_suffix:
            # Truthy check (not ``is not None``) so empty string is skipped —
            # Anthropic API rejects ``{"type":"text","text":""}`` with 400.
            system_blocks.append({"type": "text", "text": dynamic_system_suffix})

        user_content: list[dict[str, Any]] = []
        if retrieval_block:
            # Truthy check — see ``dynamic_system_suffix`` comment above.
            block: dict[str, Any] = {"type": "text", "text": retrieval_block}
            if cache_retrieval:
                # T-190: opt-out for single-shot callers whose BP2 write is never
                # read back (no rounds follow; cross-turn needs a byte-identical
                # block). Default True = unchanged.
                block["cache_control"] = {"type": "ephemeral"}
            user_content.append(block)
        user_content.extend(img.to_block() for img in images)
        if user_message or not images:
            user_content.append({"type": "text", "text": user_message})

        messages: list[dict[str, Any]] = assemble_history_messages(
            history, cache_history=cache_history
        )
        messages.append({"role": "user", "content": user_content})

        stream_kwargs: dict[str, Any] = {}
        if on_stream is not None:  # only when set: test doubles may lack the keyword
            stream_kwargs["on_stream"] = on_stream
        return await self.complete_messages(
            system_blocks=system_blocks,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            **stream_kwargs,
        )

    def _parse_response(self, raw: Any) -> ClaudeResponse:
        content_blocks = getattr(raw, "content", None)
        if not content_blocks:
            raise LLMResponseError("response has no content blocks")

        text_parts: list[str] = []
        tool_use: list[ToolUseBlock] = []
        unknown = 0
        for block in content_blocks:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_id = getattr(block, "id", None)
                if not tool_id:
                    # Empty/missing id would later submit tool_use_id:"" → Anthropic
                    # 400 far from the cause (Gemini R2 F2). Drop + count as unknown.
                    unknown += 1
                    continue
                tool_use.append(
                    ToolUseBlock(
                        id=tool_id,
                        name=getattr(block, "name", ""),
                        input=getattr(block, "input", {}) or {},
                    )
                )
            else:
                unknown += 1

        if not text_parts and not tool_use:
            # An image/thinking-only first block with no usable content — same
            # failure class as before (previously: "first block is not text").
            first_type = getattr(content_blocks[0], "type", type(content_blocks[0]).__name__)
            raise LLMResponseError(f"response has no text or tool_use blocks (first={first_type})")
        if unknown:
            # `count` = number of unknown-typed blocks (changed from v0.5.0 where
            # count was total extra blocks regardless of type). Downstream alerting
            # keyed on `count` should note this semantic shift (Sonnet F2).
            self._audit.warning("llm_unexpected_extra_blocks", model=raw.model, count=unknown)

        usage = raw.usage
        one_hour, five_min = _creation_split(usage)
        return ClaudeResponse(
            content="".join(text_parts),
            model=raw.model,
            stop_reason=raw.stop_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            tool_use=tuple(tool_use),
            cache_creation_1h_input_tokens=one_hour,
            cache_creation_5m_input_tokens=five_min,
        )
