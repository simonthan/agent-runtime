"""ToolUseLoop — generic, policy-free fenced model-driven tool-use loop.

Drives Anthropic tool-use over a caller-supplied tool set + executor, bounded by a
caller-supplied round cap. Owns NO policy: no cap value, no result classification,
no user messaging, no MCP knowledge, no notion of a "write tool". The consumer
(teams-bot-platform's agent loop) supplies the cap, the tools, the executor, and an
optional ``confirm`` predicate that flags tool calls requiring human approval before
dispatch. See teams-bot-platform/docs/agentic-consumer-design.md.

Confirm-before-dispatch (T-025a): when ``confirm(name, input)`` returns True the loop
SUSPENDS instead of executing — ``run`` returns a ToolLoopResult whose
``pending_confirmation`` carries the proposed call plus an opaque, JSON-serializable
``state``. The consumer persists ``state`` (it survives an async approval round-trip
across processes), surfaces the proposal, then calls ``resume(state=..., decision=...)``
once the user decides. The loop stays policy-free: it never learns the approval UX,
the persistence, or which tools are writes. With ``confirm=None`` (the default) the loop
behaves byte-for-byte as before — the regression guarantee.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent_runtime.llm.client import AnthropicClient, assemble_history_messages
from agent_runtime.llm.compaction import estimate_tokens
from agent_runtime.llm.models import LLMImage
from agent_runtime.llm.round_context import ToolRoundContext, bind_tool_round
from agent_runtime.logging import AuditLogger, NullAuditLogger
from agent_runtime.safety.prompt_sanitizer import repair_clipped_tool_result

__all__ = [
    "ConfirmPredicate",
    "ExecuteDecision",
    "InjectResultDecision",
    "PendingConfirmation",
    "ResumeDecision",
    "ToolCall",
    "ToolExecutor",
    "ToolLoopResult",
    "ToolLoopStep",
    "ToolResult",
    "ToolUseLoop",
]

# Appended to a tool result that exceeded `max_result_chars`. EXPLICIT by design:
# a knowledge bot handed a silently-clipped corpus will summarise a fragment as if it
# were whole (a correctness bug, not just a cost bug). See TBP T-081.
# Carries the `[platform]` provenance prefix (T-155), exactly like _IMAGES_DROPPED_MARKER:
# nothing else distinguishes a first-party platform notice from tool-supplied data, so a
# hostile server echoing this literal could otherwise fake a platform truncation. Forged
# copies lose the prefix at the sanitizer boundary (safety/prompt_sanitizer.py
# _PLATFORM_PROVENANCE_PATTERN, alternated into BOTH _SENTINEL_RE and _NEUTRALIZE_RE);
# genuine ones are appended AFTER the consumer sanitizes and never pass through it.
_TRUNCATION_MARKER = (
    "\n\n[platform] [TRUNCATED BY agent-runtime: this tool result was {original} characters "
    "(~{est_tokens} tokens) and exceeded the {cap}-character limit; {removed} "
    "characters were removed from the end. This is a PARTIAL result — do not assume "
    "it is complete, and do not treat the omitted content as unimportant.]"
)

# Appended to a tool result whose images were dropped by the per-turn image budget.
# EXPLICIT by design, exactly like _TRUNCATION_MARKER: a model told nothing will answer
# about pages it was never shown. See TBP T-135.
_IMAGES_DROPPED_MARKER = (
    "[platform] [IMAGES WITHHELD BY agent-runtime: this tool result carried {original} "
    "image(s); {dropped} of them were NOT delivered to you ({reasons}). You CANNOT see "
    "the withheld image(s) — do not describe, summarise, or cite their contents, and do "
    "not assume they were unimportant. Disregard any earlier statement in this result "
    "claiming those images are attached — this notice supersedes it. Say so plainly if "
    "answering requires them.]"
)


def _decoded_b64_len(data_b64: str) -> int:
    """Decoded byte length of standard base64, computed WITHOUT decoding.

    `base64.b64decode` on a multi-megabyte image allocates the whole payload just to
    call `len()` on it — once per image, per round. This is O(1) and allocation-free,
    preserving the "exact, zero-cost, deterministic" measurement property the sibling
    char cap relies on (T-081a §2). Exact for both padded and unpadded standard base64.
    """
    n = len(data_b64)
    if n == 0:
        return 0
    pad = 2 if data_b64.endswith("==") else (1 if data_b64.endswith("=") else 0)
    return (n * 3) // 4 - pad


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of executing one tool call. `is_error=True` is fed back to the model
    as a tool_result error block (the model may recover); it is NOT an exception.

    `images` (T-118a) are carried back to the model as image content blocks inside
    the tool_result. Empty (the default) reproduces the pre-T-118a string-content
    block byte-for-byte. The loop applies NO policy to them: it does not cap their
    count or size and does not sanitize them (`max_result_chars` bounds TEXT only,
    see `_cap_result`). The consumer owns those decisions — and owns telling the
    model that image bytes from a tool are untrusted external data.
    """

    content: str
    is_error: bool = False
    images: tuple[LLMImage, ...] = ()


# (tool_name, tool_input) -> ToolResult. Must not raise for expected failures.
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]
# (tool_name, tool_input) -> True if this call must be confirmed before dispatch.
ConfirmPredicate = Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One executed tool call, captured for replay/audit."""

    id: str
    name: str
    input: dict[str, Any]
    result: str
    is_error: bool
    images: tuple[LLMImage, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolLoopStep:
    """One model round: the assistant's text + the tools it called that round."""

    assistant_text: str
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True, slots=True)
class ExecuteDecision:
    """Approve (Send) or approve-with-edit (Edit): the loop runs the executor.
    `tool_input=None` reuses the pending call's original input; a dict replaces it.
    The consumer validates edited input against the tool schema before resuming —
    the loop owns no schema knowledge."""

    tool_input: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InjectResultDecision:
    """Discard / substitute: the loop feeds `content` as the tool_result WITHOUT
    calling the executor, then lets the model react (e.g. acknowledge the decline)."""

    content: str
    is_error: bool = False


ResumeDecision = ExecuteDecision | InjectResultDecision


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    """The loop paused before dispatching a confirm-required tool. `state` is opaque
    and JSON-serializable — persist it and pass it back to `ToolUseLoop.resume()`.
    `resume` must be called with the SAME `tools`/`executor`/`confirm`/system args.

    `state` is JSON-safe provided the `history` passed to `run()` and every tool
    `input`/`ToolResult.content` contain only JSON-native types (str/int/float/
    bool/None/list/dict). Tool inputs originate from the Anthropic SDK (already
    JSON-deserialized) so they are safe; the consumer is responsible for JSON-safe
    history. `state["v"]` is a schema-version tag for future cross-process migration."""

    tool_call_id: str
    tool_name: str
    tool_input: dict[str, Any]
    state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    """Result of a fenced loop. The caller classifies this into PATH A/B — UNLESS
    `pending_confirmation` is set, in which case the loop suspended awaiting human
    approval and the caller must surface it and call `resume()` (check this FIRST)."""

    final_text: str
    stop_reason: str  # last stop_reason, "cap_exhausted", or "pending_confirmation"
    cap_exhausted: bool
    steps: tuple[ToolLoopStep, ...]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    pending_confirmation: PendingConfirmation | None = None


@dataclass(frozen=True, slots=True)
class _RoundCompleted:
    calls: list[ToolCall]


@dataclass(frozen=True, slots=True)
class _RoundSuspended:
    pending_index: int
    calls: list[ToolCall]


_RoundOutcome = _RoundCompleted | _RoundSuspended


def _tool_result_content(call: ToolCall) -> str | list[dict[str, Any]]:
    """The `content` value for one tool_result block.

    No images -> the bare string, byte-for-byte identical to pre-T-118a.
    With images -> [text block?] + image blocks, matching the T-067d user-turn
    order (text first, then images). The text block is OMITTED when `result` is
    falsy: the Anthropic API rejects an empty text block (same rule as
    ToolUseLoop.run's images-with-empty-user_message case).
    """
    if not call.images:
        return call.result
    parts: list[dict[str, Any]] = []
    if call.result:
        parts.append({"type": "text", "text": call.result})
    parts.extend(img.to_block() for img in call.images)
    return parts


class ToolUseLoop:
    def __init__(self, *, client: AnthropicClient, audit_logger: AuditLogger | None = None) -> None:
        self._client = client
        self._audit: AuditLogger = audit_logger or NullAuditLogger()

    async def run(
        self,
        *,
        static_system_prefix: str,
        user_message: str,
        tools: list[dict[str, Any]],
        executor: ToolExecutor,
        max_rounds: int,
        confirm: ConfirmPredicate | None = None,
        dynamic_system_suffix: str | None = None,
        retrieval_block: str | None = None,
        images: tuple[LLMImage, ...] = (),
        history: tuple[dict[str, Any], ...] = (),
        cache_history: bool = False,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_result_chars: int | None = None,
        max_turn_images: int | None = None,
        max_turn_image_bytes: int | None = None,
    ) -> ToolLoopResult:
        """Run the fenced loop. `max_rounds` caps model turns that return
        stop_reason=='tool_use'. Returns once the model stops requesting tools, the
        cap is reached, OR a confirm-required tool suspends the loop.

        CONTRACT (Opus R3 C1): on cap exhaustion `final_text` MAY be empty. The
        consumer MUST route `cap_exhausted is True` to PATH B regardless of
        `final_text`. Likewise it MUST check `pending_confirmation is not None`
        BEFORE PATH A/B classification — a suspended turn is neither A nor B.

        `max_rounds=N` issues up to N+1 SDK calls (N tool rounds + 1 final no-tools
        call). With `confirm=None` (default) behaviour is byte-for-byte unchanged.

        `cache_history=True` marks the last history message with a ``cache_control``
        ephemeral breakpoint so Anthropic caches the stable history prefix across
        turns. `cache_history=False` (default) keeps behaviour byte-for-byte
        unchanged — the regression guarantee for existing callers (T-038a).

        `images` (T-067d) inserts base64 image content blocks between the cached
        retrieval block and the user text (vision passthrough). Default () is
        byte-for-byte unchanged — the regression guarantee for existing callers.
        Image bytes bypass `sanitize_for_llm_prompt` (text-only by design) and,
        if the turn suspends on a confirm-gated tool, ride `state["messages"]`
        into the consumer's suspend store verbatim — cap count/size upstream.
        Consumers persist their OWN history; store a text manifest for image
        turns, not content blocks.

        `max_result_chars` (T-081a) caps each EXECUTOR-produced tool result to that
        many characters, appending an explicit truncation marker on overflow. Default
        `None` = no cap = byte-for-byte unchanged (regression guarantee). The loop owns
        no default; the consumer supplies the ceiling, exactly as it supplies max_rounds.

        `max_turn_images` / `max_turn_image_bytes` (T-135) bound tool-result images for
        the WHOLE turn (not per result): `messages` is re-sent every round and the
        Anthropic API rejects a request carrying more than 100 images, so a per-result
        cap cannot bound a render-heavy turn. Over-budget images are DROPPED (never
        downscaled) and an explicit marker tells the model what it cannot see. Bytes are
        measured DECODED. `0` is meaningful: it drops every image. Default `None` on both
        = no budget = byte-for-byte unchanged (regression guarantee). The loop owns no
        default; the consumer supplies the ceilings, exactly as with max_result_chars."""
        system_blocks = self._build_system_blocks(static_system_prefix, dynamic_system_suffix)
        first_user: list[dict[str, Any]] = []
        if retrieval_block:
            first_user.append(
                {"type": "text", "text": retrieval_block, "cache_control": {"type": "ephemeral"}}
            )
        first_user.extend(img.to_block() for img in images)
        if user_message or not images:
            # Byte-identical to pre-T-067d for images=(); with images present an
            # empty text block is omitted (Anthropic rejects empty text).
            first_user.append({"type": "text", "text": user_message})
        messages: list[dict[str, Any]] = assemble_history_messages(
            history, cache_history=cache_history
        )
        messages.append({"role": "user", "content": first_user})

        return await self._drive(
            system_blocks=system_blocks,
            messages=messages,
            tools=tools,
            executor=executor,
            confirm=confirm,
            max_rounds=max_rounds,
            rounds=0,
            steps=[],
            agg={"in": 0, "out": 0, "cc": 0, "cr": 0},
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_result_chars=max_result_chars,
            images_used={"count": 0, "bytes": 0},
            max_turn_images=max_turn_images,
            max_turn_image_bytes=max_turn_image_bytes,
        )

    async def resume(
        self,
        *,
        state: dict[str, Any],
        decision: ResumeDecision,
        tools: list[dict[str, Any]],
        executor: ToolExecutor,
        confirm: ConfirmPredicate,
        static_system_prefix: str,
        max_rounds: int,
        dynamic_system_suffix: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_result_chars: int | None = None,
        max_turn_images: int | None = None,
        max_turn_image_bytes: int | None = None,
    ) -> ToolLoopResult:
        """Resume a loop suspended by a confirm-required tool. `state` is the opaque
        dict from `PendingConfirmation.state` (may have been JSON round-tripped through
        persistence). `decision` resolves the pending call; the loop then finishes the
        round (may suspend AGAIN on a later confirm-required block in the same round —
        D5) and drives on. Re-supply the same `tools`/`executor`/`confirm`/system args
        as the originating `run()`; only conversation progress lives in `state`.

        `max_result_chars` mirrors `run()` (T-081a) and must be re-supplied by the
        consumer on every call — only conversation progress lives in `state`.

        TOKENS (split billing, D6): the returned counts are CONTINUATION-ONLY — they do
        NOT include the suspending run()'s tokens. A consumer tracking per-turn spend
        MUST record budget on EVERY ToolLoopResult it receives (the suspend result AND
        each resume result), not once per logical turn, or it will under-report. A
        re-suspend that makes no model call reports zero tokens (correct).

        IDEMPOTENCY: resume CONSUMES `state` (it appends to the live `messages` list).
        Re-resuming the same `state` object is undefined — persist a fresh copy per
        attempt if you need to retry.

        `max_turn_images` / `max_turn_image_bytes` mirror `run()` (T-135) and must be
        re-supplied on every call. The consumed budget itself DOES ride `state` — it is a
        bound, not a bill."""
        # cache_history: no param — the run()-time history marker rides state["messages"].
        system_blocks = self._build_system_blocks(static_system_prefix, dynamic_system_suffix)
        messages: list[dict[str, Any]] = state["messages"]
        steps: list[ToolLoopStep] = [self._step_from_dict(s) for s in state["steps"]]
        agg: dict[str, int] = {"in": 0, "out": 0, "cc": 0, "cr": 0}  # D6: continuation-only
        rounds: int = state["rounds"]
        # T-135 — the per-turn image budget SURVIVES the suspend/resume boundary. Unlike
        # `agg` (reset above: D6 continuation-only BILLING), this is a resource BOUND —
        # resetting it would let every resume re-spend the whole allowance. `.get` is
        # load-bearing for DEPLOY SAFETY: a card suspended before this release has no
        # such key and resumes with a fresh budget. state["v"] stays 1.
        images_used: dict[str, int] = dict(state.get("images_used", {"count": 0, "bytes": 0}))
        rnd = state["round"]
        tool_uses: list[dict[str, Any]] = rnd["tool_uses"]
        pending_index: int = rnd["pending_index"]
        calls: list[ToolCall] = [self._call_from_dict(c) for c in rnd["calls"]]
        # T-115j — `state["rounds"]` already counts the suspending round (see _suspend's
        # docstring), so it IS this round's 1-based index; resume must not re-increment.
        round_ctx = ToolRoundContext(round_index=rounds, max_rounds=max_rounds)

        # Resolve the pending block per the user's decision (D1).
        pending = tool_uses[pending_index]
        if isinstance(decision, ExecuteDecision):
            tool_input = (
                decision.tool_input if decision.tool_input is not None else pending["input"]
            )
            with bind_tool_round(round_ctx):
                outcome = await executor(pending["name"], tool_input)
            outcome = self._cap_result(
                tool_name=pending["name"],
                outcome=outcome,
                max_result_chars=max_result_chars,
                images_used=images_used,
                max_turn_images=max_turn_images,
                max_turn_image_bytes=max_turn_image_bytes,
            )
            call_input, call_result, call_is_error = tool_input, outcome.content, outcome.is_error
            call_images = outcome.images
        else:  # InjectResultDecision — no executor call (D2); first-party text, never images
            call_input, call_result, call_is_error = (
                pending["input"],
                decision.content,
                decision.is_error,
            )
            call_images = ()
        calls.append(
            ToolCall(
                id=pending["id"],
                name=pending["name"],
                input=call_input,
                result=call_result,
                is_error=call_is_error,
                images=call_images,
            )
        )

        # Finish the rest of the round (may suspend again — D5).
        round_outcome = await self._resolve_round(
            tool_uses=tool_uses,
            start_index=pending_index + 1,
            calls=calls,
            executor=executor,
            confirm=confirm,
            max_result_chars=max_result_chars,
            images_used=images_used,
            max_turn_images=max_turn_images,
            max_turn_image_bytes=max_turn_image_bytes,
            round_ctx=round_ctx,
        )
        if isinstance(round_outcome, _RoundSuspended):
            return self._suspend(
                assistant_text=rnd["assistant_text"],
                tool_uses=tool_uses,
                outcome=round_outcome,
                messages=messages,
                steps=steps,
                agg=agg,
                rounds=rounds,
                images_used=images_used,
            )
        self._commit_round(
            messages=messages,
            steps=steps,
            assistant_text=rnd["assistant_text"],
            tool_uses=tool_uses,
            calls=round_outcome.calls,
        )
        return await self._drive(
            system_blocks=system_blocks,
            messages=messages,
            tools=tools,
            executor=executor,
            confirm=confirm,
            max_rounds=max_rounds,
            rounds=rounds,
            steps=steps,
            agg=agg,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_result_chars=max_result_chars,
            images_used=images_used,
            max_turn_images=max_turn_images,
            max_turn_image_bytes=max_turn_image_bytes,
        )

    async def _drive(
        self,
        *,
        system_blocks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        executor: ToolExecutor,
        confirm: ConfirmPredicate | None,
        max_rounds: int,
        rounds: int,
        steps: list[ToolLoopStep],
        agg: dict[str, int],
        model: str | None,
        max_tokens: int | None,
        temperature: float | None,
        max_result_chars: int | None,
        images_used: dict[str, int],
        max_turn_images: int | None,
        max_turn_image_bytes: int | None,
    ) -> ToolLoopResult:
        """Shared round engine. `while rounds < max_rounds` (correct at the
        max_rounds=0 boundary — zero tool rounds, straight to the forced answer)."""
        while rounds < max_rounds:
            resp = await self._client.complete_messages(
                system_blocks=system_blocks,
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._accumulate(agg, resp)
            if resp.stop_reason != "tool_use" or not resp.tool_use:
                return self._result(
                    resp.content, resp.stop_reason, cap_exhausted=False, steps=steps, agg=agg
                )
            rounds += 1
            tool_uses = [{"id": tu.id, "name": tu.name, "input": tu.input} for tu in resp.tool_use]
            outcome = await self._resolve_round(
                tool_uses=tool_uses,
                start_index=0,
                calls=[],
                executor=executor,
                confirm=confirm,
                max_result_chars=max_result_chars,
                images_used=images_used,
                max_turn_images=max_turn_images,
                max_turn_image_bytes=max_turn_image_bytes,
                # T-115j — the executor's only view of the round budget. `rounds` was
                # incremented for THIS round on the line above, so it is the 1-based index.
                round_ctx=ToolRoundContext(round_index=rounds, max_rounds=max_rounds),
            )
            if isinstance(outcome, _RoundSuspended):
                return self._suspend(
                    assistant_text=resp.content,
                    tool_uses=tool_uses,
                    outcome=outcome,
                    messages=messages,
                    steps=steps,
                    agg=agg,
                    rounds=rounds,
                    images_used=images_used,
                )
            self._commit_round(
                messages=messages,
                steps=steps,
                assistant_text=resp.content,
                tool_uses=tool_uses,
                calls=outcome.calls,
            )

        # Cap reached (or max_rounds==0). ONE final model call WITHOUT tools so the
        # model must answer from what it has — no dangling tool_use possible.
        self._audit.warning("tool_loop_cap_exhausted", rounds=rounds, max_rounds=max_rounds)
        final = await self._client.complete_messages(
            system_blocks=system_blocks,
            messages=messages,
            tools=None,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._accumulate(agg, final)
        return self._result(
            final.content, "cap_exhausted", cap_exhausted=True, steps=steps, agg=agg
        )

    async def _resolve_round(
        self,
        *,
        tool_uses: list[dict[str, Any]],
        start_index: int,
        calls: list[ToolCall],
        executor: ToolExecutor,
        confirm: ConfirmPredicate | None,
        max_result_chars: int | None,
        images_used: dict[str, int],
        max_turn_images: int | None,
        max_turn_image_bytes: int | None,
        round_ctx: ToolRoundContext,
    ) -> _RoundOutcome:
        """Iterate tool_use blocks from `start_index`, executing non-confirm tools
        (D3). Returns _RoundSuspended at the first confirm-required block, else
        _RoundCompleted once every block has a call. Mutates+returns `calls`.
        Each executor result is size-capped via `_cap_result` before it is frozen
        into a ToolCall (T-081a).

        T-115j: `round_ctx` is bound for the whole body, so `current_tool_round()`
        answers inside the executor AND inside `confirm`. Token-based, so a nested
        ToolUseLoop restores this one on exit. Binding once around the loop rather
        than per call keeps the hot path free of repeated set/reset."""
        with bind_tool_round(round_ctx):
            for i in range(start_index, len(tool_uses)):
                tu = tool_uses[i]
                if confirm is not None and confirm(tu["name"], tu["input"]):
                    return _RoundSuspended(pending_index=i, calls=calls)
                outcome = await executor(tu["name"], tu["input"])
                outcome = self._cap_result(
                    tool_name=tu["name"],
                    outcome=outcome,
                    max_result_chars=max_result_chars,
                    images_used=images_used,
                    max_turn_images=max_turn_images,
                    max_turn_image_bytes=max_turn_image_bytes,
                )
                calls.append(
                    ToolCall(
                        id=tu["id"],
                        name=tu["name"],
                        input=tu["input"],
                        result=outcome.content,
                        is_error=outcome.is_error,
                        images=outcome.images,
                    )
                )
            return _RoundCompleted(calls=calls)

    def _cap_images(
        self,
        *,
        tool_name: str,
        images: tuple[LLMImage, ...],
        used: dict[str, int],
        max_turn_images: int | None,
        max_turn_image_bytes: int | None,
    ) -> tuple[tuple[LLMImage, ...], str | None]:
        """Admit images against the TURN's remaining budget; return (kept, marker|None).

        Both ceilings are PER-TURN TOTALS, not per result (T-135 D2): `messages` is
        re-sent every round and the Anthropic API rejects a request carrying more than
        100 images, so a per-result cap cannot bound a render-heavy turn. `used` is the
        running counter, MUTATED here and carried across rounds AND across a
        suspend/resume boundary.

        Admission is a single forward pass in `images` order (deterministic — keeps the
        multi-round prefix byte-stable, T-135 D5). `0` is a meaningful ceiling meaning
        "no images at all", so every guard is `is None`, never truthiness (D9).
        """
        if max_turn_images is None and max_turn_image_bytes is None:
            return images, None  # regression guarantee: unset = untouched
        if not images:
            return images, None

        kept: list[LLMImage] = []
        by_count = 0
        by_bytes = 0
        for img in images:
            if max_turn_images is not None and used["count"] >= max_turn_images:
                by_count += 1
                continue
            size = _decoded_b64_len(img.data_b64)
            if max_turn_image_bytes is not None and used["bytes"] + size > max_turn_image_bytes:
                by_bytes += 1
                continue
            kept.append(img)
            used["count"] += 1
            used["bytes"] += size

        dropped = by_count + by_bytes
        if dropped == 0:
            return images, None

        reasons: list[str] = []
        if by_count:
            reasons.append(f"{by_count} exceeded the {max_turn_images}-image per-turn budget")
        if by_bytes:
            reasons.append(f"{by_bytes} exceeded the {max_turn_image_bytes}-byte per-turn budget")
        self._audit.warning(
            "tool_loop_result_images_dropped",
            tool_name=tool_name,
            original_images=len(images),
            dropped_images=dropped,
            dropped_by_count=by_count,
            dropped_by_bytes=by_bytes,
            max_turn_images=max_turn_images,
            max_turn_image_bytes=max_turn_image_bytes,
        )
        marker = _IMAGES_DROPPED_MARKER.format(
            original=len(images), dropped=dropped, reasons="; ".join(reasons)
        )
        return tuple(kept), marker

    def _cap_result(
        self,
        *,
        tool_name: str,
        outcome: ToolResult,
        max_result_chars: int | None,
        images_used: dict[str, int],
        max_turn_images: int | None,
        max_turn_image_bytes: int | None,
    ) -> ToolResult:
        """Bound an executor's tool result before it enters the conversation.

        Returns `outcome` unchanged when no cap is set (`None`) or the content fits —
        the byte-for-byte regression guarantee for callers that never pass a cap.
        On overflow, keeps the first `max_result_chars` characters and appends an
        EXPLICIT truncation marker (silent clipping misleads the model). `is_error`
        is preserved: truncation is not a tool failure. Measurement is on characters
        (exact, zero-cost, deterministic — see plan §2); the marker reports an
        estimated token figure via `estimate_tokens` for readability only.

        Only `content` is measured and clipped — `images` pass through untouched. The cap
        bounds prose; counting base64 against it would let one rendered page evict the
        tool's actual text answer. Image budgeting is the consumer's policy (T-118a D4)."""
        # Text cap first (unchanged); then the image budget appends its own marker.
        # Independent by design (T-135 D7) — final length may exceed max_result_chars by
        # the marker lengths, already true of _TRUNCATION_MARKER alone.
        content = outcome.content
        if max_result_chars is not None and len(content) > max_result_chars:
            original = len(content)
            removed = original - max_result_chars
            marker = _TRUNCATION_MARKER.format(
                original=original,
                est_tokens=estimate_tokens(content),
                cap=max_result_chars,
                removed=removed,
            )
            self._audit.warning(
                "tool_loop_result_truncated",
                tool_name=tool_name,
                original_chars=original,
                cap_chars=max_result_chars,
                removed_chars=removed,
            )
            # T-162: the clip is the THIRD site that can manufacture a sentinel out of
            # already-neutralized text, and the only one that can sever the tool_output
            # envelope. Repair the head BEFORE the `[platform]`-framed marker is appended —
            # that marker's provenance is POSITIONAL (it must sit outside a CLOSED
            # envelope; T-155/T-164), so appending it to a severed envelope buries a
            # first-party notice inside untrusted data. `removed` above deliberately stays
            # `original - max_result_chars`: it describes the CLIP, and the re-close ADDS
            # characters (recomputing it from len(head) can go negative on a small cap).
            content = repair_clipped_tool_result(content[:max_result_chars]) + marker

        kept_images, img_marker = self._cap_images(
            tool_name=tool_name,
            images=outcome.images,
            used=images_used,
            max_turn_images=max_turn_images,
            max_turn_image_bytes=max_turn_image_bytes,
        )
        if img_marker is not None:
            # Non-empty content is an INVARIANT when images were dropped: with no images
            # left, _tool_result_content returns the bare string, and an empty parts list
            # would be rejected by the API (T-135 D4).
            content = f"{content}\n\n{img_marker}" if content else img_marker

        if content is outcome.content and kept_images is outcome.images:
            return outcome  # nothing changed — byte-for-byte identity
        return ToolResult(content=content, is_error=outcome.is_error, images=kept_images)

    @staticmethod
    def _commit_round(
        *,
        messages: list[dict[str, Any]],
        steps: list[ToolLoopStep],
        assistant_text: str,
        tool_uses: list[dict[str, Any]],
        calls: list[ToolCall],
    ) -> None:
        """Append the completed round's assistant turn (text + every tool_use block)
        and the user turn (a tool_result for every call, derived from `calls`), and
        record the step. Anthropic requires a tool_result for every tool_use; `calls`
        is complete + in tool_use order here."""
        assistant_blocks: list[dict[str, Any]] = []
        if assistant_text:
            assistant_blocks.append({"type": "text", "text": assistant_text})
        assistant_blocks.extend(
            {"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]}
            for tu in tool_uses
        )
        tool_result_blocks = [
            {
                "type": "tool_result",
                "tool_use_id": c.id,
                "content": _tool_result_content(c),
                "is_error": c.is_error,
            }
            for c in calls
        ]
        steps.append(ToolLoopStep(assistant_text=assistant_text, tool_calls=tuple(calls)))
        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": tool_result_blocks})

    def _suspend(
        self,
        *,
        assistant_text: str,
        tool_uses: list[dict[str, Any]],
        outcome: _RoundSuspended,
        messages: list[dict[str, Any]],
        steps: list[ToolLoopStep],
        agg: dict[str, int],
        rounds: int,
        images_used: dict[str, int],
    ) -> ToolLoopResult:
        """Build the suspended ToolLoopResult. `state` is JSON-serializable: messages
        (already plain dicts), steps + the in-flight round's calls serialized to dicts,
        token aggregates, and the rounds consumed so far (the suspending round counted —
        on resume the round is committed without re-incrementing)."""
        pending = tool_uses[outcome.pending_index]
        state: dict[str, Any] = {
            "v": 1,  # schema-version tag for future cross-process migration
            "messages": messages,
            "steps": [self._step_to_dict(s) for s in steps],
            "agg": dict(agg),
            "rounds": rounds,
            # T-135 — per-turn image budget consumed so far. Read back with .get() so
            # pre-release suspended states still load. state["v"] intentionally stays 1.
            "images_used": dict(images_used),
            "round": {
                "assistant_text": assistant_text,
                "tool_uses": tool_uses,
                "calls": [self._call_to_dict(c) for c in outcome.calls],
                "pending_index": outcome.pending_index,
            },
        }
        pending_confirmation = PendingConfirmation(
            tool_call_id=pending["id"],
            tool_name=pending["name"],
            tool_input=pending["input"],
            state=state,
        )
        return ToolLoopResult(
            final_text="",
            stop_reason="pending_confirmation",
            cap_exhausted=False,
            steps=tuple(steps),
            input_tokens=agg["in"],
            output_tokens=agg["out"],
            cache_creation_input_tokens=agg["cc"],
            cache_read_input_tokens=agg["cr"],
            pending_confirmation=pending_confirmation,
        )

    @staticmethod
    def _build_system_blocks(
        static_system_prefix: str, dynamic_system_suffix: str | None
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": static_system_prefix, "cache_control": {"type": "ephemeral"}}
        ]
        if dynamic_system_suffix:
            blocks.append({"type": "text", "text": dynamic_system_suffix})
        return blocks

    @staticmethod
    def _accumulate(agg: dict[str, int], resp: Any) -> None:
        agg["in"] += resp.input_tokens
        agg["out"] += resp.output_tokens
        agg["cc"] += resp.cache_creation_input_tokens
        agg["cr"] += resp.cache_read_input_tokens

    @staticmethod
    def _call_to_dict(c: ToolCall, *, include_images: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": c.id,
            "name": c.name,
            "input": c.input,
            "result": c.result,
            "is_error": c.is_error,
        }
        # T-118a — key emitted ONLY when images exist, so suspend states for image-free
        # turns (every turn today) serialize byte-for-byte as before.
        if include_images and c.images:
            d["images"] = [{"media_type": i.media_type, "data_b64": i.data_b64} for i in c.images]
        return d

    @staticmethod
    def _call_from_dict(d: dict[str, Any]) -> ToolCall:
        # `.get` is load-bearing for DEPLOY SAFETY: a turn suspended on an approval card
        # BEFORE this release resumes AFTER it with no "images" key in its stored state.
        return ToolCall(
            id=d["id"],
            name=d["name"],
            input=d["input"],
            result=d["result"],
            is_error=d["is_error"],
            images=tuple(
                LLMImage(media_type=i["media_type"], data_b64=i["data_b64"])
                for i in d.get("images", ())
            ),
        )

    @classmethod
    def _step_to_dict(cls, s: ToolLoopStep) -> dict[str, Any]:
        return {
            "assistant_text": s.assistant_text,
            # include_images=False — a committed round's images already live in
            # state["messages"] as tool_result image blocks; a second copy here would
            # double the suspend-state size and is read by nobody (D6-bis).
            "tool_calls": [cls._call_to_dict(c, include_images=False) for c in s.tool_calls],
        }

    @classmethod
    def _step_from_dict(cls, d: dict[str, Any]) -> ToolLoopStep:
        return ToolLoopStep(
            assistant_text=d["assistant_text"],
            tool_calls=tuple(cls._call_from_dict(c) for c in d["tool_calls"]),
        )

    @staticmethod
    def _result(
        text: str,
        stop_reason: str,
        *,
        cap_exhausted: bool,
        steps: list[ToolLoopStep],
        agg: dict[str, int],
    ) -> ToolLoopResult:
        return ToolLoopResult(
            final_text=text,
            stop_reason=stop_reason,
            cap_exhausted=cap_exhausted,
            steps=tuple(steps),
            input_tokens=agg["in"],
            output_tokens=agg["out"],
            cache_creation_input_tokens=agg["cc"],
            cache_read_input_tokens=agg["cr"],
        )
