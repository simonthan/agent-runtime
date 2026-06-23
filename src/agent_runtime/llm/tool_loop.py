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

from agent_runtime.llm.client import AnthropicClient
from agent_runtime.logging import AuditLogger, NullAuditLogger

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


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Outcome of executing one tool call. `is_error=True` is fed back to the model
    as a tool_result error block (the model may recover); it is NOT an exception."""

    content: str
    is_error: bool = False


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
        history: tuple[dict[str, Any], ...] = (),
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ToolLoopResult:
        """Run the fenced loop. `max_rounds` caps model turns that return
        stop_reason=='tool_use'. Returns once the model stops requesting tools, the
        cap is reached, OR a confirm-required tool suspends the loop.

        CONTRACT (Opus R3 C1): on cap exhaustion `final_text` MAY be empty. The
        consumer MUST route `cap_exhausted is True` to PATH B regardless of
        `final_text`. Likewise it MUST check `pending_confirmation is not None`
        BEFORE PATH A/B classification — a suspended turn is neither A nor B.

        `max_rounds=N` issues up to N+1 SDK calls (N tool rounds + 1 final no-tools
        call). With `confirm=None` (default) behaviour is byte-for-byte unchanged."""
        system_blocks = self._build_system_blocks(static_system_prefix, dynamic_system_suffix)
        first_user: list[dict[str, Any]] = []
        if retrieval_block:
            first_user.append(
                {"type": "text", "text": retrieval_block, "cache_control": {"type": "ephemeral"}}
            )
        first_user.append({"type": "text", "text": user_message})
        messages: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
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
    ) -> ToolLoopResult:
        """Resume a loop suspended by a confirm-required tool. `state` is the opaque
        dict from `PendingConfirmation.state` (may have been JSON round-tripped through
        persistence). `decision` resolves the pending call; the loop then finishes the
        round (may suspend AGAIN on a later confirm-required block in the same round —
        D5) and drives on. Re-supply the same `tools`/`executor`/`confirm`/system args
        as the originating `run()`; only conversation progress lives in `state`.

        TOKENS (split billing, D6): the returned counts are CONTINUATION-ONLY — they do
        NOT include the suspending run()'s tokens. A consumer tracking per-turn spend
        MUST record budget on EVERY ToolLoopResult it receives (the suspend result AND
        each resume result), not once per logical turn, or it will under-report. A
        re-suspend that makes no model call reports zero tokens (correct).

        IDEMPOTENCY: resume CONSUMES `state` (it appends to the live `messages` list).
        Re-resuming the same `state` object is undefined — persist a fresh copy per
        attempt if you need to retry."""
        system_blocks = self._build_system_blocks(static_system_prefix, dynamic_system_suffix)
        messages: list[dict[str, Any]] = state["messages"]
        steps: list[ToolLoopStep] = [self._step_from_dict(s) for s in state["steps"]]
        agg: dict[str, int] = {"in": 0, "out": 0, "cc": 0, "cr": 0}  # D6: continuation-only
        rounds: int = state["rounds"]
        rnd = state["round"]
        tool_uses: list[dict[str, Any]] = rnd["tool_uses"]
        pending_index: int = rnd["pending_index"]
        calls: list[ToolCall] = [self._call_from_dict(c) for c in rnd["calls"]]

        # Resolve the pending block per the user's decision (D1).
        pending = tool_uses[pending_index]
        if isinstance(decision, ExecuteDecision):
            tool_input = (
                decision.tool_input if decision.tool_input is not None else pending["input"]
            )
            outcome = await executor(pending["name"], tool_input)
            calls.append(
                ToolCall(
                    id=pending["id"],
                    name=pending["name"],
                    input=tool_input,
                    result=outcome.content,
                    is_error=outcome.is_error,
                )
            )
        else:  # InjectResultDecision — no executor call (D2)
            calls.append(
                ToolCall(
                    id=pending["id"],
                    name=pending["name"],
                    input=pending["input"],
                    result=decision.content,
                    is_error=decision.is_error,
                )
            )

        # Finish the rest of the round (may suspend again — D5).
        round_outcome = await self._resolve_round(
            tool_uses=tool_uses,
            start_index=pending_index + 1,
            calls=calls,
            executor=executor,
            confirm=confirm,
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

    @staticmethod
    async def _resolve_round(
        *,
        tool_uses: list[dict[str, Any]],
        start_index: int,
        calls: list[ToolCall],
        executor: ToolExecutor,
        confirm: ConfirmPredicate | None,
    ) -> _RoundOutcome:
        """Iterate tool_use blocks from `start_index`, executing non-confirm tools
        (D3). Returns _RoundSuspended at the first confirm-required block, else
        _RoundCompleted once every block has a call. Mutates+returns `calls`."""
        for i in range(start_index, len(tool_uses)):
            tu = tool_uses[i]
            if confirm is not None and confirm(tu["name"], tu["input"]):
                return _RoundSuspended(pending_index=i, calls=calls)
            outcome = await executor(tu["name"], tu["input"])
            calls.append(
                ToolCall(
                    id=tu["id"],
                    name=tu["name"],
                    input=tu["input"],
                    result=outcome.content,
                    is_error=outcome.is_error,
                )
            )
        return _RoundCompleted(calls=calls)

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
                "content": c.result,
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
    def _call_to_dict(c: ToolCall) -> dict[str, Any]:
        return {
            "id": c.id,
            "name": c.name,
            "input": c.input,
            "result": c.result,
            "is_error": c.is_error,
        }

    @staticmethod
    def _call_from_dict(d: dict[str, Any]) -> ToolCall:
        return ToolCall(
            id=d["id"], name=d["name"], input=d["input"], result=d["result"], is_error=d["is_error"]
        )

    @classmethod
    def _step_to_dict(cls, s: ToolLoopStep) -> dict[str, Any]:
        return {
            "assistant_text": s.assistant_text,
            "tool_calls": [cls._call_to_dict(c) for c in s.tool_calls],
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
