"""ToolUseLoop — generic, policy-free fenced model-driven tool-use loop.

Drives Anthropic tool-use over a caller-supplied tool set + executor, bounded by a
caller-supplied round cap. Owns NO policy: no cap value, no result classification,
no user messaging, no MCP knowledge. The consumer (teams-bot-platform's agent loop)
supplies the cap, the tools, and the executor, and classifies the returned result
into its own deterministic dispatch. See teams-bot-platform/docs/agentic-consumer-design.md.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent_runtime.llm.client import AnthropicClient
from agent_runtime.logging import AuditLogger, NullAuditLogger

__all__ = [
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
class ToolLoopResult:
    """Result of a fenced loop. The caller classifies this into PATH A/B."""

    final_text: str
    stop_reason: str          # the model's last stop_reason, or "cap_exhausted"
    cap_exhausted: bool
    steps: tuple[ToolLoopStep, ...]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


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
        dynamic_system_suffix: str | None = None,
        retrieval_block: str | None = None,
        history: tuple[dict[str, Any], ...] = (),
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ToolLoopResult:
        """Run the fenced loop. `max_rounds` caps model turns that return
        stop_reason=='tool_use'. Returns once the model stops requesting tools OR
        the cap is reached. Never sends a user message; never classifies the result.

        CONTRACT (Opus R3 C1): on cap exhaustion `final_text` MAY be empty (the model
        wanted a tool it can't call). The consumer (T-011d-c) MUST route
        `cap_exhausted is True` to PATH B regardless of `final_text` — never render an
        empty PATH-A answer. Mirrors design-record §2 step 3.

        `max_rounds=N` issues up to N+1 SDK calls (N tool rounds + 1 final no-tools
        call) — size test fixtures accordingly (Opus R3 H3)."""
        system_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": static_system_prefix, "cache_control": {"type": "ephemeral"}}
        ]
        if dynamic_system_suffix:
            # Uncached 2nd system block — mirrors complete() so the single-shot and
            # loop paths assemble system identically (cache-prefix parity). TBP folds
            # everything into static_system_prefix today, so no live divergence; the
            # param exists so the two paths can't drift later (Sonnet F5 / Opus M1).
            system_blocks.append({"type": "text", "text": dynamic_system_suffix})
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

        steps: list[ToolLoopStep] = []
        agg = {"in": 0, "out": 0, "cc": 0, "cr": 0}
        # `while rounds < max_rounds` (Gemini R2 F1) — correct at the max_rounds=0
        # boundary (executes ZERO tool rounds, goes straight to the forced answer),
        # unlike a `while True` + post-increment check which would run one round at 0.
        rounds = 0
        while rounds < max_rounds:
            resp = await self._client.complete_messages(
                system_blocks=system_blocks,
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            agg["in"] += resp.input_tokens
            agg["out"] += resp.output_tokens
            agg["cc"] += resp.cache_creation_input_tokens
            agg["cr"] += resp.cache_read_input_tokens

            if resp.stop_reason != "tool_use" or not resp.tool_use:
                return self._result(
                    resp.content, resp.stop_reason, cap_exhausted=False, steps=steps, agg=agg
                )

            rounds += 1
            # Execute every tool_use block of this round, serially (sequential).
            assistant_blocks: list[dict[str, Any]] = []
            if resp.content:
                assistant_blocks.append({"type": "text", "text": resp.content})
            tool_result_blocks: list[dict[str, Any]] = []
            calls: list[ToolCall] = []
            for tu in resp.tool_use:
                assistant_blocks.append(
                    {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input}
                )
                outcome = await executor(tu.name, tu.input)
                calls.append(
                    ToolCall(id=tu.id, name=tu.name, input=tu.input,
                             result=outcome.content, is_error=outcome.is_error)
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }
                )
            steps.append(ToolLoopStep(assistant_text=resp.content, tool_calls=tuple(calls)))
            messages.append({"role": "assistant", "content": assistant_blocks})
            messages.append({"role": "user", "content": tool_result_blocks})

        # Cap reached (or max_rounds==0). ONE final model call WITHOUT tools so the
        # model must answer from what it has — no further tool requests possible, so
        # no dangling tool_use. `messages` ends on the last round's tool_result user
        # turn (or just the user turn if max_rounds==0), both API-valid. Returns
        # cap_exhausted=True regardless of whether final.content is non-empty.
        self._audit.warning("tool_loop_cap_exhausted", rounds=rounds, max_rounds=max_rounds)
        final = await self._client.complete_messages(
            system_blocks=system_blocks, messages=messages, tools=None,
            model=model, max_tokens=max_tokens, temperature=temperature,
        )
        agg["in"] += final.input_tokens
        agg["out"] += final.output_tokens
        agg["cc"] += final.cache_creation_input_tokens
        agg["cr"] += final.cache_read_input_tokens
        return self._result(
            final.content, "cap_exhausted", cap_exhausted=True, steps=steps, agg=agg
        )

    @staticmethod
    def _result(text: str, stop_reason: str, *, cap_exhausted: bool,
                steps: list[ToolLoopStep], agg: dict[str, int]) -> ToolLoopResult:
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
