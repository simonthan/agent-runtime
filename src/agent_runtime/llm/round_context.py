"""Tool-round context — which ToolUseLoop round is executing, readable by an
executor WITHOUT a signature change.

WHY A CONTEXTVAR (the backward-compatibility contract, T-115j D1):
``ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]`` is a
two-positional-argument contract consumed by teams-bot-platform AND ithelpdesk.
Adding a third parameter -- required OR optional -- breaks every executor that does
not declare it, because the LOOP is the caller: it would pass the argument
unconditionally and a 2-arg callable raises TypeError. Signature sniffing is worse
(functools.partial, decorated wrappers, *args). A contextvar is strictly additive:
the loop binds it around each executor invocation, an executor that never reads it
behaves byte-for-byte as before, and a reader outside a loop gets None.

POLICY-FREE (tool_loop.py module docstring): this publishes FACTS about the round
only -- its position in the budget and the text the model wrote alongside its
tool_use blocks (T-7154). No advisory text, no "how close is close" threshold, no
user messaging, no decision about whether that text is fit to show -- the consumer
owns all of that. Same division as ``max_rounds`` itself, which the loop enforces
but never chooses.

Precedent: ``agent_runtime.observability.request_context`` (same contextvar shape;
a distinct var name, per that module's own note about consumer collisions).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # ruff select=["ALL"] here: TC003 wants annotation-only stdlib imports guarded
    from collections.abc import Iterator

__all__ = [
    "ToolRoundContext",
    "bind_tool_round",
    "bind_tool_use_id",
    "current_tool_round",
    "current_tool_use_id",
]


@dataclass(frozen=True, slots=True)
class ToolRoundContext:
    """The tool round being executed right now.

    ``round_index`` is 1-BASED and equals the loop's internal ``rounds`` AFTER the
    increment for this round (tool_loop.py ``_drive``), so the first tool round
    reports ``round_index=1``. ``max_rounds`` is the cap the consumer passed to
    ``run()``/``resume()``, verbatim.

    ``assistant_text`` (T-7154) is ALL text blocks of the SAME response as this round's
    tool_use blocks, concatenated with no separator -- exactly what
    ``ToolLoopStep.assistant_text`` later records, and ``""`` when the model wrote none.
    It is model output: UNTRUSTED and UNGUARDED. It is written after earlier rounds'
    tool results, so it can echo content injected through them, and it is
    forward-looking, so it can claim an action that has not happened ("I've sent it").
    No consumer-side output guard has seen it. A resumed round carries the suspending
    round's text. The final answer is never a tool round, so it never appears here.
    Defaulted so keyword construction keeps working; excluded from ``repr`` (keeps model
    text out of logs and tracebacks) and from equality (compares numbers only, as before).
    """

    round_index: int
    max_rounds: int
    assistant_text: str = field(default="", repr=False, compare=False)

    @property
    def rounds_remaining(self) -> int:
        """Further TOOL rounds available AFTER this one. Never negative.

        ``0`` is exact, not approximate: when this round completes,
        ``while rounds < max_rounds`` is false and the loop issues ONE final model
        call with ``tools=None``, so the model cannot call another tool. A consumer
        may state that to the model as fact.
        """
        return max(self.max_rounds - self.round_index, 0)

    @property
    def is_final_round(self) -> bool:
        """True when no further tool round follows this one.

        Written as an attribute comparison rather than ``rounds_remaining == 0``
        so ruff's ``PLR2004`` (magic value in comparison, live under this repo's
        ``select = ["ALL"]``) has nothing to flag. Equivalent by construction.
        """
        return self.max_rounds <= self.round_index


_round_var: ContextVar[ToolRoundContext | None] = ContextVar(
    "agent_runtime_tool_round", default=None
)


def current_tool_round() -> ToolRoundContext | None:
    """The round currently executing, or ``None`` outside a ToolUseLoop round.

    ``None`` is the normal answer for an executor invoked directly by a consumer --
    a pre-loop probe, a replay, a test. Treat it as "no round budget known" and
    change nothing.
    """
    return _round_var.get()


@contextmanager
def bind_tool_round(ctx: ToolRoundContext) -> Iterator[None]:
    """Bind ``ctx`` for the duration of the block, then restore the PREVIOUS value.

    Token-based reset, not ``set(None)``: a NESTED loop (a consumer whose tool runs
    another ToolUseLoop) must restore the outer loop's budget on exit rather than
    zero it. Contextvars are per-asyncio-task, so two turns running as sibling tasks
    each see their own value with no locking.
    """
    token = _round_var.set(ctx)
    try:
        yield
    finally:
        _round_var.reset(token)


_tool_use_id_var: ContextVar[str | None] = ContextVar("agent_runtime_tool_use_id", default=None)


def current_tool_use_id() -> str | None:
    """The Anthropic ``tool_use`` id of the call being executed right now, or ``None``
    outside a ToolUseLoop executor invocation (T-7092).

    Lets an executor correlate its own measurements with the loop's ``ToolCall.id``
    without a signature change (same backward-compatibility contract as
    ``current_tool_round``). Needed once calls in a round can run concurrently: an
    ordered list of per-call measurements no longer arrives in tool_use order.
    """
    return _tool_use_id_var.get()


@contextmanager
def bind_tool_use_id(tool_use_id: str) -> Iterator[None]:
    """Bind ``tool_use_id`` for one executor invocation, then restore the previous value
    (token-based, like ``bind_tool_round``). Inside a concurrent batch each call runs in
    its own asyncio task, so each sees only its own id."""
    token = _tool_use_id_var.set(tool_use_id)
    try:
        yield
    finally:
        _tool_use_id_var.reset(token)
