"""Unit tests for ``agent_runtime.llm.round_context`` (T-115j).

No SDK import and no ``importorskip`` — the primitive is pure stdlib, so these run
even when the ``llm`` extra is absent.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_runtime.llm.round_context import (
    ToolRoundContext,
    bind_tool_round,
    current_tool_round,
)


def test_rounds_remaining_and_final_round() -> None:
    assert ToolRoundContext(round_index=1, max_rounds=20).rounds_remaining == 19
    assert ToolRoundContext(round_index=17, max_rounds=20).rounds_remaining == 3
    last = ToolRoundContext(round_index=20, max_rounds=20)
    assert last.rounds_remaining == 0
    assert last.is_final_round is True
    assert ToolRoundContext(round_index=1, max_rounds=20).is_final_round is False


def test_rounds_remaining_is_clamped_never_negative() -> None:
    """D3: a consumer formats this into a message; a negative must be impossible."""
    assert ToolRoundContext(round_index=25, max_rounds=20).rounds_remaining == 0
    assert ToolRoundContext(round_index=1, max_rounds=0).rounds_remaining == 0


def test_unbound_context_is_none() -> None:
    """The compat contract: a reader outside a loop learns nothing and changes nothing."""
    assert current_tool_round() is None


def test_bind_restores_previous_value_on_nesting() -> None:
    """D4: a NESTED loop must restore the outer budget, not clear it."""
    outer = ToolRoundContext(round_index=2, max_rounds=5)
    inner = ToolRoundContext(round_index=1, max_rounds=2)
    with bind_tool_round(outer):
        assert current_tool_round() == outer
        with bind_tool_round(inner):
            assert current_tool_round() == inner
        assert current_tool_round() == outer
    assert current_tool_round() is None


def test_bind_restores_on_exception() -> None:
    # `pytest.raises`, NOT try/except/pass: this repo runs ruff `select = ["ALL"]` and the
    # `tests/**` per-file-ignores do not include SIM, so a suppressed exception is SIM105.
    with (
        pytest.raises(RuntimeError),
        bind_tool_round(ToolRoundContext(round_index=1, max_rounds=3)),
    ):
        raise RuntimeError("boom")
    assert current_tool_round() is None


async def test_sibling_tasks_do_not_see_each_others_context() -> None:
    """D4: contextvars are per-task, so two concurrent turns cannot cross-talk."""
    seen: dict[str, int | None] = {}
    started = asyncio.Event()

    async def turn(label: str, cap: int) -> None:
        with bind_tool_round(ToolRoundContext(round_index=1, max_rounds=cap)):
            started.set()
            await asyncio.sleep(0)
            ctx = current_tool_round()
            seen[label] = None if ctx is None else ctx.max_rounds

    async def bystander() -> None:
        await started.wait()
        ctx = current_tool_round()
        seen["bystander"] = None if ctx is None else ctx.max_rounds

    await asyncio.gather(turn("a", 7), turn("b", 11), bystander())
    assert seen == {"a": 7, "b": 11, "bystander": None}
