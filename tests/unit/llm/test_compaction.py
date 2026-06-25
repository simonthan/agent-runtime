"""Unit tests for agent_runtime.llm.compaction."""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from agent_runtime.llm.compaction import (
    CompactionConfig,
    WorkingMemory,
    estimate_tokens,
)


def test_estimate_tokens_is_chars_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_working_memory_round_trips_through_dict() -> None:
    wm = WorkingMemory(
        running_summary="so far we discussed X",
        summary_token_estimate=5,
        last_compacted_turn_index=4,
        compaction_count=1,
    )
    restored = WorkingMemory.from_dict(wm.to_dict())
    assert restored == wm


def test_working_memory_from_none_is_empty() -> None:
    wm = WorkingMemory.from_dict(None)
    assert wm == WorkingMemory()
    assert wm.running_summary is None
    assert wm.last_compacted_turn_index == 0


def test_compaction_config_defaults() -> None:
    cfg = CompactionConfig(model_window_tokens=200_000)
    assert cfg.threshold_fraction == 0.6
    assert cfg.keep_k == 6
    assert cfg.summary_model is None


# ---------------------------------------------------------------------------
# Task 2: should_compact threshold logic
# ---------------------------------------------------------------------------
from agent_runtime.llm.compaction import CompactionEngine  # noqa: E402


def _turns(n: int, *, content: str = "x" * 400) -> list[dict]:
    """n turns, each ~100 estimated tokens (400 chars)."""
    return [{"role": "user", "content": content, "timestamp": "t"} for _ in range(n)]


def test_should_not_compact_when_few_verbatim_turns(client) -> None:
    # keep_k=6; with 6 turns there is nothing to fold even if huge window crossed
    cfg = CompactionConfig(model_window_tokens=100, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)
    assert engine.should_compact(working_memory=WorkingMemory(), history=_turns(6)) is False


def test_should_compact_when_over_threshold_with_foldable_turns(client) -> None:
    # 20 turns * ~100 tokens = ~2000 est; threshold = 0.6 * 1000 = 600 -> compact
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)
    assert engine.should_compact(working_memory=WorkingMemory(), history=_turns(20)) is True


def test_should_not_compact_when_under_threshold(client) -> None:
    # 8 turns * ~100 = ~800 est; threshold = 0.6 * 100000 = 60000 -> no
    cfg = CompactionConfig(model_window_tokens=100_000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)
    assert engine.should_compact(working_memory=WorkingMemory(), history=_turns(8)) is False


def test_should_compact_counts_only_turns_after_last_compacted_index(client) -> None:
    # 20 turns total but 18 already summarized -> only 2 verbatim, below keep_k -> no
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)
    wm = WorkingMemory(last_compacted_turn_index=18)
    assert engine.should_compact(working_memory=wm, history=_turns(20)) is False


# ---------------------------------------------------------------------------
# Task 3: _merge_summary LLM call
# ---------------------------------------------------------------------------
from .fakes import make_ok  # noqa: E402


@pytest.mark.asyncio
async def test_merge_summary_calls_client_and_returns_text(client, fake_sdk) -> None:
    fake_sdk.messages.responses.append(make_ok(text="MERGED SUMMARY"))
    cfg = CompactionConfig(model_window_tokens=1000, summary_model="claude-haiku-4-5-20251001")
    engine = CompactionEngine(client=client, config=cfg)

    out = await engine._merge_summary(
        existing_summary="earlier: user wants weekly reports",
        turns_to_fold=[{"role": "user", "content": "also track the Q3 renewal", "timestamp": "t"}],
    )

    assert out == "MERGED SUMMARY"
    req = fake_sdk.messages.captured_requests[0]
    # both the existing summary and the folded turn content reach the model
    sent = str(req)
    assert "weekly reports" in sent
    assert "Q3 renewal" in sent
    # honors the configured cheaper summary model
    assert req["model"] == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Task 4: maybe_compact orchestration + event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_compact_noop_when_under_threshold(client, fake_sdk) -> None:
    cfg = CompactionConfig(model_window_tokens=100_000, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)
    wm = WorkingMemory()
    result = await engine.maybe_compact(working_memory=wm, history=_turns(8))
    assert result.compacted is False
    assert result.working_memory == wm
    assert fake_sdk.messages.captured_requests == []  # no LLM call


@pytest.mark.asyncio
async def test_maybe_compact_folds_all_but_keep_k(client, fake_sdk) -> None:
    fake_sdk.messages.responses.append(make_ok(text="SUMMARY v1"))
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)

    result = await engine.maybe_compact(working_memory=WorkingMemory(), history=_turns(20))

    assert result.compacted is True
    assert result.working_memory.running_summary == "SUMMARY v1"
    assert result.working_memory.compaction_count == 1
    # 20 turns, keep_k=6 -> folded 14 -> index advances to 14
    assert result.working_memory.last_compacted_turn_index == 14


@pytest.mark.asyncio
async def test_maybe_compact_emits_event(client, fake_sdk, audit) -> None:
    fake_sdk.messages.responses.append(make_ok(text="S"))
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg, audit_logger=audit)

    await engine.maybe_compact(
        working_memory=WorkingMemory(), history=_turns(20), session_id="sess-1"
    )

    events = [e for e in audit.events if e[1] == "memory_compacted"]
    assert len(events) == 1
    _, _, kwargs = events[0]
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["folded_turns"] == 14
    assert kwargs["compaction_count"] == 1


@pytest.mark.asyncio
async def test_planted_early_fact_survives_two_compactions(client, fake_sdk) -> None:
    # The merge fake echoes a sentinel so we can assert the early fact is carried
    # forward into the second summary (running_summary is fed back in).
    fake_sdk.messages.responses.append(make_ok(text="EARLY_FACT preserved; batch 1"))
    fake_sdk.messages.responses.append(make_ok(text="EARLY_FACT preserved; batch 2"))
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg)

    r1 = await engine.maybe_compact(working_memory=WorkingMemory(), history=_turns(20))
    # second batch: more turns appended; previous summary must be fed back to merge
    r2 = await engine.maybe_compact(working_memory=r1.working_memory, history=_turns(40))

    assert r2.compacted is True
    assert r2.working_memory.compaction_count == 2
    # the existing summary from r1 was passed into the second merge call
    second_req = str(fake_sdk.messages.captured_requests[1])
    assert "batch 1" in second_req  # r1 summary fed into r2 merge


# ---------------------------------------------------------------------------
# Task 5: LLM-failure safety
# ---------------------------------------------------------------------------
from agent_runtime.llm import LLMError  # noqa: E402


@pytest.mark.asyncio
async def test_maybe_compact_returns_unchanged_on_llm_error(client, fake_sdk, audit) -> None:
    fake_sdk.messages.exceptions.append(LLMError("boom"))
    cfg = CompactionConfig(model_window_tokens=1000, threshold_fraction=0.6, keep_k=6)
    engine = CompactionEngine(client=client, config=cfg, audit_logger=audit)
    wm = WorkingMemory(running_summary="prior", last_compacted_turn_index=2)

    result = await engine.maybe_compact(working_memory=wm, history=_turns(20))

    assert result.compacted is False
    assert result.working_memory == wm  # unchanged: nothing folded, index intact
    assert any(e[1] == "memory_compaction_failed" for e in audit.events)


# ---------------------------------------------------------------------------
# Task 6: Public exports
# ---------------------------------------------------------------------------


def test_public_exports_from_llm_package() -> None:
    from agent_runtime.llm import (
        CompactionConfig,
        CompactionEngine,
        CompactionResult,
        WorkingMemory,
        estimate_tokens,
    )
