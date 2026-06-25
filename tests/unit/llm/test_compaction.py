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
