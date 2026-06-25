# CompactionEngine Implementation Plan (agent-runtime)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persona-agnostic in-session conversation compaction primitive to `agent-runtime`: fold the oldest turns into a running prose summary while keeping the most recent K turns verbatim, so long sessions stay inside the model context window.

**Architecture:** A new `agent_runtime.llm.compaction` module exposing `estimate_tokens()`, value types (`CompactionConfig`, `WorkingMemory`, `CompactionResult`), and a `CompactionEngine` that calls the existing `AnthropicClient.complete()` to merge turns into the summary and emits a `memory_compacted` audit event. No changes to `SessionData` — consumers persist `WorkingMemory.to_dict()` in `SessionData.data["working_memory"]`. Ends with a minor version bump (0.6.8 → 0.7.0) and a release tag so `teams-bot-platform` can pin it.

**Tech Stack:** Python 3.12, uv, ruff, ty, pytest + pytest-asyncio. Reuses `AnthropicClient` (`src/agent_runtime/llm/client.py`), `AuditLogger`/`NullAuditLogger` (`src/agent_runtime/logging/`), and the `FakeAsyncAnthropic` test fakes (`tests/unit/llm/fakes.py`).

**Spec:** `teams-bot-platform/docs/superpowers/specs/2026-06-25-chief-of-staff-context-memory-design.md` §3 (working memory), §6 (components), §10 (testing).

---

## File Structure

- **Create** `src/agent_runtime/llm/compaction.py` — the whole feature (value types + engine + estimator). One module, one responsibility: in-session compaction.
- **Modify** `src/agent_runtime/llm/__init__.py` — export the public surface.
- **Create** `tests/unit/llm/test_compaction.py` — unit tests (mirrors `tests/unit/llm/test_tool_loop.py` style; reuses `client`/`audit`/`fake_sdk` fixtures from `tests/unit/llm/conftest.py`).
- **Modify** `pyproject.toml` (line 3) and `src/agent_runtime/__init__.py` (`__version__`) — version bump.

Turn shape (matches `SessionManager.update_session` appends): `{"role": str, "content": str, "timestamp": str}`.

---

### Task 1: Value types + token estimator

**Files:**
- Create: `src/agent_runtime/llm/compaction.py`
- Test: `tests/unit/llm/test_compaction.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/llm/test_compaction.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_runtime.llm.compaction'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/agent_runtime/llm/compaction.py
"""In-session conversation compaction — running-summary working memory.

Folds the oldest turns of a long live session into a running prose summary
while preserving the most recent K turns verbatim, keeping the prompt inside
the model's context window. Persona-agnostic: consumers persist the returned
``WorkingMemory.to_dict()`` in their session state (e.g.
``SessionData.data["working_memory"]``) and append ``running_summary`` to their
cache-breakpoint-2 retrieval block.

See teams-bot-platform spec 2026-06-25-chief-of-staff-context-memory-design §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_runtime.logging import AuditLogger, NullAuditLogger

if TYPE_CHECKING:
    from agent_runtime.llm.client import AnthropicClient


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token).

    Used only for the compaction trigger decision; real token accounting uses
    the ``ClaudeResponse`` token fields returned by the API.
    """
    return len(text) // 4


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """Per-persona compaction tuning."""

    model_window_tokens: int
    threshold_fraction: float = 0.6
    keep_k: int = 6
    summary_model: str | None = None
    summary_max_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class WorkingMemory:
    """Compaction state. Stored by consumers in ``SessionData.data``."""

    running_summary: str | None = None
    summary_token_estimate: int = 0
    last_compacted_turn_index: int = 0
    compaction_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "running_summary": self.running_summary,
            "summary_token_estimate": self.summary_token_estimate,
            "last_compacted_turn_index": self.last_compacted_turn_index,
            "compaction_count": self.compaction_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> WorkingMemory:
        if not d:
            return cls()
        return cls(
            running_summary=d.get("running_summary"),
            summary_token_estimate=d.get("summary_token_estimate", 0),
            last_compacted_turn_index=d.get("last_compacted_turn_index", 0),
            compaction_count=d.get("compaction_count", 0),
        )


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Outcome of a ``maybe_compact`` call."""

    working_memory: WorkingMemory
    compacted: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/llm/test_compaction.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/compaction.py tests/unit/llm/test_compaction.py
git commit -m "Add compaction value types + token estimator (T-XXX)"
```

---

### Task 2: `should_compact` threshold logic

**Files:**
- Modify: `src/agent_runtime/llm/compaction.py`
- Test: `tests/unit/llm/test_compaction.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/llm/test_compaction.py
from agent_runtime.llm.compaction import CompactionEngine


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k should_compact -v`
Expected: FAIL — `ImportError: cannot import name 'CompactionEngine'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/agent_runtime/llm/compaction.py`:

```python
class CompactionEngine:
    """Folds old turns into a running summary when the live prompt grows too large."""

    def __init__(
        self,
        *,
        client: AnthropicClient,
        config: CompactionConfig,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._client = client
        self._cfg = config
        self._audit = audit_logger or NullAuditLogger()

    def _verbatim_turns(
        self, history: list[dict], wm: WorkingMemory
    ) -> list[dict]:
        """Turns not yet folded into the summary."""
        return history[wm.last_compacted_turn_index :]

    def _live_token_estimate(
        self,
        *,
        wm: WorkingMemory,
        history: list[dict],
        extra_block_text: str,
    ) -> int:
        verbatim = self._verbatim_turns(history, wm)
        body = (
            extra_block_text
            + (wm.running_summary or "")
            + "".join(str(t.get("content", "")) for t in verbatim)
        )
        return estimate_tokens(body)

    def should_compact(
        self,
        *,
        working_memory: WorkingMemory,
        history: list[dict],
        extra_block_text: str = "",
    ) -> bool:
        """True when there are foldable turns AND the live prompt crosses the threshold."""
        verbatim = self._verbatim_turns(history, working_memory)
        if len(verbatim) <= self._cfg.keep_k:
            return False
        threshold = int(self._cfg.model_window_tokens * self._cfg.threshold_fraction)
        live = self._live_token_estimate(
            wm=working_memory, history=history, extra_block_text=extra_block_text
        )
        return live >= threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k should_compact -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/compaction.py tests/unit/llm/test_compaction.py
git commit -m "Add CompactionEngine.should_compact threshold logic (T-XXX)"
```

---

### Task 3: Summary merge call (`_merge_summary`)

**Files:**
- Modify: `src/agent_runtime/llm/compaction.py`
- Test: `tests/unit/llm/test_compaction.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/llm/test_compaction.py
from .fakes import make_ok


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k merge_summary -v`
Expected: FAIL — `AttributeError: 'CompactionEngine' object has no attribute '_merge_summary'`

- [ ] **Step 3: Write minimal implementation**

Append the constant and method to `src/agent_runtime/llm/compaction.py` (constant near the top, after `estimate_tokens`):

```python
_MERGE_SYSTEM_PROMPT = (
    "You maintain a running summary of an ongoing assistant conversation. "
    "Given the existing summary and the next batch of conversation turns, return "
    "a single updated summary that preserves durable facts, decisions, open "
    "threads, and user preferences from BOTH the existing summary and the new "
    "turns. Be concise and factual. Do not add commentary, headers, or "
    "meta-text — return only the updated summary prose."
)
```

Add the method to `CompactionEngine`:

```python
    def _format_turns(self, turns: list[dict]) -> str:
        return "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns
        )

    async def _merge_summary(
        self,
        *,
        existing_summary: str | None,
        turns_to_fold: list[dict],
    ) -> str:
        existing = existing_summary or "(none yet)"
        user_message = (
            f"EXISTING SUMMARY:\n{existing}\n\n"
            f"NEW TURNS TO FOLD IN:\n{self._format_turns(turns_to_fold)}"
        )
        response = await self._client.complete(
            static_system_prefix=_MERGE_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=self._cfg.summary_max_tokens,
            model=self._cfg.summary_model,
        )
        return response.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k merge_summary -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/compaction.py tests/unit/llm/test_compaction.py
git commit -m "Add CompactionEngine summary-merge call (T-XXX)"
```

---

### Task 4: `maybe_compact` orchestration + event

**Files:**
- Modify: `src/agent_runtime/llm/compaction.py`
- Test: `tests/unit/llm/test_compaction.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/llm/test_compaction.py


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k maybe_compact -v`
Expected: FAIL — `AttributeError: 'CompactionEngine' object has no attribute 'maybe_compact'`

- [ ] **Step 3: Write minimal implementation**

Append the method to `CompactionEngine`:

```python
    async def maybe_compact(
        self,
        *,
        working_memory: WorkingMemory,
        history: list[dict],
        extra_block_text: str = "",
        session_id: str | None = None,
    ) -> CompactionResult:
        """Compact if over threshold; otherwise return the input unchanged.

        On compaction: folds all verbatim turns except the most recent
        ``keep_k`` into the running summary, advances the covered-index, bumps
        the count, and emits a ``memory_compacted`` event. The LLM merge call is
        only made when compaction actually fires.
        """
        wm = working_memory
        if not self.should_compact(
            working_memory=wm, history=history, extra_block_text=extra_block_text
        ):
            return CompactionResult(working_memory=wm, compacted=False)

        verbatim = self._verbatim_turns(history, wm)
        n_to_fold = len(verbatim) - self._cfg.keep_k
        to_fold = verbatim[:n_to_fold]

        new_summary = await self._merge_summary(
            existing_summary=wm.running_summary, turns_to_fold=to_fold
        )
        new_wm = WorkingMemory(
            running_summary=new_summary,
            summary_token_estimate=estimate_tokens(new_summary),
            last_compacted_turn_index=wm.last_compacted_turn_index + n_to_fold,
            compaction_count=wm.compaction_count + 1,
        )
        self._audit.info(
            "memory_compacted",
            session_id=session_id,
            folded_turns=n_to_fold,
            compaction_count=new_wm.compaction_count,
            summary_token_estimate=new_wm.summary_token_estimate,
        )
        return CompactionResult(working_memory=new_wm, compacted=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/llm/test_compaction.py -v`
Expected: PASS (all compaction tests green)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/compaction.py tests/unit/llm/test_compaction.py
git commit -m "Add CompactionEngine.maybe_compact orchestration (T-XXX)"
```

---

### Task 5: LLM-failure safety (history preserved on merge error)

**Files:**
- Modify: `src/agent_runtime/llm/compaction.py`
- Test: `tests/unit/llm/test_compaction.py`

Spec §9: a failed compaction call must never drop turns — return the input working-memory unchanged and let the next threshold crossing retry.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/llm/test_compaction.py
from agent_runtime.llm import LLMError


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k on_llm_error -v`
Expected: FAIL — the raw `LLMError` propagates (no try/except yet)

- [ ] **Step 3: Write minimal implementation**

In `maybe_compact`, wrap the merge call. Replace the `new_summary = await self._merge_summary(...)` line with:

```python
        try:
            new_summary = await self._merge_summary(
                existing_summary=wm.running_summary, turns_to_fold=to_fold
            )
        except Exception as exc:  # noqa: BLE001 — compaction is best-effort; never drop turns
            self._audit.warning(
                "memory_compaction_failed", session_id=session_id, error=str(exc)
            )
            return CompactionResult(working_memory=wm, compacted=False)
```

Note: `LLMError` is the documented failure surface of `AnthropicClient.complete`; the broad catch is the deliberate best-effort contract from spec §9.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/llm/test_compaction.py -v`
Expected: PASS (all green)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/compaction.py tests/unit/llm/test_compaction.py
git commit -m "Add best-effort failure handling to maybe_compact (T-XXX)"
```

---

### Task 6: Public exports

**Files:**
- Modify: `src/agent_runtime/llm/__init__.py`
- Test: `tests/unit/llm/test_compaction.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/llm/test_compaction.py
def test_public_exports_from_llm_package() -> None:
    from agent_runtime.llm import (  # noqa: F401
        CompactionConfig,
        CompactionEngine,
        CompactionResult,
        WorkingMemory,
        estimate_tokens,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/llm/test_compaction.py -k public_exports -v`
Expected: FAIL — `ImportError: cannot import name 'CompactionEngine' from 'agent_runtime.llm'`

- [ ] **Step 3: Write minimal implementation**

In `src/agent_runtime/llm/__init__.py`, add the import block (after the `tool_loop` import) and the `__all__` entries (keep `__all__` alphabetically sorted, matching the existing style):

```python
from agent_runtime.llm.compaction import (
    CompactionConfig,
    CompactionEngine,
    CompactionResult,
    WorkingMemory,
    estimate_tokens,
)
```

Add to `__all__` (in sorted position): `"CompactionConfig"`, `"CompactionEngine"`, `"CompactionResult"`, `"WorkingMemory"`, `"estimate_tokens"`.

- [ ] **Step 4: Run the full suite + lint**

Run: `uv run pytest tests/unit/llm/test_compaction.py -v && make lint`
Expected: PASS; ruff + ty clean. (If ruff reports `__all__` ordering, run `make format` and re-commit.)

- [ ] **Step 5: Commit**

```bash
git add src/agent_runtime/llm/__init__.py tests/unit/llm/test_compaction.py
git commit -m "Export compaction primitives from agent_runtime.llm (T-XXX)"
```

---

### Task 7: Version bump + release tag

**Files:**
- Modify: `pyproject.toml:3`
- Modify: `src/agent_runtime/__init__.py` (`__version__`)

agent-runtime keeps the version in **two** places that must stay in sync (per `CLAUDE.md`). Both are currently `0.6.8`; bump both to `0.7.0` (new feature → minor).

- [ ] **Step 1: Bump `pyproject.toml`**

Change line 3 from `version = "0.6.8"` to `version = "0.7.0"`.

- [ ] **Step 2: Bump `src/agent_runtime/__init__.py`**

Change `__version__ = "0.6.8"` to `__version__ = "0.7.0"`.

- [ ] **Step 3: Full verification before tagging**

Run: `make lint && make test && make build`
Expected: lint clean, all tests pass, wheel builds. Do not tag if any step fails.

- [ ] **Step 4: Commit and tag**

```bash
git add pyproject.toml src/agent_runtime/__init__.py
git commit -m "Bump version to 0.7.0 (T-XXX)"
git tag v0.7.0
git push && git push --tags
```

Expected: tag `v0.7.0` pushed. This is the tag `teams-bot-platform` Plan 2 pins.

---

## Self-Review

- **Spec coverage:** §3.1 working_memory shape → Task 1. §3.2 running summary in retrieval block → consumed by `extra_block_text` param (Task 2) + Plan 2 wiring. §3.3 deterministic trigger + keep_k + `memory_compacted` event → Tasks 2, 4. §3.4 running summary not sliding window → Tasks 3, 4 (summarize-and-merge). §9 LLM-failure preserves history → Task 5. §10 unit tests incl. planted-fact-survives-2-compactions, recent-K retained, threshold boundary, event assertion → Tasks 2, 4. ✅
- **Placeholder scan:** none. `T-XXX` in commit messages is the task-ID convention; the implementer substitutes the real ID when this plan is promoted to a task stub.
- **Type consistency:** `WorkingMemory`, `CompactionConfig`, `CompactionResult`, `CompactionEngine`, `estimate_tokens` named identically across all tasks; `maybe_compact`/`should_compact`/`_merge_summary` signatures stable. ✅
- **Dependency lockfiles:** no dependency manifest changes (uses existing `[llm]` extra / `anthropic`). No `uv.lock` change needed. ✅

## Notes for Plan 2 (tbp)

- The consumer calls `engine.maybe_compact(working_memory=WorkingMemory.from_dict(session.data.get("working_memory")), history=session.conversation_history, extra_block_text=memory_block or "")` after each turn, then persists `result.working_memory.to_dict()` into `session.data["working_memory"]` via `SessionManager.update_session(..., data=...)`.
- The verbatim tail to send the model is `history[result.working_memory.last_compacted_turn_index:]`; the summary goes into `retrieval_block` alongside the T-027 memory block.
- Set the persona's `SessionManager(max_history=None)` (or high) so compaction — not the crude drop-oldest cap — governs context.
