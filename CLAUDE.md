# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`agent-runtime` is a **shared library**, not an application. It packages runtime
primitives (resilience, safety, session flows, LLM access, Teams transport) consumed
by two downstream repos — `teams-bot-platform` and `ithelpdesk` — via a git-pinned
dependency. It is Layer 1 of the three-layer reuse model defined in
`teams-bot-platform/ARCHITECTURE.md §2` (a *sibling* repo, not in this tree).

This "I am a dependency" framing drives nearly every design decision below.

## Commands

```bash
make dev          # uv sync --all-groups --all-extras  (install everything)
make lint         # ruff format --check + ruff check + ty check src/
make format       # ruff format + ruff check --fix  (auto-fix)
make test         # pytest (241 tests, asyncio_mode=auto)
make build        # uv build
make clean        # remove dist/build/caches

# Single test / file / pattern:
uv run pytest tests/session/test_manager_resume.py
uv run pytest tests/unit/llm/test_tool_loop.py::test_name
uv run pytest -k resume
```

Python 3.12+. Tooling is `uv` (build backend `uv_build`), `ruff` (lint+format),
`ty` (type checker — note: *not* mypy). There is no separate run target — this is a
library with no entrypoint.

## Architecture and conventions

### Dependency injection via structural Protocols
Nothing is imported and instantiated internally; collaborators are injected through
the constructor and typed as `Protocol`s. The Anthropic SDK client, Redis client, and
session repository are all DI surfaces (`_AnthropicAPI`, `RedisClientProtocol`,
`SessionRepositoryProtocol`). This gives downstream consumers (a) shared `httpx`
connection pooling and (b) deterministic tests against **fakes** rather than
monkeypatched SDKs. Test fakes live in `*/testing.py` modules (shipped, importable by
consumers) and in `tests/**/fakes.py`.

### AuditLogger is injected everywhere, NullAuditLogger is the default
Every component takes an optional `AuditLogger` (an `agent_runtime.logging` Protocol)
and falls back to `NullAuditLogger()` (silent). Module-level components
(`connectors.base`, `resilience.circuit_breaker`) use a module-global `_default_audit`
set via a `set_audit_logger()` hook the consumer calls once at startup. The library
never picks a concrete logging sink — it only emits structured events.

### Optional extras gate heavy dependencies
Base install pulls only `httpx`, `pydantic`, `tenacity`. Heavy deps are behind extras
in `pyproject.toml`: `[llm]` (anthropic), `[teams]` (botbuilder/aiohttp), `[redis]`,
`[postgres]` (asyncpg). When adding code that needs one of these, gate the import and
add/extend the extra — do not add it to base `dependencies`.

### The "verbatim lift" contract — read before editing lifted modules
Much of `src/` was lifted verbatim from `ithelpdesk` (`session/`, `connectors/base.py`,
`resilience/`, `flows/`, `protocol.py`, `llm/client.py`). The repo runs ruff with
`select = ["ALL"]`, so lifted code violates many rules. **The large
`[tool.ruff.lint.per-file-ignores]` block in `pyproject.toml` is intentional
documentation** — each ignore has a comment stating why (almost always "ihd style").

Rule: **do not "tidy" lifted modules to satisfy the linter.** Style cleanups happen on
the `ithelpdesk` side via stubs and propagate back here on the next version. Freshly
written modules (notably `transport/teams/`, `safety/`) carry **no** per-file-ignores
and must stay clean — keep them passing `select = ["ALL"]` without exceptions.

### Cross-repo decision references
Code comments cite decisions like `ARCH §4 #5`, `D4a`/`D4b`/`D5`, and task IDs
`T-XXX`. These live in `teams-bot-platform`'s docs/tasks, not here. Treat them as
stable invariants — e.g. `ARCH §4 #4` mandates session keying by `(user_id, bot_id)`;
`ARCH §4 #5` mandates the two-cache-breakpoint LLM contract. Don't change behavior a
comment pins to such a decision without surfacing it.

## Module-specific invariants

- **`llm/client.py` — two-cache-breakpoint contract.** Every request carries exactly
  two `cache_control: {type: ephemeral}` breakpoints: one on `static_system_prefix`
  (system), one on `retrieval_block` (user). Cache hits require a **byte-identical**
  prefix *and* identical `model` across turns; a per-call `model=` override fragments
  the cache and fires an `llm_cache_not_written` audit warning — that's expected.
  The wrapper is bot-agnostic and does **not** sanitize input (see next point).

- **`safety/` — sanitize at the boundary, two distinct primitives.**
  `sanitize_for_llm_prompt` is for short *user turns* (collapses whitespace, strips
  ` ``` `/`{{`/`}}`). `sanitize_tool_result` is for *tool/MCP output* (preserves
  newlines and ` ``` `, wraps in an `<tool_output>` "treat as data" envelope,
  neutralizes forged envelope tags case-insensitively). Callers sanitize; the LLM
  wrapper deliberately does not, so the defense layer is auditable. **Always
  `str.replace`/`re.sub` with literal replacements — never `str.format`** (a known
  injection landmine).

- **`llm/tool_loop.py` — `ToolUseLoop` is policy-free.** It owns no cap value, no
  result classification, no MCP knowledge. The consumer supplies the round cap, tools,
  and executor, and classifies the returned `ToolLoopResult`. `ToolResult` couples to
  the consumer by duck-typed attributes (`.content`, `.is_error`) only — no
  `isinstance`, no import cycle.

- **`session/manager.py` — Redis hot path, Postgres durable fallback.** Keyed by
  `(user_id, bot_id)`. Duplicate-create races are rejected with atomic `SET NX`; lease
  extension on resume uses atomic `SET XX`. `get_or_prompt_resume` returns a sealed
  union `ResumeDecision` = `NewSession | Resumable | Active`.

- **`connectors/base.py` — retry + throttle.** `is_retryable_error` classifies
  transient (5xx/429/timeouts) vs permanent (4xx). Per-connector-class rate limits are
  registered globally via `register_rate_limit(class_name, ...)` at import time.
  `_handle_error` returns a generic user-facing message and tucks internal detail into
  `data._internal_error` — never leak internals to the user.

## Release / versioning

Version lives in **two** places that must be bumped together:
`pyproject.toml` `version` and `src/agent_runtime/__init__.py` `__version__`.
⚠️ They are currently out of sync (`__init__.py` lags at `0.6.1`,
`pyproject.toml` is `0.6.3`) — the `__init__.py` bump was missed in recent releases;
fix it when you next touch versioning. Every release also gets a `CHANGELOG.md` entry.

Commit format: `[Action] [scope] (T-XXX)` — e.g. `Add sanitize_tool_result primitive
(T-012a)`. Version bumps are their own commit: `Bump version to X.Y.Z (T-XXX)`.
Feature branches merge with a `Merge T-XXX: … (vX.Y.Z)` commit.
