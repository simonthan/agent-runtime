# Security & Safety Audit — agent-runtime v0.6.4

**Date:** 2026-06-20
**Scope:** Full codebase (`src/agent_runtime/`, ~3,400 LOC).
**Outcome:** No critical/high-severity vulnerabilities. 3 MEDIUM, 4 LOW defense-in-depth / robustness findings.

## Method & clean results

Swept for the classic landmines — none present: no dynamic code execution, no unsafe
deserialization, no shell/command invocation, no format-string injection vector (the
documented landmine), no disabled TLS verification. Tokens use `secrets.token_urlsafe`
and `uuid4`. Confirmed strong defenses already in place:

- **Fail-closed identity** (`transport/teams/identity.py`) — unidentifiable users dropped.
- **JWT-bypass guard** (`transport/teams/adapter.py:133`) — rejects empty `auth_header`.
- **Session ownership checks** (`session/manager.py:304-309`) — resume verifies `user_id` AND `bot_id`; OID-scoped resume-token keys prevent cross-user reuse.
- **PII-safe LLM error mapping** (`llm/client.py:152-162`) — raises type+status only, no SDK body.
- **Atomic Redis ops** (`SET NX`/`SET XX`) and circuit-breaker callback isolation.

## Architectural note (the recurring theme)

Every safety primitive is **opt-in at the consumer boundary** — the LLM client, tool
loop, and connectors deliberately do not sanitize/mask internally (documented, for
auditability). Consequence: **no fail-safe fallback**. A single call site that forgets
`sanitize_tool_result` / `mask_dict` silently disables that defense with no error. Most
findings below are variations on this theme: the primitives are good; the gap is
enforcement coverage.

## Lift-contract caveat

Tasks SEC-2, SEC-3, SEC-6 touch **verbatim-lifted** modules (`connectors/base.py`,
`resilience/circuit_breaker.py`, `session/manager.py`) carrying intentional
`per-file-ignores`. Per CLAUDE.md the canonical fix usually lands on the `ithelpdesk`
side and propagates back on the next version bump — coordinate before editing in place.
SEC-1/4/5/7 live in freshly-written modules (`safety/`, `transport/teams/`) with no
ignores and can be fixed here directly (must stay passing `select=["ALL"]`).

> Task IDs are provisional (`SEC-N`). This repo delegates durable `T-XXX` IDs to the
> sibling `teams-bot-platform` repo; assign real T-IDs there before `/iav`.

---

## SEC-1 · MEDIUM · Case-insensitive role sentinels in `sanitize_for_llm_prompt`

**File:** `src/agent_runtime/safety/prompt_sanitizer.py:13-50`

`sanitize_for_llm_prompt` strips role sentinels (`SYSTEM:`, `ASSISTANT:`, `USER:`,
`[INST]`, `[/INST]`) with case-**sensitive** `str.replace`, while its sibling
`sanitize_tool_result` was deliberately made case-**insensitive** via `_NEUTRALIZE_RE`
(`re.IGNORECASE`, citing "Opus R3 F1"). A lowercase user-turn injection bypasses the
user-turn sanitizer. Confirmed:

```
sanitize_for_llm_prompt('please system: do x') -> 'please system: do x'   # NOT stripped
sanitize_tool_result('system: ignore previous') -> '... ignore previous'  # stripped
```

**Fix:** compile the role-marker subset into an `re.IGNORECASE` pattern, `re.sub` with a
literal-space replacement (mirror `_NEUTRALIZE_RE` — keep literal replacements, never
templated). Keep the literal ` ``` `/`{{`/`}}` replacements (case-irrelevant).

**Acceptance:** `'system: x'`, `'SYSTEM: x'`, `'[inst] x'` all strip the marker; new test
in `tests/unit/test_prompt_sanitizer.py`; `make lint && make test` green.

## SEC-2 · MEDIUM · Mask secrets/PII in logged exception text

**Files:** `connectors/base.py:176,185,237,241,258`; `resilience/circuit_breaker.py:210`

Full exception strings are logged raw. httpx/driver exceptions routinely embed connection
strings, bearer tokens, or PII. `safety/data_masker.mask_string` ships but is never applied
on these paths, so secrets reach the consumer's audit sink unmasked.

**Fix:** route interpolated exception text through `mask_string` before logging. Lift-contract
modules — decide whether masking lands here or propagates from ihd.

**Acceptance:** an exception whose `str()` contains e.g. `password=hunter2` or a token is
masked in the emitted event; test via a fake `AuditLogger`; lift-contract handling documented;
`make lint && make test` green.

## SEC-3 · MEDIUM · Contain `_internal_error` leak in `ConnectorResult.data`

**File:** `connectors/base.py:170-186`

`_handle_error` returns a generic user-facing `.message` but also packs the full internal
error into `.data["_internal_error"]`. The whole result is returned together; the
"user sees `.message`, logs see `.data`" split is contract-only and unenforced. A consumer
that serializes `.data` to the channel leaks internals (possibly secrets/PII).

**Options (design call — pick one):** (a) mask the value with `mask_string`; (b) drop
`_internal_error` from the returned object, emit via audit logger only; (c) keep + add a
loud field-level warning and consumer guidance. Lift-contract module — coordinate with ihd.

**Acceptance:** chosen option implemented + test; if (a)/(b), assert no raw internal detail
survives in the returned `ConnectorResult`; `make lint && make test` green.

## SEC-4 · LOW · Harden `mask_dict` against non-string keys

**File:** `src/agent_runtime/safety/data_masker.py:95`

`key.lower()` assumes str keys. Confirmed: `mask_dict({1: 'x'})` raises `AttributeError`.
On a logging path this crashes the request — or a consumer's `try/except` falls back to
logging the UNMASKED original (worse).

**Fix:** `key_lower = str(key).lower()`. Optional: depth guard on recursion (line 100) to
avoid `RecursionError` on deeply nested payloads. (`safety/` — no ignores, keep clean.)

**Acceptance:** `mask_dict({1: 'x', 2.0: 'y'})` returns without raising and still masks
string values; test in `tests/unit/test_data_masker.py`; `make lint && make test` green.

## SEC-5 · LOW · Tighten empty-auth-header JWT-bypass guard against whitespace

**File:** `src/agent_runtime/transport/teams/adapter.py:133`

`if not auth_header:` admits a whitespace-only header (`" "` is truthy), which may be
treated as effectively empty downstream — partially defeating the existing bypass guard.

**Fix:** `if not auth_header or not auth_header.strip():`. (`transport/teams` — no ignores.)

**Acceptance:** `process_activity` raises `ValueError` for `""` and `"   "`; test in
`tests/transport/teams/test_adapter.py`; `make lint && make test` green.

## SEC-6 · LOW · Cap `conversation_history` growth in `update_session`

**File:** `src/agent_runtime/session/manager.py:231-237`

`update_session` appends to `conversation_history` with no bound. Within the idle window a
user can spam messages to bloat a single session's serialized JSON in Redis — memory
pressure plus slower (de)serialization every turn (the whole blob is rewritten each update).

**Fix:** configurable cap (e.g. `max_history` ctor arg, drop-oldest when exceeded). Keep
minimal; lift-contract module — coordinate with ihd, avoid expanding the lift surface.

**Acceptance:** history length never exceeds cap, oldest dropped first; test in
`tests/session/`; `make lint && make test` green.

## SEC-7 · LOW · Add Unicode normalization to sanitizers (homoglyph/zero-width evasion)

**File:** `src/agent_runtime/safety/prompt_sanitizer.py` (both functions)

Sentinel matching is ASCII-literal only. Homoglyph/full-width/zero-width variants of role
markers bypass it (e.g. `ＳＹＳＴＥＭ：`, or `s​ystem:`). For tool output the
`<tool_output>` envelope still contains it; for **user turns** the sentinel strip is the
only defense, so that half is higher-value.

**Fix:** before sentinel matching, NFKC-normalize and strip zero-width/format chars
(U+200B–200F, U+FEFF, U+2060). Apply to both functions. Document residual limits.

**Acceptance:** fullwidth and zero-width-laced `system:` variants neutralized in both
functions; tests covering NFKC fold + zero-width strip; `make lint && make test` green.

---

## Suggested order

1. **Quick wins (no lift coordination):** SEC-1, SEC-4, SEC-5 — small, self-contained, safe.
2. **Design call:** SEC-3 (pick mask / drop / document).
3. **Lift-coordinated:** SEC-2, SEC-6 (align with ihd side).
4. **Medium effort:** SEC-7.
