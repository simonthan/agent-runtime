# Changelog

## v0.5.0 — 2026-06-02

### Added
- `agent_runtime.session` subpackage (optional extras `[redis]`, `[postgres]`):
  Redis-backed conversation session store with Postgres durable resume fallback.
  Lift from ithelpdesk's `app.core.session_*` family with three new surfaces:
  (1) `(user_id, bot_id)` keying per teams-bot-platform `ARCHITECTURE.md` §4 #4;
  (2) `get_or_prompt_resume(...) -> ResumeDecision` sealed-union API for T-008f
  Resume-card UX; (3) `SessionAlreadyActive` typed exception for dispatcher
  pattern-matching when concurrent panes are attempted in v1.
- `pydantic >= 2.6` is now a base runtime dep (was only required by `[llm]`).

### Notes
- ORM-free posture at the library boundary: `ResumeRow` is a Pydantic model;
  no SQLAlchemy mixin. Consumers own their `sessions` table schema, FKs, and RLS.
- Redis key prefix is consumer-configurable via `SessionManager(key_prefix=...)`.
  Resume tokens are scoped by user OID + token, mirroring T-512's SQL-layer fix.
- Atomic lease extension uses `SET ... EX ttl XX` to prevent TOCTOU resurrection
  when a key Redis-evicts between read and write.
- Cold-cache rehydration: a Redis miss + Postgres hit within the idle window
  silently repopulates the cache and returns `Resumable`. Redis restarts no
  longer lose session continuity (same `session_id`, same Resume-card UX).
- **Cold-cache history limitation (v1)**: `ResumeRow` carries metadata only
  (`id`, `user_id`, `bot_id`, `status`, `last_message_at`, `client_context`).
  `data` and `conversation_history` live in Redis only. A cold-cache rehydration
  therefore restores session identity but presents the LLM with an empty turn
  history — the user resumes the same logical session but the model has no
  recall of prior turns. Acceptable for v1 because (a) Redis evictions are rare
  in the 30-min window, (b) the retrieval-snapshot store (teams-bot-platform
  T-008i) is the durable record for replay/audit. v2 will add durable turn
  history to `ResumeRow` if telemetry shows evictions are user-visible.
- IT-specific extension pattern preserved — `session_state_ihd.py` in ithelpdesk
  continues to subclass `ConversationState`, demonstrating the consumer-extension
  contract.

### Breaking changes
- `SessionRepositoryProtocol.upsert_resume_data` and `get_session_for_resume` now
  require `bot_id: str` kwarg. v0.4.0 consumers (none yet for sessions) must
  update their concrete repositories before pinning v0.5.0.
- `SessionManager.update_session` no longer applies an internal model-filter to
  `data` (the IHD consumer model round-trip is gone). Consumers that depended
  on the filter must apply it upstream before calling `update_session`.

## v0.4.0 — 2026-06-02

### Added
- `agent_runtime.transport.teams` subpackage (optional extra `[teams]`): framework-agnostic
  Bot Framework SDK wrapper providing `TeamsAdapter`, `TeamsHandler` Protocol,
  `OutboundChannel` Protocol, `InboundMessage`/`InboundMembersAdded`/`InboundInvoke`
  event dataclasses, and `ConversationRef`. Public test helpers in
  `agent_runtime.transport.teams.testing` (`FakeOutboundChannel` + event factories).
- Optional dependencies: `botbuilder-core>=4.15,<5`, `botbuilder-schema>=4.15,<5`,
  `aiohttp>=3.9,<4` (required by botbuilder's async connector).

### Notes
- Fresh-write subpackage — no per-file ruff ignores added. Code passes `select = ["ALL"]`
  cleanly. Future changes should preserve this property.
- Identity resolution fails closed: inbound activities from users with no resolvable
  email are dropped with a WARNING log; handler is not invoked.

## v0.3.0 — 2026-05-31 (backfilled)

### Added
- `agent_runtime.connectors` — `BaseConnector` ABC, `ConnectorResult`, `RetryMixin`,
  throttle mechanism (lifted from ithelpdesk `service_registry.py` family; T-490a).
- `agent_runtime.protocol` — `NodeResult`, `NodeHandler` / `TemplateResolver` /
  `NodeExecutor` Protocols.

### Notes
- Pre-existing gap — v0.3.0 was released without a CHANGELOG entry. Backfilled here
  for completeness; see git commit `016207f` for the canonical commit history.

## v0.2.0 — 2026-05-30

### Added
- `agent_runtime.llm` subpackage (install via extras: `agent-runtime[llm]`)
- `AnthropicClient` — async wrapper around `anthropic` SDK with opinionated
  two-`cache_control`-breakpoint contract (static system prefix + per-turn
  retrieval block) per teams-bot-platform `ARCHITECTURE.md` §4 decision #5
- `ClaudeResponse` — frozen dataclass with token-usage and cache statistics
- `Message` / `History` — `TypedDict`-based conversation history types
- `LLMError` / `LLMRateLimitError` / `LLMAPIError` / `LLMResponseError` —
  exception hierarchy wrapping SDK exceptions (consumers never import from
  `anthropic` to catch wrapper errors)
- Post-call cache-write detection: AuditLogger `llm_cache_not_written`
  WARNING when `cache_creation_input_tokens == 0` despite caching being
  requested (catches the silent-failure trap when cached blocks are below
  the model's min-cache threshold)

### Notes
- Runtime dep on `anthropic >= 0.42` is **optional** — `agent_runtime.llm`
  is the first subpackage with an external runtime dep; install with
  `pip install agent-runtime[llm]` to opt in
- SDK client is constructor-injected (DI of `AsyncAnthropic`) so consumers
  can share one `httpx` connection pool across many wrapper instances and
  tests can inject a fake without monkeypatching the SDK
- Wrapper is **bot-agnostic** — no `bot_id`/`user_id` knowledge; per-tenant
  budget enforcement happens at the service-layer call site
- v0.1.0 surface unchanged; this is a fully additive release

## v0.1.0 — 2026-05-29

Initial release. Extracted from `ithelpdesk` per
[teams-bot-platform/docs/extraction-inventory-review.md](https://github.com/simonthan/teams-bot-platform/blob/master/docs/extraction-inventory-review.md) (T-001).

### Added
- `agent_runtime.logging.AuditLogger` Protocol + `NullAuditLogger` no-op default
- `agent_runtime.safety` — `sanitize_for_llm_prompt`, `InjectionDetector`
- `agent_runtime.resilience` — `CircuitBreaker` + registry
- `agent_runtime.flows` — `MessageRouter`
- `agent_runtime.context` — `PluginExecutionContext` (without ihd's `.state` property; deferred until `session_state` lifts)

### Notes
- Zero runtime deps; pure stdlib + `typing.Protocol`
- Consumer logger injected via `AuditLogger` Protocol; default `NullAuditLogger` discards events
- ihd's `app.utils.audit_logger.AuditLogger` satisfies the Protocol structurally
