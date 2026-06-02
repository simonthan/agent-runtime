# Changelog

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
