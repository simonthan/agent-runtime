# Changelog

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
