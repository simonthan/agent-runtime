# agent-runtime

Shared runtime primitives for Microsoft Teams knowledge bots. Consumed by
[teams-bot-platform](https://github.com/simonthan/teams-bot-platform) and
[ithelpdesk](https://github.com/simonthan/ithelpdesk) via git pin.

Layer in the three-layer reuse model defined in
[teams-bot-platform/ARCHITECTURE.md §2](https://github.com/simonthan/teams-bot-platform/blob/master/ARCHITECTURE.md).

## Install

```bash
uv add 'agent-runtime @ git+https://github.com/simonthan/agent-runtime.git@v0.1.0'
```

## v0.1.0 modules

- `agent_runtime.safety` — `sanitize_for_llm_prompt`, `InjectionDetector`
- `agent_runtime.resilience` — `CircuitBreaker` + registry
- `agent_runtime.flows` — `MessageRouter` priority-chain dispatch
- `agent_runtime.context` — `PluginExecutionContext`
- `agent_runtime.logging` — `AuditLogger` Protocol + `NullAuditLogger` default

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Development

```bash
make dev          # uv sync --all-groups
make lint         # ruff format --check + ruff check + ty check
make test         # pytest
make build        # uv build
```
