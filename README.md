# agent-runtime

Shared runtime primitives for Microsoft Teams knowledge bots. Consumed by
[teams-bot-platform](https://github.com/simonthan/teams-bot-platform) and
[ithelpdesk](https://github.com/simonthan/ithelpdesk) via git pin.

Layer in the three-layer reuse model defined in
[teams-bot-platform/ARCHITECTURE.md §2](https://github.com/simonthan/teams-bot-platform/blob/master/ARCHITECTURE.md).

## Install

```bash
# Base (no LLM extras)
uv add 'agent-runtime @ git+https://github.com/simonthan/agent-runtime.git@v0.2.0'

# With agent_runtime.llm (installs the `anthropic` SDK)
uv add 'agent-runtime[llm] @ git+https://github.com/simonthan/agent-runtime.git@v0.2.0'
```

## Modules

- `agent_runtime.safety` — `sanitize_for_llm_prompt`, `InjectionDetector`
- `agent_runtime.resilience` — `CircuitBreaker` + registry
- `agent_runtime.flows` — `MessageRouter` priority-chain dispatch
- `agent_runtime.context` — `PluginExecutionContext`
- `agent_runtime.logging` — `AuditLogger` Protocol + `NullAuditLogger` default
- `agent_runtime.llm` — `AnthropicClient` with two-cache-breakpoint contract _(extras: `[llm]`)_

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Development

```bash
make dev          # uv sync --all-groups
make lint         # ruff format --check + ruff check + ty check
make test         # pytest
make build        # uv build
```
