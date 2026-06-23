"""agent-runtime — shared runtime primitives for Microsoft Teams knowledge bots.

Each subpackage owns one slice of the runtime vocabulary:

- ``safety``     — prompt + tool-result sanitization
- ``resilience`` — circuit breaker fault-tolerance pattern
- ``flows``      — message routing priority chain
- ``context``    — plugin execution context
- ``logging``    — AuditLogger Protocol + NullAuditLogger default
- ``llm``        — Anthropic API wrapper with two-cache-breakpoint contract (extras: ``[llm]``)
- ``connectors`` — BaseConnector ABC, ConnectorResult, RetryMixin, throttle mechanism
- ``protocol``   — NodeResult + NodeHandler/TemplateResolver/NodeExecutor Protocols
- ``transport``  — channel adapters (Microsoft Teams via ``transport.teams``; extras: ``[teams]``)

Consumed by teams-bot-platform and ithelpdesk via git-pinned dependency.
See the three-layer reuse model in teams-bot-platform/ARCHITECTURE.md §2.
"""

__version__ = "0.6.6"
