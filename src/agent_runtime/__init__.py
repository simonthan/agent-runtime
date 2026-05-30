"""agent-runtime — shared runtime primitives for Microsoft Teams knowledge bots.

Each subpackage owns one slice of the runtime vocabulary:

- ``safety``     — prompt sanitization + injection detection
- ``resilience`` — circuit breaker fault-tolerance pattern
- ``flows``      — message routing priority chain
- ``context``    — plugin execution context
- ``logging``    — AuditLogger Protocol + NullAuditLogger default

Consumed by teams-bot-platform and ithelpdesk via git-pinned dependency.
See the three-layer reuse model in teams-bot-platform/ARCHITECTURE.md §2.
"""

__version__ = "0.1.0"
