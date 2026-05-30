"""Canonical PluginExecutionContext for plugin execution.

Extracted from plugin_executor.py so action handlers can import it without
creating circular dependencies. During the Phase 1 migration, session_data
remains as the live dict backing store for full backward compatibility.
New code can use the typed `state` property for structured access.
"""

from typing import Any


class PluginExecutionContext:
    """Context for plugin execution.

    `user` is an opaque user-identity object provided by the consumer.
    agent_runtime does not introspect it. The state-typed view that lived
    here in ihd's original is deferred until session_state.py lifts.
    """

    def __init__(
        self,
        conversation_id: str,
        user: Any,
        session_data: dict[str, Any],
        client_context: dict[str, Any] | None = None,
    ):
        self.conversation_id = conversation_id
        self.user = user
        self.session_data = session_data
        self.client_context = client_context or {}
        self.variables: dict[str, Any] = {}
        self.current_node: str | None = None
        self.last_response: str | None = None
        self.last_action_result: dict | None = None


__all__ = ["PluginExecutionContext"]
