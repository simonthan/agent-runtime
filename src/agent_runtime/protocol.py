"""Protocol types for node handlers.

Defines the structural protocols (NodeHandler, TemplateResolver, NodeExecutor)
and the NodeResult dataclass used as the return type for all handlers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class NodeResult:
    """Result returned by a NodeHandler.handle() call.

    Attributes:
        responses: List of response dicts (type/content/quick_replies) to send to the client.
        next_node: Name of the next node to execute, if any.
        should_stop: True if the flow should pause and wait for user input.
        should_recurse: True if the caller should immediately execute next_node.
        metadata: Arbitrary key/value pairs for handler-specific data.
    """

    responses: list[dict]
    next_node: str | None = None
    should_stop: bool = False
    should_recurse: bool = False
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class TemplateResolver(Protocol):
    """Resolves template strings against an execution context."""

    def resolve(self, template: str, context: Any) -> str:
        ...

    def resolve_user_facing(self, s: str, ctx: Any) -> str:
        """Resolve template variables and apply user-facing hygiene (separator collapse + render-guard)."""
        ...


@runtime_checkable
class NodeExecutor(Protocol):
    """Executes a node in the decision tree and returns a raw response dict."""

    async def execute_node(
        self,
        node_name: str,
        node: dict,
        nodes: dict,
        user_message: str,
        context: Any,
        plugin: dict,
    ) -> dict:
        ...


@runtime_checkable
class NodeHandler(Protocol):
    """Handles a single node type in the decision tree."""

    async def handle(
        self,
        node_name: str,
        node: dict,
        nodes: dict,
        user_message: str,
        context: Any,
        plugin: dict,
    ) -> NodeResult:
        ...
