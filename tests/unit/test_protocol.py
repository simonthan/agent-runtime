"""Tests for agent_runtime.protocol — NodeResult dataclass + 3 runtime-checkable Protocols."""

from agent_runtime.protocol import (
    NodeExecutor,
    NodeHandler,
    NodeResult,
    TemplateResolver,
)


class TestNodeResultDefaults:
    def test_minimal_construction(self):
        result = NodeResult(responses=[])
        assert result.responses == []
        assert result.next_node is None
        assert result.should_stop is False
        assert result.should_recurse is False
        assert result.metadata == {}

    def test_metadata_is_independent_per_instance(self):
        a = NodeResult(responses=[])
        b = NodeResult(responses=[])
        a.metadata["key"] = "value"
        assert "key" not in b.metadata  # field(default_factory=dict) wired correctly


class TestProtocolRuntimeChecks:
    def test_node_handler_protocol_accepts_compatible_stub(self):
        class _Stub:
            async def handle(self, node_name, node, nodes, user_message, context, plugin):
                return NodeResult(responses=[])

        assert isinstance(_Stub(), NodeHandler)

    def test_template_resolver_protocol_accepts_compatible_stub(self):
        class _Stub:
            def resolve(self, template, context):
                return template

            def resolve_user_facing(self, s, ctx):
                return s

        assert isinstance(_Stub(), TemplateResolver)

    def test_node_executor_protocol_accepts_compatible_stub(self):
        class _Stub:
            async def execute_node(self, node_name, node, nodes, user_message, context, plugin):
                return {}

        assert isinstance(_Stub(), NodeExecutor)
