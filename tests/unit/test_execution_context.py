"""Unit tests for agent_runtime.context.PluginExecutionContext."""

from unittest.mock import MagicMock

from agent_runtime.context import PluginExecutionContext


class TestConstruction:
    def test_minimal_construction(self):
        ctx = PluginExecutionContext(
            conversation_id="conv-1",
            user=MagicMock(),
            session_data={},
        )
        assert ctx.conversation_id == "conv-1"
        assert ctx.session_data == {}
        assert ctx.client_context == {}
        assert ctx.variables == {}
        assert ctx.current_node is None
        assert ctx.last_response is None
        assert ctx.last_action_result is None

    def test_client_context_defaults_to_empty(self):
        ctx = PluginExecutionContext("c", MagicMock(), {}, client_context=None)
        assert ctx.client_context == {}

    def test_client_context_propagates_when_provided(self):
        cc = {"channel": "teams"}
        ctx = PluginExecutionContext("c", MagicMock(), {}, client_context=cc)
        assert ctx.client_context is cc

    def test_user_is_opaque_any(self):
        for user_obj in [{"id": "u1"}, MagicMock(), "raw-string", None, 42]:
            ctx = PluginExecutionContext("c", user_obj, {})
            assert ctx.user is user_obj


class TestStatePropertyAbsent:
    def test_no_state_property(self):
        ctx = PluginExecutionContext("c", MagicMock(), {})
        assert not hasattr(ctx, "state")

    def test_no_session_state_import(self):
        import agent_runtime.context.execution_context as mod

        assert "SessionState" not in dir(mod)
