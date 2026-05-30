"""Unit tests for the MessageRouter priority chain (T-321c).

Tests verify:
1. Priority ordering — which handler claims a message before which
2. RouteResult contract — claimed/save_session semantics
3. MessageHandler protocol — try_handle returns RouteResult or None
4. MessageRouter.route() raises RuntimeError when no handler claims
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_runtime.context import PluginExecutionContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = "user-123"
    user.email = "test@example.com"
    user.display_name = "Test User"
    user.roles = ["user"]
    return user


def _make_context(session_data: dict | None = None) -> PluginExecutionContext:
    return PluginExecutionContext(
        conversation_id="conv-123",
        user=_make_user(),
        session_data=session_data or {},
    )


# ---------------------------------------------------------------------------
# RouteResult tests
# ---------------------------------------------------------------------------


class TestRouteResult:
    def test_claimed_by_default(self):
        from agent_runtime.flows import RouteResult

        r = RouteResult(response={"type": "text", "content": "hi"})
        assert r.claimed is True

    def test_save_session_by_default(self):
        from agent_runtime.flows import RouteResult

        r = RouteResult(response={"type": "text", "content": "hi"})
        assert r.save_session is True

    def test_can_set_save_session_false(self):
        from agent_runtime.flows import RouteResult

        r = RouteResult(response={"type": "text", "content": "hi"}, save_session=False)
        assert r.save_session is False

    def test_response_list_allowed(self):
        from agent_runtime.flows import RouteResult

        msgs = [{"type": "text", "content": "a"}, {"type": "text", "content": "b"}]
        r = RouteResult(response=msgs)
        assert r.response is msgs


# ---------------------------------------------------------------------------
# MessageRouter core behaviour
# ---------------------------------------------------------------------------


class TestMessageRouterCore:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_no_handler_claims(self):
        """FallbackHandler is missing — router must raise, not silently return None."""
        from agent_runtime.flows import MessageRouter

        pass_handler = MagicMock()
        pass_handler.try_handle = AsyncMock(return_value=None)

        router = MessageRouter(handlers=[pass_handler], session_manager=MagicMock())
        context = _make_context()

        with pytest.raises(RuntimeError, match="No handler claimed message"):
            await router.route("hello", context)

    @pytest.mark.asyncio
    async def test_first_claiming_handler_wins(self):
        """Router returns the first non-None RouteResult."""
        from agent_runtime.flows import MessageRouter, RouteResult

        first = MagicMock()
        first.try_handle = AsyncMock(
            return_value=RouteResult(
                response={"type": "text", "content": "first"},
            )
        )

        second = MagicMock()
        second.try_handle = AsyncMock(
            return_value=RouteResult(
                response={"type": "text", "content": "second"},
            )
        )

        router = MessageRouter(
            handlers=[first, second],
            session_manager=AsyncMock(),
        )
        context = _make_context()
        result = await router.route("hello", context)

        assert result == {"type": "text", "content": "first"}
        second.try_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_passing_handlers(self):
        """Router skips handlers that return None and continues to claimer."""
        from agent_runtime.flows import MessageRouter, RouteResult

        passer = MagicMock()
        passer.try_handle = AsyncMock(return_value=None)

        claimer = MagicMock()
        claimer.try_handle = AsyncMock(
            return_value=RouteResult(
                response={"type": "text", "content": "claimed"},
            )
        )

        router = MessageRouter(
            handlers=[passer, claimer],
            session_manager=AsyncMock(),
        )
        context = _make_context()
        result = await router.route("hello", context)

        assert result == {"type": "text", "content": "claimed"}
        passer.try_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_saves_session_when_save_session_true(self):
        """Router calls session_manager.update_session when save_session=True."""
        from agent_runtime.flows import MessageRouter, RouteResult

        session_manager = AsyncMock()
        session_manager.update_session = AsyncMock()

        claimer = MagicMock()
        claimer.try_handle = AsyncMock(
            return_value=RouteResult(
                response={"type": "text", "content": "ok"},
                save_session=True,
            )
        )

        router = MessageRouter(handlers=[claimer], session_manager=session_manager)
        context = _make_context()
        await router.route("hello", context)

        session_manager.update_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_session_save_when_save_session_false(self):
        """Router does NOT call update_session when save_session=False."""
        from agent_runtime.flows import MessageRouter, RouteResult

        session_manager = AsyncMock()
        session_manager.update_session = AsyncMock()

        claimer = MagicMock()
        claimer.try_handle = AsyncMock(
            return_value=RouteResult(
                response={"type": "text", "content": "ok"},
                save_session=False,
            )
        )

        router = MessageRouter(handlers=[claimer], session_manager=session_manager)
        context = _make_context()
        await router.route("hello", context)

        session_manager.update_session.assert_not_called()
