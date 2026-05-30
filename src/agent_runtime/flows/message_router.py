"""MessageRouter priority chain for message routing (T-321c).

Replaces the scattered if/elif guards in process_message() and _execute_plugin()
with an explicit priority chain. Each handler either claims the message or passes
it down. The ordering is explicit, documented, and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agent_runtime.context import PluginExecutionContext


@dataclass
class RouteResult:
    """Result of a message handler claiming a message.

    Attributes:
        response: The response dict (or list of dicts) to return to the caller.
        claimed: Always True — exists to signal explicit ownership (default True).
        save_session: If True, router calls session_manager.update_session after.
    """

    response: dict[str, Any] | list[dict[str, Any]]
    claimed: bool = True
    save_session: bool = True


class MessageHandler(Protocol):
    """Protocol for a handler in the priority chain.

    Each handler either claims the message (returns RouteResult) or passes
    it to the next handler (returns None).
    """

    async def try_handle(
        self,
        message: str,
        context: PluginExecutionContext,
    ) -> RouteResult | None:
        """Return RouteResult to claim the message, or None to pass."""
        ...


class MessageRouter:
    """Priority chain router for user messages.

    Iterates through handlers in order; the first handler that returns a
    non-None RouteResult wins. The FallbackHandler at the end of the chain
    ensures we never fall through silently.

    Priority order (T-321c, T-382b):
      1. InjectionDetectionHandler  — block injection attempts
      2. FeedbackHandler            — capture `feedback:`-prefixed messages (T-382b)
      3. WindDownHandler            — post-ticket follow-up to Claude
      4. DisengagementHandler       — continue/new-issue quick replies
      5. OTPHandler                 — 6-digit codes during OTP verification
      6. QuickReplyHandler          — exact quick-reply payload matching
      7. GlobalResolutionHandler    — "it's working now" patterns
      8. ExitRampHandler            — "create a ticket" / escalation intent
      9. PluginExecutionHandler     — advance active plugin decision tree
     10. KeywordTriggerHandler      — YAML keyword triggers for fresh conversations
     11. IntentClassificationHandler — LLM classification → plugin selection
     12. FallbackHandler            — route to Claude for general conversation
    """

    def __init__(
        self,
        handlers: list[MessageHandler],
        session_manager: Any = None,
    ):
        self._chain = handlers
        self._session_manager = session_manager

    async def route(
        self,
        message: str,
        context: PluginExecutionContext,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Route a message through the priority chain.

        Args:
            message: The raw user message.
            context: Execution context with session data.

        Returns:
            Response dict (or list of dicts for resolution sequences).

        Raises:
            RuntimeError: If no handler claims the message (FallbackHandler missing).
        """
        for handler in self._chain:
            result = await handler.try_handle(message, context)
            if result is not None:
                if result.save_session and self._session_manager is not None:
                    await self._session_manager.update_session(
                        conversation_id=context.conversation_id,
                        data=context.session_data,
                        replace_data=True,
                    )
                return result.response

        raise RuntimeError(
            f"No handler claimed message for conversation {context.conversation_id!r}. "
            "Ensure FallbackHandler is the last entry in the handler chain."
        )


__all__ = ["RouteResult", "MessageHandler", "MessageRouter"]
