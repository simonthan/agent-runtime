"""In-session conversation compaction — running-summary working memory.

Folds the oldest turns of a long live session into a running prose summary
while preserving the most recent K turns verbatim, keeping the prompt inside
the model's context window. Persona-agnostic: consumers persist the returned
``WorkingMemory.to_dict()`` in their session state (e.g.
``SessionData.data["working_memory"]``) and append ``running_summary`` to their
cache-breakpoint-2 retrieval block.

See teams-bot-platform spec 2026-06-25-chief-of-staff-context-memory-design §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_runtime.logging import AuditLogger, NullAuditLogger

if TYPE_CHECKING:
    from agent_runtime.llm.client import AnthropicClient


_MERGE_SYSTEM_PROMPT = (
    "You maintain a running summary of an ongoing assistant conversation. "
    "Given the existing summary and the next batch of conversation turns, return "
    "a single updated summary that preserves durable facts, decisions, open "
    "threads, and user preferences from BOTH the existing summary and the new "
    "turns. Be concise and factual. Do not add commentary, headers, or "
    "meta-text — return only the updated summary prose."
)


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token).

    Used only for the compaction trigger decision; real token accounting uses
    the ``ClaudeResponse`` token fields returned by the API.
    """
    return len(text) // 4


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """Per-persona compaction tuning."""

    model_window_tokens: int
    threshold_fraction: float = 0.6
    keep_k: int = 6
    summary_model: str | None = None
    summary_max_tokens: int = 1024


@dataclass(frozen=True, slots=True)
class WorkingMemory:
    """Compaction state. Stored by consumers in ``SessionData.data``."""

    running_summary: str | None = None
    summary_token_estimate: int = 0
    last_compacted_turn_index: int = 0
    compaction_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for storage in ``SessionData.data``."""
        return {
            "running_summary": self.running_summary,
            "summary_token_estimate": self.summary_token_estimate,
            "last_compacted_turn_index": self.last_compacted_turn_index,
            "compaction_count": self.compaction_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> WorkingMemory:
        """Restore from a stored dict, or return an empty instance for ``None``."""
        if not d:
            return cls()
        return cls(
            running_summary=d.get("running_summary"),
            summary_token_estimate=d.get("summary_token_estimate", 0),
            last_compacted_turn_index=d.get("last_compacted_turn_index", 0),
            compaction_count=d.get("compaction_count", 0),
        )


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Outcome of a ``maybe_compact`` call."""

    working_memory: WorkingMemory
    compacted: bool


class CompactionEngine:
    """Folds old turns into a running summary when the live prompt grows too large."""

    def __init__(
        self,
        *,
        client: AnthropicClient,
        config: CompactionConfig,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialise with an ``AnthropicClient``, compaction config, and optional audit logger."""
        self._client = client
        self._cfg = config
        self._audit = audit_logger or NullAuditLogger()

    def _verbatim_turns(
        self, history: list[dict[str, Any]], wm: WorkingMemory
    ) -> list[dict[str, Any]]:
        """Turns not yet folded into the summary."""
        return history[wm.last_compacted_turn_index :]

    def _live_token_estimate(
        self,
        *,
        wm: WorkingMemory,
        history: list[dict[str, Any]],
        extra_block_text: str,
    ) -> int:
        """Estimate the live prompt token count from verbatim turns + summary + extra block."""
        verbatim = self._verbatim_turns(history, wm)
        body = (
            extra_block_text
            + (wm.running_summary or "")
            + "".join(str(t.get("content", "")) for t in verbatim)
        )
        return estimate_tokens(body)

    def should_compact(
        self,
        *,
        working_memory: WorkingMemory,
        history: list[dict[str, Any]],
        extra_block_text: str = "",
    ) -> bool:
        """True when there are foldable turns AND the live prompt crosses the threshold."""
        verbatim = self._verbatim_turns(history, working_memory)
        if len(verbatim) <= self._cfg.keep_k:
            return False
        threshold = int(self._cfg.model_window_tokens * self._cfg.threshold_fraction)
        live = self._live_token_estimate(
            wm=working_memory, history=history, extra_block_text=extra_block_text
        )
        return live >= threshold

    def _format_turns(self, turns: list[dict[str, Any]]) -> str:
        """Format a list of turns as ``role: content`` lines for the merge prompt."""
        return "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in turns
        )

    async def _merge_summary(
        self,
        *,
        existing_summary: str | None,
        turns_to_fold: list[dict[str, Any]],
    ) -> str:
        """Call the LLM to merge old turns into the running summary."""
        existing = existing_summary or "(none yet)"
        user_message = (
            f"EXISTING SUMMARY:\n{existing}\n\n"
            f"NEW TURNS TO FOLD IN:\n{self._format_turns(turns_to_fold)}"
        )
        response = await self._client.complete(
            static_system_prefix=_MERGE_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=self._cfg.summary_max_tokens,
            model=self._cfg.summary_model,
        )
        return response.content

    async def maybe_compact(
        self,
        *,
        working_memory: WorkingMemory,
        history: list[dict[str, Any]],
        extra_block_text: str = "",
        session_id: str | None = None,
    ) -> CompactionResult:
        """Compact if over threshold; otherwise return the input unchanged.

        On compaction: folds all verbatim turns except the most recent
        ``keep_k`` into the running summary, advances the covered-index, bumps
        the count, and emits a ``memory_compacted`` event. The LLM merge call is
        only made when compaction actually fires.
        """
        wm = working_memory
        if not self.should_compact(
            working_memory=wm, history=history, extra_block_text=extra_block_text
        ):
            return CompactionResult(working_memory=wm, compacted=False)

        verbatim = self._verbatim_turns(history, wm)
        n_to_fold = len(verbatim) - self._cfg.keep_k
        to_fold = verbatim[:n_to_fold]

        new_summary = await self._merge_summary(
            existing_summary=wm.running_summary, turns_to_fold=to_fold
        )
        new_wm = WorkingMemory(
            running_summary=new_summary,
            summary_token_estimate=estimate_tokens(new_summary),
            last_compacted_turn_index=wm.last_compacted_turn_index + n_to_fold,
            compaction_count=wm.compaction_count + 1,
        )
        self._audit.info(
            "memory_compacted",
            session_id=session_id,
            folded_turns=n_to_fold,
            compaction_count=new_wm.compaction_count,
            summary_token_estimate=new_wm.summary_token_estimate,
        )
        return CompactionResult(working_memory=new_wm, compacted=True)
