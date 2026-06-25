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
