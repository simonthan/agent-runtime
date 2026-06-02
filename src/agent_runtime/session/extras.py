"""Conversation-session extras whitelist — base set lifted from ithelpdesk.

Consumers extend by unioning these BASE_* frozensets with their own
domain-specific extras.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# History cap
# ---------------------------------------------------------------------------

MAX_HISTORY_ENTRIES = 30

# ---------------------------------------------------------------------------
# Transient extras: plugin-scoped keys that must be cleared on plugin end
# ---------------------------------------------------------------------------
#
# BASE_TRANSIENT_EXTRAS — generic plugin/gather lifecycle keys; cleanly
#                         lifts into agent-runtime SessionState.
# Consumers extend by unioning with their own domain-specific transient keys.

BASE_TRANSIENT_EXTRAS: frozenset[str] = frozenset(
    {
        "_gather_thanked",  # T-345b — cleared at end of plugin
        # T-468 R2: clear on plugin reset so an aborted plugin.transfer cannot
        # leave a stale override that misroutes the next successful transfer.
        "_transfer_start_node_override",
        "gathered_fields",
        "gathered_fields_summary",
    }
)

# ---------------------------------------------------------------------------
# Known extras: YAML store_as keys that don't have dedicated typed fields
# ---------------------------------------------------------------------------
#
# BASE_KNOWN_EXTRAS — generic conversation + classifier infrastructure;
#                     cleanly lifts into agent-runtime SessionState.
# Consumers extend by unioning with their own domain-specific known keys.

BASE_KNOWN_EXTRAS: frozenset[str] = frozenset(
    {
        # General
        "history",
        "conversation_history",
        "variables",
        "feedback",
        "gathered_context",
        # T-348 — routing-menu user response (quick-reply or free text)
        "response",
        # T-429: universal classifier output (runs for every plugin)
        "classified_routing",
        # T-429: cache key for classified_routing (sha256(summary+details)[:16])
        "classified_routing_input_hash",
        # T-413b — automation harness correlation (survives plugin boundaries)
        "automation_correlation_id",
    }
)

# Dynamic-prefix keys that land in extras (via from_legacy_dict) and must be
# cleared on plugin end. Subset of utils._PLUGIN_SCOPED_PREFIXES — only the
# prefixes that route to extras rather than dedicated sub-state fields.
_EXTRA_PLUGIN_SCOPED_PREFIXES: tuple[str, ...] = (
    "_completed_",
    "_pending_action_ticket_",
    "_clarification_failures_",
    "_prefill_done_for_",  # T-430 — idempotency flag for context pre-fill at plugin entry
)

# Regex patterns for dynamic key routing
_PENDING_ESCALATION_RE = re.compile(r"^_pending_escalation_(.+)$")
_AI_GATHER_ROUND_RE = re.compile(r"^_ai_gather_(.+)_round$")
_AI_GATHER_PAIRS_RE = re.compile(r"^_ai_gather_(.+)_pairs$")
_CONNECTOR_RESULT_RE = re.compile(r"^_cr:(.+)$")

__all__ = [
    "BASE_KNOWN_EXTRAS",
    "BASE_TRANSIENT_EXTRAS",
    "MAX_HISTORY_ENTRIES",
    "_AI_GATHER_PAIRS_RE",
    "_AI_GATHER_ROUND_RE",
    "_CONNECTOR_RESULT_RE",
    "_EXTRA_PLUGIN_SCOPED_PREFIXES",
    "_PENDING_ESCALATION_RE",
]
