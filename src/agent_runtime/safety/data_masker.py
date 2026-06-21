"""PII / secret masking for audit logs and structured payloads.

Generic regex-based redaction — no domain coupling. Consumers call ``mask_string``
on free text and ``mask_dict`` on structured payloads *before* handing them to their
own logging sink (``agent_runtime`` does not own the sink; see ``logging/protocol.py``).

Lifted from ithelpdesk ``app/utils/security.py::DataMasker`` (teams-bot-platform task
T-015a) and reshaped to the ``safety/`` free-function convention (cf.
``sanitize_for_llm_prompt``). Behavioral semantics preserved verbatim, including the
``patterns or ...`` / ``sensitive_keys or ...`` fall-through (an empty list masks all).
Always ``re.sub`` with literal/callable replacements — never ``str.format`` (a known
injection landmine).
"""

import re
from collections.abc import Callable, Sequence
from typing import Any

# name -> (regex, replacer). Partial-value replacers keep the last 4 chars where that
# is still useful (ssn / credit_card / phone); secret-shaped values are fully redacted.
PATTERNS: dict[str, tuple[str, Callable[[re.Match[str]], str]]] = {
    "ssn": (
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        lambda m: "***-**-" + m.group()[-4:],
    ),
    "credit_card": (
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        lambda m: "****-****-****-" + m.group()[-4:],
    ),
    "email": (
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        lambda m: m.group()[:3] + "***@***." + m.group().split(".")[-1],
    ),
    "phone": (
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        lambda m: "***-***-" + m.group()[-4:],
    ),
    "otp": (
        r"\b(?:OTP|otp|code|Code|verification)[:\s]+(\d{6})\b",
        lambda _m: "[OTP REDACTED]",
    ),
    "api_key": (
        r"\b(?:sk|pk|api|key|token)[-_][a-zA-Z0-9]{20,}\b",
        lambda _m: "[API_KEY_REDACTED]",
    ),
    "password": (
        r"(?i)(password|pwd|pass|passwd|secret)[=:]\s*\S+",
        lambda m: re.split(r"[=:]", m.group(), maxsplit=1)[0] + "=********",
    ),
}

# Substrings matched (case-insensitively) against dict keys; matching keys are fully
# masked regardless of value. Mirrors ithelpdesk's default set verbatim.
_DEFAULT_SENSITIVE_KEYS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "ssn",
    "credit_card",
)


def mask_string(text: str, patterns: list[str] | None = None) -> str:
    """Redact sensitive substrings in free text.

    ``patterns``: optional subset of ``PATTERNS`` keys to apply. Falsy (``None`` or
    empty) falls through to *all* patterns (ithelpdesk-compatible). Falsy ``text`` is
    returned unchanged.
    """
    if not text:
        return text
    patterns_to_use = patterns or list(PATTERNS)
    for name in patterns_to_use:
        if name in PATTERNS:
            pattern, replacer = PATTERNS[name]
            text = re.sub(pattern, replacer, text)
    return text


# Cap on recursion into nested dicts/lists. A logging path must not raise
# RecursionError on a pathologically deep (or cyclically shaped) payload — past the
# cap the substructure passes through unmasked rather than crashing the request
# (SEC-4). 64 is far deeper than any realistic audit payload.
_MAX_MASK_DEPTH = 64


# Telemetry-only patterns (Entra OID/GUID + AAD tenant URL). Deliberately NOT in the
# default PATTERNS set: applying these via mask_dict would shred a consumer's
# intentionally-retained id fields (e.g. a forensic user_id=<oid> audit key). They apply
# ONLY to free-text exception/telemetry bodies, via mask_telemetry() (T-021a).
_TELEMETRY_PATTERNS: tuple[tuple[str, Callable[[re.Match[str]], str]], ...] = (
    (
        # RFC 4122 8-4-4-4-12 hex — Entra OIDs / tenant GUIDs, incl. inside URLs
        # (Graph .../users/<oid>/..., login.microsoftonline.com/<tenant-guid>/...).
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
        lambda _m: "[GUID_REDACTED]",
    ),
    (
        # Non-GUID tenant segment the GUID pattern misses (e.g. contoso.onmicrosoft.com).
        # Redact only the tenant path segment; keep the host.
        r"login\.microsoftonline\.com/[^/\s]+",
        lambda _m: "login.microsoftonline.com/[TENANT_REDACTED]",
    ),
)


def mask_telemetry(text: str) -> str:
    """Mask free-text telemetry / exception bodies before logging.

    Applies the default secret/PII patterns (via ``mask_string``) AND telemetry-only
    GUID/OID + AAD-tenant-URL patterns. Use this — NOT ``mask_string`` — on library-
    internal exception text so an OID-bearing Graph URL or tenant token-endpoint URL is
    redacted. Kept separate from the default ``PATTERNS`` so ``mask_string`` / ``mask_dict``
    default behaviour is unchanged: masking GUIDs by default would shred a consumer's
    intentionally-retained id fields (T-021a). Falsy ``text`` returned unchanged.
    """
    if not text:
        return text
    # Apply telemetry-only patterns FIRST (Entra OID/GUID + AAD tenant URL) so the GUID
    # regex runs on raw text before mask_string's credit_card pattern can partially mangle
    # hex sequences, which would prevent the GUID pattern from matching.
    for pattern, replacer in _TELEMETRY_PATTERNS:
        text = re.sub(pattern, replacer, text)
    return mask_string(text)


# Cap on recursion into nested dicts/lists. A logging path must not raise
def mask_dict(
    data: dict[Any, Any],
    sensitive_keys: Sequence[str] | None = None,
    *,
    _depth: int = 0,
) -> dict[Any, Any]:
    """Redact a dict: fully mask values whose key looks sensitive, else scan strings.

    Recurses into nested dicts and lists. ``sensitive_keys``: substrings matched
    case-insensitively against each key; falsy falls through to ``_DEFAULT_SENSITIVE_KEYS``.
    Non-str / non-dict / non-list values pass through unchanged. Non-str keys are
    coerced via ``str()`` before matching (SEC-4) so an ``int``/``float``/``None`` key
    can never raise ``AttributeError`` on a logging path.
    """
    keys = sensitive_keys or _DEFAULT_SENSITIVE_KEYS
    if _depth >= _MAX_MASK_DEPTH:
        return data
    result: dict[Any, Any] = {}
    for key, value in data.items():
        key_lower = str(key).lower()
        if any(sensitive in key_lower for sensitive in keys):
            result[key] = "********"
        elif isinstance(value, str):
            result[key] = mask_string(value)
        elif isinstance(value, dict):
            result[key] = mask_dict(value, keys, _depth=_depth + 1)
        elif isinstance(value, list):
            result[key] = [
                mask_dict(item, keys, _depth=_depth + 1)
                if isinstance(item, dict)
                else mask_string(item)
                if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result
