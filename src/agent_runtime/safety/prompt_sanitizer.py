"""Defense-in-depth sanitization for user-controlled text fed to LLM prompts.

Strips control chars + prompt-injection sentinels; caps length. Use at every
boundary where user input enters an LLM prompt. Always str.replace, never
str.format (per feedback_str_format_db_injection landmine).
"""

# Keep \t (9), \n (10), \r (13); strip every other control char.
_CONTROL_CHARS = "".join(chr(c) for c in range(32) if c not in (9, 10, 13))

_INJECTION_SENTINELS = (
    "```",
    "{{",
    "}}",
    "<|",
    "|>",
    "SYSTEM:",
    "ASSISTANT:",
    "USER:",
    "[INST]",
    "[/INST]",
)


def sanitize_for_llm_prompt(text: str | None, max_len: int = 2000) -> str:
    """Return text safe to interpolate into an LLM prompt.

    - None / non-str → ""
    - Control chars → space
    - Injection sentinels → space
    - Whitespace collapsed
    - Truncated to max_len with "…(truncated)" suffix
    """
    if text is None:
        return ""
    s = str(text)
    for ch in _CONTROL_CHARS:
        s = s.replace(ch, " ")
    for sentinel in _INJECTION_SENTINELS:
        s = s.replace(sentinel, " ")
    s = " ".join(s.split())  # collapse whitespace
    if len(s) > max_len:
        s = s[:max_len] + "…(truncated)"
    return s
