import pytest

from agent_runtime.safety import sanitize_for_llm_prompt


class TestPromptSanitizer:
    def test_none_returns_empty_string(self):
        assert sanitize_for_llm_prompt(None) == ""

    def test_non_str_coerced(self):
        assert sanitize_for_llm_prompt(42) == "42"

    @pytest.mark.parametrize(
        "sentinel",
        [
            "```",
            "{{secret}}",
            "}}",
            "SYSTEM:",
            "ASSISTANT:",
            "USER:",
            "[INST]",
            "[/INST]",
            "<|test|>",
        ],
    )
    def test_sentinels_stripped(self, sentinel):
        s = f"prefix {sentinel} suffix"
        out = sanitize_for_llm_prompt(s)
        assert sentinel not in out
        assert "prefix" in out and "suffix" in out

    def test_control_chars_replaced(self):
        s = "hello\x00world\x07evil"
        out = sanitize_for_llm_prompt(s)
        assert "\x00" not in out and "\x07" not in out

    def test_keeps_newlines_and_tabs(self):
        s = "line1\nline2\tcol"
        # collapse_whitespace turns these into single spaces, but they are
        # NOT replaced with the empty string first — they pass through
        # _CONTROL_CHARS unmodified.
        out = sanitize_for_llm_prompt(s)
        assert "line1" in out and "line2" in out and "col" in out

    def test_truncates_long_input(self):
        s = "x" * 5000
        out = sanitize_for_llm_prompt(s, max_len=100)
        assert len(out) <= 100 + len("…(truncated)") + 1
        assert out.endswith("…(truncated)")

    def test_short_input_not_truncated(self):
        s = "short"
        out = sanitize_for_llm_prompt(s, max_len=100)
        assert out == "short"

    def test_empty_string(self):
        assert sanitize_for_llm_prompt("") == ""

    def test_whitespace_only(self):
        assert sanitize_for_llm_prompt("   \n\n  \t ") == ""
