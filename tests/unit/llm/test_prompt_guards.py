from agent_runtime.llm.prompt_guards import GROUNDED_DELIVERY_INSTRUCTIONS


def test_grounded_delivery_guard_content() -> None:
    text = GROUNDED_DELIVERY_INSTRUCTIONS
    text.encode("ascii")  # ASCII-only, matches template conventions
    assert text.startswith("Grounded delivery:")
    assert "verbatim" in text
    assert "not sure" in text
    assert "never" in text
