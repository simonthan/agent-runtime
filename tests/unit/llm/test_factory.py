"""Unit tests for ``agent_runtime.llm.build_anthropic_sdk_client``.

Pure construction — no network. ``AsyncAnthropic`` / ``AsyncAnthropicFoundry``
build their httpx clients lazily, so isinstance + base_url assertions are safe.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")  # file-level guard (matches test_tool_loop.py)

from anthropic import AsyncAnthropic, AsyncAnthropicFoundry

from agent_runtime.llm import build_anthropic_sdk_client


@pytest.fixture(autouse=True)
def _clear_foundry_env(monkeypatch) -> None:
    """Isolate from the SDK's own ANTHROPIC_FOUNDRY_* env fallback (determinism)."""
    for var in (
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_FOUNDRY_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_anthropic_provider_returns_public_client() -> None:
    client = build_anthropic_sdk_client(provider="anthropic", api_key="sk-test")
    # exact type — Foundry is a subclass, so use `type is` to distinguish providers
    assert type(client) is AsyncAnthropic
    assert "api.anthropic.com" in str(client.base_url)


def test_default_provider_is_anthropic() -> None:
    client = build_anthropic_sdk_client(api_key="sk-test")
    assert type(client) is AsyncAnthropic


def test_foundry_provider_with_resource_builds_tenant_url() -> None:
    client = build_anthropic_sdk_client(
        provider="foundry", api_key="key", foundry_resource="my-res"
    )
    assert isinstance(client, AsyncAnthropicFoundry)
    assert str(client.base_url) == "https://my-res.services.ai.azure.com/anthropic/"


def test_foundry_provider_with_base_url() -> None:
    client = build_anthropic_sdk_client(
        provider="foundry",
        api_key="key",
        foundry_base_url="https://custom.example.com/anthropic/",
    )
    assert isinstance(client, AsyncAnthropicFoundry)
    assert "custom.example.com" in str(client.base_url)


def test_empty_api_key_raises() -> None:
    with pytest.raises(ValueError, match="api_key must be non-empty"):
        build_anthropic_sdk_client(provider="anthropic", api_key="")


def test_foundry_without_resource_or_base_url_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_anthropic_sdk_client(provider="foundry", api_key="key")


def test_foundry_with_both_resource_and_base_url_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_anthropic_sdk_client(
            provider="foundry",
            api_key="key",
            foundry_resource="r",
            foundry_base_url="https://x/anthropic/",
        )


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        build_anthropic_sdk_client(provider="bedrock", api_key="key")  # type: ignore[arg-type]
