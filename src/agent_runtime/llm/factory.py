"""Provider factory for the Anthropic SDK client injected into ``AnthropicClient``.

Selects between the public Anthropic API (``AsyncAnthropic`` → api.anthropic.com)
and **Azure AI Foundry** (``AsyncAnthropicFoundry`` → the in-tenant ``/anthropic/``
passthrough, keeping request/response data inside the Azure tenant).

``AsyncAnthropicFoundry`` IS an ``AsyncAnthropic`` subclass, so both satisfy the
``AnthropicClient`` injected-client Protocol and the two-cache-breakpoint contract
is identical across providers — this is a *transport* change, not a *contract* one.

Provider-agnostic: reads NO env vars (config injection is the consumer's job, same
as the bot-agnostic wrapper). Foundry auth is **API-key only** — the SDK sets the
Foundry auth header(s) automatically from ``api_key`` (``x-api-key`` and/or
``api-key`` depending on SDK version; 0.109 sends both, 0.105 only ``api-key``).

NOTE: although this factory reads no env vars, the underlying ``AsyncAnthropicFoundry``
falls back to the SDK's own ``ANTHROPIC_FOUNDRY_{API_KEY,RESOURCE,BASE_URL}`` OS env
vars when a constructor arg is ``None``. Keep those UNSET in foundry deploys (TBP
injects ``TBP_``-prefixed config explicitly) so a polluted process env can't trip the
SDK's resource/base_url mutual-exclusion check and surface as a 500.
"""

from __future__ import annotations

from typing import Literal

from anthropic import AsyncAnthropic, AsyncAnthropicFoundry

__all__ = ["build_anthropic_sdk_client"]

Provider = Literal["anthropic", "foundry"]


def build_anthropic_sdk_client(
    *,
    provider: Provider = "anthropic",
    api_key: str,
    foundry_resource: str | None = None,
    foundry_base_url: str | None = None,
) -> AsyncAnthropic:
    """Construct the SDK client for the chosen inference provider.

    - ``provider="anthropic"`` → ``AsyncAnthropic(api_key=...)`` (public API).
    - ``provider="foundry"``  → ``AsyncAnthropicFoundry`` against the Azure tenant.
      Requires **exactly one** of ``foundry_resource`` (→ the
      ``https://{resource}.services.ai.azure.com/anthropic/`` URL) or an explicit
      ``foundry_base_url``; the SDK raises if both are passed, so we fail earlier
      with a clearer message.

    Fail-fast on bad config so a misconfigured deploy surfaces at construction,
    not as an opaque 4xx mid-conversation. The caller still wraps the result in
    ``AnthropicClient(client=..., default_model=...)``.
    """
    if not api_key:
        raise ValueError("api_key must be non-empty")

    if provider == "anthropic":
        return AsyncAnthropic(api_key=api_key)

    if provider == "foundry":
        # xor: exactly one of resource / base_url. Truthy checks so "" counts as unset.
        if bool(foundry_resource) == bool(foundry_base_url):
            raise ValueError(
                "foundry provider requires exactly one of foundry_resource or foundry_base_url"
            )
        if foundry_base_url:
            return AsyncAnthropicFoundry(api_key=api_key, base_url=foundry_base_url)
        return AsyncAnthropicFoundry(api_key=api_key, resource=foundry_resource)

    raise ValueError(f"unknown provider: {provider!r}")
