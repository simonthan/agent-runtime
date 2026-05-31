"""Shared fixtures for ``tests/unit/llm/``."""

from __future__ import annotations

from typing import Any

import pytest

# Skip the whole llm-test package when ``[llm]`` extras aren't installed —
# Round 3 critic H3: dev running ``uv sync`` (no ``--all-extras``) would
# otherwise get ModuleNotFoundError at collection time, reddening the full
# suite instead of just skipping the optional subpackage.
pytest.importorskip("anthropic")

from agent_runtime.llm import AnthropicClient

from .fakes import FakeAsyncAnthropic


class RecordingAudit:
    """AuditLogger that records every call for later assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def debug(self, message: str, **kwargs: Any) -> None:
        self.events.append(("debug", message, kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self.events.append(("info", message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self.events.append(("warning", message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self.events.append(("error", message, kwargs))

    def security(self, message: str, **kwargs: Any) -> None:
        self.events.append(("security", message, kwargs))

    def action(self, action: str, result: str, **kwargs: Any) -> None:
        self.events.append(("action", f"{action}:{result}", kwargs))


@pytest.fixture
def fake_sdk() -> FakeAsyncAnthropic:
    return FakeAsyncAnthropic()


@pytest.fixture
def audit() -> RecordingAudit:
    return RecordingAudit()


@pytest.fixture
def client(fake_sdk: FakeAsyncAnthropic, audit: RecordingAudit) -> AnthropicClient:
    return AnthropicClient(client=fake_sdk, audit_logger=audit)  # type: ignore[arg-type]
