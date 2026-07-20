"""Shared fixtures for ``tests/unit/observability/``.

The request-id contextvar is process/thread-scoped outside of any asyncio
task boundary, so state set by one test module (e.g. ``test_envelope.py``)
would otherwise leak into the next (alphabetical collection order runs
``test_envelope.py`` before ``test_middleware.py`` before
``test_request_context.py``). Reset it before every test so each test starts
from a known-clean state regardless of collection/execution order.
"""

from __future__ import annotations

import pytest

from agent_runtime.observability.request_context import clear_request_id


@pytest.fixture(autouse=True)
def _reset_request_id():
    clear_request_id()
    yield
    clear_request_id()
