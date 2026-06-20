"""Guard the dual-source version invariant.

The package version lives in two places that must stay in lockstep:
``pyproject.toml`` ``[project].version`` (the installed distribution metadata) and
``agent_runtime.__version__`` (the runtime constant consumers read). These drifted
silently across v0.6.2/v0.6.3 when a release bumped pyproject but not ``__init__``.
This test fails loudly the next time only one side is bumped.
"""

from importlib.metadata import version

import agent_runtime


def test_runtime_version_matches_distribution_metadata():
    assert agent_runtime.__version__ == version("agent-runtime")
