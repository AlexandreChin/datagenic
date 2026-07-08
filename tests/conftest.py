"""Shared pytest setup for the whole suite.

Tests are grouped into feature directories that mirror the ``datagenic/`` module
map (engine/, scaffolding/, semantics/, quantities/, market_finance/, scale/,
storage/, …). This root conftest is discovered before any test module is
imported, so it centralises the one cross-cutting need: putting the golden
fixtures package (``tests/golden``) on ``sys.path`` so any test — at any depth —
can ``import golden_fixtures``. Previously every file did this itself with a
brittle ``Path(__file__).parent / "golden"`` insert that broke once files moved
into subdirectories.
"""

import sys
from pathlib import Path

_GOLDEN = Path(__file__).parent / "golden"
if str(_GOLDEN) not in sys.path:
    sys.path.insert(0, str(_GOLDEN))


def pytest_collection_modifyitems(config, items):
    """Auto-apply trait markers from a test's location, so the feature directory is the
    single source of truth and individual test files stay marker-free. These are slices
    for ``pytest -m <marker>``; they do not change default selection (only ``integration``
    is excluded in CI). ``integration`` stays explicit — it means "needs real external
    credentials", which a directory can't infer.
    """
    tests_root = Path(__file__).parent
    for item in items:
        rel = Path(item.fspath).relative_to(tests_root).as_posix()
        if rel.startswith("storage/"):
            item.add_marker("cloud")
        if Path(rel).name in ("test_agent.py", "test_agent_inference.py"):
            item.add_marker("agent")
        if rel == "scale/test_benchmark.py":
            item.add_marker("slow")
