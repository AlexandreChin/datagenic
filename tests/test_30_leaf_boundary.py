"""Package split Part 3 — datagenic (generation) is a standalone leaf.

The generation leaf (`scaffolding`, `engine`, `generators`, `postprocessors`, generation-side
`orchestration`) must not import any platform module (`opensee_core` / dms / apicalls
/ scripts / data_pusher / patches / opensee_apis). The platform may import generation
(e.g. `ChunkedRun`), never the reverse — that direction is asserted in the opensee-core
suite. Importing `datagenic` itself must stay import-light (no platform, no cloud SDK).

Checks run in a clean subprocess so they reflect a fresh interpreter.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_PLATFORM = (
    "opensee_core",
    "opensee_core.dms",
    "opensee_core.apicalls",
    "opensee_core.scripts",
    "opensee_core.data_pusher",
    "opensee_core.patches",
    "opensee_core.opensee_apis",
)


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_importing_datagenic_pulls_no_platform_or_cloud():
    code = (
        "import datagenic, sys; "
        f"print(any(m in sys.modules for m in {_PLATFORM + ('boto3', 'paramiko')!r}))"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_generation_modules_import_no_platform():
    """Importing the generation leaf (incl. gen-side orchestration) pulls in no
    platform module."""
    code = (
        "import datagenic.engine, datagenic.scaffolding, datagenic.generators, "
        "datagenic.orchestration, postprocessors_udfs.generic_udfs, sys; "
        f"print(any(m in sys.modules for m in {_PLATFORM!r}))"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_gen_side_orchestration_stayed_in_datagenic():
    """The pure-generation work-split stayed in datagenic.orchestration."""
    from datagenic import orchestration

    for name in ("shard_batch_range", "run_shard", "run_parallel", "check_distributable"):
        assert hasattr(orchestration, name)
    assert not hasattr(orchestration, "ChunkedRun")
