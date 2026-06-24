"""Package split Part 4 — the datagenic CLI is generation-only.

`datagenic` CLI = generation only (generate-data / generate-data-config /
check-distributable). It lists none of the platform (`opensee-core`) subcommands
and runs with no platform dependency. (The platform CLI and the cross-CLI
separation are tested in the opensee-core suite.)

Invoked via `python -m datagenic` (the console entry point `datagenic` is
registered in pyproject for installs).
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_CMDS = ("generate-data", "generate-data-config", "check-distributable")
PLATFORM_CMDS = ("dms", "run-script", "push-data", "patch")


def _help(module):
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    ).stdout


def test_datagenic_cli_lists_only_generation_commands():
    out = _help("datagenic")
    assert all(c in out for c in GEN_CMDS)
    assert not any(c in out for c in PLATFORM_CMDS)


def test_datagenic_cli_generate_data_config_runs(tmp_path):
    """The generation CLI scaffolds a config end-to-end from the native sample IR."""
    out = tmp_path / "config.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datagenic",
            "generate-data-config",
            "samples/ir/solutions_ir.json",
            "-o",
            str(out),
            "--output-folder",
            str(tmp_path / "data"),
            "--rows-fact",
            "50",
            "--rows-dict",
            "20",
            "--batches",
            "1",
            "--seed",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
