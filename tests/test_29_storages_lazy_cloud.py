"""Package split Part 2 — lazy cloud backends in storages (future `iokit`).

Importing `storages` (and therefore `datagenic`) must NOT pull in any cloud SDK
(boto3 / paramiko / azure). The backends are imported lazily per `storage_type`,
so local parquet/csv works with zero cloud dependencies — and a stripped env
(no boto3/paramiko) can still import and run generation.

Checks run in a clean subprocess so they reflect a fresh interpreter, not state
left by other tests. (Moves to `test_29_iokit_*` when Part 2b carves the package.)
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_CLOUD_SDKS = ("boto3", "paramiko", "azure")


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )


def test_importing_storages_pulls_no_cloud_sdk():
    code = f"import datagenic.iokit.storages, sys; print(any(m in sys.modules for m in {_CLOUD_SDKS!r}))"
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_importing_engine_pulls_no_cloud_sdk():
    """The original friction: importing the generator dragged in boto3 eagerly."""
    code = f"import datagenic.engine, sys; print(any(m in sys.modules for m in {_CLOUD_SDKS!r}))"
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_local_roundtrip_imports_no_cloud_sdk(tmp_path):
    """A full local write+read works and still imports no cloud SDK."""
    out = tmp_path / "x.parquet"
    code = f"""
import sys, polars as pl
from datagenic.iokit import storages
s = storages.Storages("local")
s.dump_dataframe(pl.DataFrame({{"id": [1, 2, 3]}}), {str(out)!r})
assert s.read_parquet_as_polars({str(out)!r}).height == 3
print(any(m in sys.modules for m in {_CLOUD_SDKS!r}))
"""
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
