"""Scalability Step 2 (Part 0) — benchmark & memory harness.

Makes peak memory and throughput observable so later steps (streaming writer,
chunked generation) are *provable*. The heavy ``test_benchmark_runs`` is opt-in
(set ``RUN_SCALE_BENCH=1``) so CI can skip it; the lightweight harness/fixture
tests always run so the instrumentation itself stays covered.

Baseline (to beat in later steps) — current materialise-whole-table path,
recorded via ``RUN_SCALE_BENCH=1 .venv/bin/python -m pytest -s`` on the dev box
(Python 3.14, pyarrow 23.0.1, polars 1.39.3):

    fact: 250000 rows x 1 batch x 1 date -> peak_rss ≈ 298 MB, wall ≈ 1.9 s

Step 3 (streaming writer) / Step 4 (chunked gen) must push peak_rss well below
this row-count-proportional figure (re-run the benchmark there to compare).
"""

import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import scale_fixtures as sf  # noqa: E402

_BENCH_ROWS = 250_000


def _row_count(out_dir, table="fact"):
    files = sorted((Path(out_dir) / table).rglob("*.parquet"))
    return sum(pq.read_metadata(f).num_rows for f in files), files


def test_fixture_generates_requested_rows(tmp_path):
    """scale_fixtures builds a real, runnable config of the requested size."""
    cfg = sf.fact_table_config(tmp_path / "out", n_rows=500, n_batches=1, n_dates=1)
    sf.generate(cfg)

    total_rows, files = _row_count(tmp_path / "out")
    assert files, "no parquet produced"
    assert total_rows == 500


def test_peak_rss_harness_records(tmp_path):
    """The memory/timing harness reports a non-zero peak and wall time."""
    cfg = sf.fact_table_config(tmp_path / "out", n_rows=2_000, n_batches=1, n_dates=1)

    with sf.peak_rss() as m:
        sf.generate(cfg)

    assert m.peak_mb > 0
    assert m.elapsed_s > 0


@pytest.mark.skipif(
    not os.environ.get("RUN_SCALE_BENCH"),
    reason="set RUN_SCALE_BENCH=1 to run the slow scale benchmark",
)
def test_benchmark_runs(tmp_path):
    """Generate a medium fact table; record peak RSS + wall time as the baseline."""
    cfg = sf.fact_table_config(tmp_path / "out", n_rows=_BENCH_ROWS, n_batches=1, n_dates=1)

    with sf.peak_rss() as m:
        sf.generate(cfg)

    total_rows, files = _row_count(tmp_path / "out")
    assert total_rows == _BENCH_ROWS
    assert files

    # Surfaced with `pytest -s`; copy into the module docstring as the baseline.
    print(
        f"\n[scale-benchmark] fact: {_BENCH_ROWS} rows x 1 batch x 1 date "
        f"-> peak_rss ≈ {m.peak_mb:.1f} MB, wall ≈ {m.elapsed_s:.2f} s"
    )
