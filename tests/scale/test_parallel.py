"""Scalability Step 8 (Part 5) — parallel execution (multiprocessing).

``run_parallel(config, workers)`` splits the batch space into ``workers`` disjoint
slices (the Step-6 partition) and generates each in its own process. Per-unit
seeded RNG + partitioned output mean no coordination is needed, so parallel output
is byte-identical to sequential. ``workers=1`` is sequential (today's behaviour).
"""

import hashlib
from pathlib import Path

import golden_fixtures as gf  # noqa: E402
import numpy as np

from datagenic import Generator, orchestration


def _cfg(out, *, batches=8, seed=0):
    rows = 80
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "number_of_batches": batches,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": seed,
        },
        "tables": [
            {
                "table_name": "fact",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [gf._col("fact", "id", "int", True, 1, rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "dim",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [gf._col("dim", "id", "int", True, 1, rows)],
                "single_batch": True,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def _hash_tree(out):
    out = Path(out)
    return {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*.parquet"))
    }


def _sequential(out):
    Generator(data_generator_config_file=_cfg(out, batches=8)).generate()
    return _hash_tree(out)


def test_parallel_equals_sequential(tmp_path):
    """Parallel output is byte-identical to a sequential run (seeded-RNG payoff)."""
    sequential = _sequential(tmp_path / "seq")

    parallel_out = tmp_path / "par"
    orchestration.run_parallel(_cfg(parallel_out, batches=8), workers=4)

    assert _hash_tree(parallel_out) == sequential


def test_workers_1_equals_sequential(tmp_path):
    """workers=1 reproduces the plain sequential run exactly."""
    sequential = _sequential(tmp_path / "seq")

    one = tmp_path / "one"
    orchestration.run_parallel(_cfg(one, batches=8), workers=1)

    assert _hash_tree(one) == sequential


def test_retry_is_idempotent(tmp_path):
    """Re-running reproduces identical bytes (seeded + checkpoint)."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    orchestration.run_parallel(_cfg(first, batches=8), workers=4)
    orchestration.run_parallel(_cfg(second, batches=8), workers=4)

    assert _hash_tree(first) == _hash_tree(second)


def test_no_shared_state_leak(tmp_path):
    """Workers don't depend on the global np.random: perturbing it in the parent
    does not change the output (generation uses the injected _rng)."""
    baseline = tmp_path / "baseline"
    orchestration.run_parallel(_cfg(baseline, batches=8), workers=4)

    np.random.seed(424242)
    np.random.random(10_000)
    perturbed = tmp_path / "perturbed"
    orchestration.run_parallel(_cfg(perturbed, batches=8), workers=4)

    assert _hash_tree(perturbed) == _hash_tree(baseline)
