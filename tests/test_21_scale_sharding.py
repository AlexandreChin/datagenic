"""Scalability Step 6 (Part 4) — sharded, master-free distribution.

``run_shard(config, shard=i, of=N)`` generates a disjoint batch slice; the union
of all shards is byte-identical to one ``of=1`` run. ``of=1`` is today's behaviour
untouched (G1). The framework computes the batch range from ``--shard/--of`` so the
developer never does batch math, and the config file is never mutated.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator, orchestration


def _cfg(out, *, batches=8, seed=0):
    """A batched fact table + a single_batch reference (dim) table — the dim must
    appear once in the union (written by shard 0), like FK reference pools."""
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


def test_sharded_equals_single_run(tmp_path):
    """Union of all --shard i --of N outputs == one --of 1 run, byte-identical."""
    single = _hash_tree(_run_single(tmp_path / "single"))

    sharded_out = tmp_path / "sharded"
    n = 4
    for shard in range(1, n + 1):
        orchestration.run_shard(_cfg(sharded_out, batches=8), shard=shard, of=n)

    assert _hash_tree(sharded_out) == single


def _run_single(out):
    Generator(data_generator_config_file=_cfg(out, batches=8)).generate()
    return out


def test_of_1_unchanged(tmp_path):
    """--of 1 reproduces the plain Generator run exactly (no batch_range injected)."""
    plain = tmp_path / "plain"
    Generator(data_generator_config_file=_cfg(plain, batches=8)).generate()

    shard = tmp_path / "shard"
    orchestration.run_shard(_cfg(shard, batches=8), shard=1, of=1)

    assert _hash_tree(shard) == _hash_tree(plain)


def test_shards_disjoint_and_complete():
    """Shard ranges tile [0, B) exactly — every batch covered once, no gaps."""
    batches, n = 8, 3
    covered = []
    for shard in range(1, n + 1):  # shards are 1-based
        lo, hi = orchestration.shard_batch_range(batches, shard, n)
        covered.extend(range(lo, hi))

    assert sorted(covered) == list(range(batches))  # each 0..B-1 exactly once

    # First shard starts at 0, last shard (== of) clamps to B.
    assert orchestration.shard_batch_range(batches, 1, n)[0] == 0
    assert orchestration.shard_batch_range(batches, n, n)[1] == batches


def test_framework_computes_range_and_rejects_bad_args():
    """Developer passes shard/of, never batch indices; bad values error clearly."""
    assert orchestration.shard_batch_range(10, 1, 4) == [0, 3]  # ceil(0), ceil(10/4)

    with pytest.raises(ValueError, match=r"--shard must be in"):
        orchestration.shard_batch_range(10, 0, 4)  # 0 is below 1 (1-based)
    with pytest.raises(ValueError, match=r"--shard must be in"):
        orchestration.shard_batch_range(10, 5, 4)  # above of
    with pytest.raises(ValueError, match=r"--of must be >= 1"):
        orchestration.shard_batch_range(10, 1, 0)


def test_config_file_untouched(tmp_path):
    """Distributing never modifies the canonical config file."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(_cfg(tmp_path / "out", batches=8), indent=2))
    before = cfg_path.read_bytes()

    orchestration.run_shard(str(cfg_path), shard=1, of=4)

    assert cfg_path.read_bytes() == before
