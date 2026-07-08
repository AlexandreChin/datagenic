"""Phase 3 / file 07 — table-level checkpointing + config-hash metadata.

Existing parquet files carry a config-hash in their metadata; a re-run reuses a
file when (and only when) its hash matches, so a crashed run resumes instead of
restarting. `force_regenerate=True` overrides reuse.
"""

import hashlib
import os
from pathlib import Path

import golden_fixtures as gf  # noqa: E402
import pyarrow.parquet as pq

from datagenic import Generator


def _cfg(out, seed=0, rows=40, batches=2):
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
                "table_name": "t",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [gf._col("t", "id", "int", True, 1, rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _files(out):
    return sorted((Path(out) / "t").rglob("*.parquet"))


def _hash_tree(out):
    return {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _files(out)
    }


def test_parquet_has_config_hash_metadata(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_cfg(out)).generate()
    meta = pq.read_metadata(str(_files(out)[0])).metadata or {}
    assert b"datagenic_config_hash" in meta
    assert meta[b"datagenic_seed"] == b"0"


def test_config_hash_is_content_based(tmp_path):
    g = Generator(data_generator_config_file=_cfg(tmp_path / "a"))
    # different output location -> SAME hash (location is not content)
    g2 = Generator(data_generator_config_file=_cfg(tmp_path / "b"))
    assert g._config_hash() == g2._config_hash()
    # different content (rows) -> different hash
    g3 = Generator(data_generator_config_file=_cfg(tmp_path / "a", rows=99))
    assert g._config_hash() != g3._config_hash()


def test_checkpoint_skips_regeneration(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_cfg(out)).generate()
    old = 1_000_000.0  # back-date every file
    for p in _files(out):
        os.utime(p, (old, old))

    Generator(data_generator_config_file=_cfg(out)).generate()  # default: reuse
    assert all(os.path.getmtime(p) == old for p in _files(out)), "files were rewritten, not reused"

    Generator(data_generator_config_file=_cfg(out)).generate(force_regenerate=True)
    assert all(os.path.getmtime(p) > old for p in _files(out)), "force_regenerate did not rewrite"


def test_changed_seed_invalidates_checkpoint(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_cfg(out, seed=0)).generate()
    before = _hash_tree(out)
    # same dir, different seed -> stored hash mismatches -> regenerate (different data)
    Generator(data_generator_config_file=_cfg(out, seed=123)).generate()
    after = _hash_tree(out)
    assert before != after


def test_crash_restart_matches_clean_run(tmp_path):
    clean = tmp_path / "clean"
    Generator(data_generator_config_file=_cfg(clean)).generate()
    clean_hashes = _hash_tree(clean)

    crashed = tmp_path / "crashed"
    Generator(data_generator_config_file=_cfg(crashed)).generate()
    # simulate a crash that lost some outputs
    for p in _files(crashed)[:2]:
        p.unlink()
    # restart: missing files regenerated, existing reused
    Generator(data_generator_config_file=_cfg(crashed)).generate()

    assert _hash_tree(crashed) == clean_hashes


def test_resume_equals_full_through_postprocessor_chain(tmp_path):
    """Idempotency: full run == (partial run + resume), even for a table downstream
    of a checkpointed one through pandas postprocessors.

    On resume the upstream (checkpointed) table is reloaded as Polars and
    round-tripped through parquet, yet the regenerated downstream output is
    byte-identical to the clean run — guaranteed by per-unit deterministic seeding
    and explicit RNG (skipping a unit cannot perturb any other).
    """
    import json

    with open("samples/data_generation_config/data_generator_config_post_processor.json") as f:
        cfg0 = json.load(f)

    full = tmp_path / "full"
    c = json.loads(json.dumps(cfg0))
    c["default_parameters"]["output_folder_path"] = str(full)
    Generator(data_generator_config_file=c).generate()
    full_hashes = {
        str(p.relative_to(full)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(full.rglob("*.parquet"))
    }

    res = tmp_path / "res"
    c2 = json.loads(json.dumps(cfg0))
    c2["default_parameters"]["output_folder_path"] = str(res)
    Generator(data_generator_config_file=c2).generate()
    # simulate a crash that lost the downstream table (table4), keeping table2
    for p in [p for p in res.rglob("*.parquet") if f"{os.sep}table4{os.sep}" in str(p)]:
        p.unlink()
    Generator(data_generator_config_file=c2).generate()  # resume
    res_hashes = {
        str(p.relative_to(res)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(res.rglob("*.parquet"))
    }

    assert res_hashes == full_hashes


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
