"""A 'job' (generate(tables=...)) produces a root table plus its full transitive
upstream closure — and only that — byte-identically to a full run.
"""

import copy
import hashlib
from pathlib import Path

import pytest

from datagenic import Configs, Generator


def _ir():
    # orders (fact) -> FK to products (referential); regions (referential) is unrelated.
    return {
        "name": "shop",
        "tables": [
            {
                "type": "referential",
                "name": "products",
                "keys": ["sku"],
                "columns": [{"name": "sku", "type": "string"}, {"name": "price", "type": "float"}],
            },
            {
                "type": "referential",
                "name": "regions",
                "keys": ["rid"],
                "columns": [{"name": "rid", "type": "string"}],
            },
            {
                "type": "fact",
                "name": "orders",
                "keys": ["oid"],
                "columns": [{"name": "oid", "type": "int"}, {"name": "sku", "type": "string"}],
                "foreign_keys": [{"column": "sku", "references": "products.sku"}],
            },
        ],
    }


def _cfg(out):
    cfg = Configs(
        input=_ir(),
        output_format="parquet",
        number_of_batches=2,
        start_date="2024-01-01",
        end_date="2024-01-02",
        seed=0,
    ).config_template
    cfg = copy.deepcopy(cfg)
    cfg["default_parameters"]["output_folder_path"] = str(out)
    return cfg


def _table_dirs(out):
    return sorted(p.name for p in Path(out).iterdir() if p.is_dir())


def _file_hashes(out):
    root = Path(out)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.parquet"))
    }


def test_job_writes_root_plus_upstream_closure_only(tmp_path):
    out = tmp_path / "job"
    Generator(_cfg(out)).generate(tables=["orders"])
    # orders' closure = orders + products (its FK pool); the unrelated `regions` is excluded.
    assert _table_dirs(out) == ["orders", "products"]


def test_job_for_leaf_table_writes_only_itself(tmp_path):
    out = tmp_path / "job"
    Generator(_cfg(out)).generate(tables=["products"])
    assert _table_dirs(out) == ["products"]


def test_job_output_is_byte_identical_to_full_run(tmp_path):
    full = tmp_path / "full"
    Generator(_cfg(full)).generate()
    job = tmp_path / "job"
    Generator(_cfg(job)).generate(tables=["orders"])

    full_hashes = _file_hashes(full)
    job_hashes = _file_hashes(job)
    assert job_hashes, "job wrote nothing"
    # every file the job produced matches the full run's byte-for-byte
    for rel, digest in job_hashes.items():
        assert full_hashes.get(rel) == digest, f"{rel} differs from the full run"


def test_job_unknown_root_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown table"):
        Generator(_cfg(tmp_path / "x")).generate(tables=["does_not_exist"])
