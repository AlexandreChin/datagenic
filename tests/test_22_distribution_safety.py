"""Scalability Step 7 (Part 10) — distribution-safety gate for postprocessors.

Sharding (Step 6) is correct only if every batch is independent. Column
generators are safe (seeded ``_rng``), but postprocessors are arbitrary user code.
This gate makes shardability declarable + checkable:
- static `validate(distributed=True)` — (a) reject `INPUT_SCOPE="global"` on a
  sharded table; (d) a sharded table's postprocessor may read reference
  (single_batch) tables only.
- runtime `check_distributable(config)` — adds (b) a determinism probe and (c) a
  sample sharded==single equivalence proof.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator, orchestration

UDF = "postprocessors_udfs.generic_udfs"


def _params(out, batches=2):
    return {
        "output_format": "parquet",
        "storage_type": "local",
        "output_folder_path": str(out),
        "tables_prefix": "",
        "start_date": "2024-01-01",
        "end_date": "2024-01-01",
        "number_of_batches": batches,
        "offset_batches": 0,
        "postprocessor_args": {},
        "seed": 0,
    }


def _fact(name, rows=40, *, post=None, single_batch=False):
    table = {
        "table_name": name,
        "number_of_rows": rows,
        "post_processor": post or {},
        "columns": [gf._col(name, "id", "int", True, 1, rows)],
        "single_batch": single_batch,
        "single_date": False,
        "dump_results": True,
    }
    return table


def _pp(udf_name, input_tables=None):
    return {
        "udf_module": UDF,
        "udf_name": udf_name,
        "udf_args": {"_input_tables": input_tables} if input_tables else {},
    }


# --- static gate (a) + (d) -------------------------------------------------- #
def test_global_scope_postprocessor_rejected(tmp_path):
    cfg = {
        "default_parameters": _params(tmp_path / "out"),
        "tables": [_fact("fact", post=_pp("GlobalScopePostProcessor"))],
    }
    with pytest.raises(ValueError, match=r"INPUT_SCOPE='global'"):
        Generator(data_generator_config_file=cfg).validate(distributed=True)


def test_bulk_reading_via_input_tables_allowed(tmp_path):
    """Reading another (even non-single_batch) table via _input_tables is batch-local
    — the framework injects only the current batch's slice — so the static gate must
    NOT reject it. Genuine cross-batch breakage is caught by check_distributable (c)."""
    cfg = {
        "default_parameters": _params(tmp_path / "out"),
        "tables": [
            _fact("dim_bulk", single_batch=False),
            _fact("fact", post=_pp("ReferenceReadingPostProcessor", input_tables=["dim_bulk"])),
        ],
    }
    # Should not raise — batch-local reads are shardable.
    Generator(data_generator_config_file=cfg).validate(distributed=True)


def test_reference_only_pp_allowed(tmp_path):
    # 'dim' IS single_batch (a reference pool) -> reading it is allowed.
    cfg = {
        "default_parameters": _params(tmp_path / "out"),
        "tables": [
            _fact("dim", single_batch=True),
            _fact("fact", post=_pp("ReferenceReadingPostProcessor", input_tables=["dim"])),
        ],
    }
    # Should not raise.
    Generator(data_generator_config_file=cfg).validate(distributed=True)


def test_default_validate_skips_gate(tmp_path):
    """validate() without distributed=True does not run the gate (generate() path)."""
    cfg = {
        "default_parameters": _params(tmp_path / "out"),
        "tables": [_fact("fact", post=_pp("GlobalScopePostProcessor"))],
    }
    Generator(data_generator_config_file=cfg).validate()  # no raise


# --- runtime probes (b) + (c) ---------------------------------------------- #
def test_safe_config_passes(tmp_path):
    """A reference-only, _rng-using config passes all gates and is sharded==single."""
    cfg = {
        "default_parameters": _params(tmp_path / "out", batches=4),
        "tables": [
            _fact("dim", single_batch=True),
            _fact("fact", post=_pp("ReferenceReadingPostProcessor", input_tables=["dim"])),
        ],
    }
    verdict = orchestration.check_distributable(cfg)
    assert verdict["ok"], verdict["failures"]


def test_nondeterministic_pp_detected(tmp_path):
    """A postprocessor using the global numpy RNG fails the determinism probe."""
    cfg = {
        "default_parameters": _params(tmp_path / "out", batches=2),
        "tables": [
            {
                "table_name": "fact",
                "number_of_rows": 3,
                "columns": [],
                "post_processor": _pp("NonDeterministicPostProcessor"),
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }
    verdict = orchestration.check_distributable(cfg)
    assert not verdict["ok"]
    assert any("determinism probe" in f for f in verdict["failures"])
