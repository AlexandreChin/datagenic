"""Phase 0 — Safety net.

These tests are the regression guard every later refactoring phase relies on
(see refactoring_plan/00_overview.md, "Shared verification harness"):

  G1 — Golden byte-diff : regenerating at the same seed/config is byte-identical.
  G2 — Determinism      : same seed -> identical; different seed -> different.
  Idempotency           : two generators built from the same in-memory config
                          produce identical output (guards the config-immutability fix).
  Immutability          : generate() does not mutate the caller's config dict.

The primary fixture is a tiny, hand-built column-generator config: it exercises
the RNG-driven generators (so seed isolation is meaningful) with a single integer
unicity key (avoiding the unrelated composite-key cardinality bug in the sample
DMS path) and no external dependencies. A second fixture uses the committed
post-processor config to guard the in-place ``udf_args["tables"]`` mutation.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from datagenic import Columns, Generator
from postprocessors_udfs import generic_udfs

_GOLDEN_DIR = Path(__file__).parent / "golden"
sys.path.insert(0, str(_GOLDEN_DIR))
import golden_fixtures  # noqa: E402

POSTPROC_CONFIG = "samples/data_generation_config/data_generator_config_post_processor.json"


def _col(table, name, col_type, is_key, n_keys, rows):
    c = Columns(
        table,
        name,
        col_type,
        is_key,
        n_keys,
        number_of_rows=rows,
        _number_of_rows_per_batch=rows,
    )
    return {
        "col_name": name,
        "col_type": col_type,
        "is_unicity_key": is_key,
        "column_generator": {
            "udf_name": c.default_generator_name,
            "udf_module": "datagenic.generators",
            "args": c.default_arguments,
            "description": c.default_generator_description,
        },
    }


def _rng_config():
    """A tiny self-contained, RNG-driven column-generator config (as a dict)."""
    rows = 200
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": "",
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "number_of_batches": 2,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
            "scale": 0.5,  # exercises the scale path _apply_scale used to pop in place
        },
        "tables": [
            {
                "table_name": "t_rng",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    _col("t_rng", "id", "int", True, 1, rows),
                    _col("t_rng", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _run(cfg, out_dir):
    cfg = copy.deepcopy(cfg)
    cfg["default_parameters"]["output_folder_path"] = str(out_dir)
    Generator(data_generator_config_file=cfg).generate()


def _hash_tree(root):
    """Map every parquet under root to its sha256 (relative path -> digest)."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*.parquet")):
        out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# --------------------------------------------------------------------------- #
# G1 / G2 — determinism
# --------------------------------------------------------------------------- #
def test_same_seed_is_byte_identical(tmp_path):
    """G1/G2: two runs at the same seed produce byte-identical parquet files."""
    a, b = tmp_path / "a", tmp_path / "b"
    _run(_rng_config(), a)
    _run(_rng_config(), b)

    ha, hb = _hash_tree(a), _hash_tree(b)
    assert ha, "no parquet files were produced"
    assert ha == hb, "same seed must yield byte-identical output"


def test_different_seed_changes_output(tmp_path):
    """G2: a different seed must actually change the generated data."""
    a, b = tmp_path / "a", tmp_path / "b"
    cfg_b = _rng_config()
    cfg_b["default_parameters"]["seed"] = 12345
    _run(_rng_config(), a)
    _run(cfg_b, b)

    ha, hb = _hash_tree(a), _hash_tree(b)
    assert set(ha) == set(hb), "the set of output files should not depend on the seed"
    assert ha != hb, "a different seed must change at least one file's contents"


# --------------------------------------------------------------------------- #
# Config immutability + idempotency
# --------------------------------------------------------------------------- #
def test_generate_does_not_mutate_caller_config(tmp_path):
    """The caller's config dict is byte-identical before and after generate()."""
    cfg = _rng_config()
    cfg["default_parameters"]["output_folder_path"] = str(tmp_path / "out")
    before = json.dumps(cfg, sort_keys=True)

    Generator(data_generator_config_file=cfg).generate()

    after = json.dumps(cfg, sort_keys=True)
    assert before == after, "generate() must not mutate the caller's config dict"


def test_postprocessor_config_not_mutated(tmp_path):
    """The notorious udf_args['tables'] union must not leak into the caller's dict."""
    with open(POSTPROC_CONFIG) as f:
        cfg = json.load(f)
    cfg["default_parameters"]["output_folder_path"] = str(tmp_path / "out")
    before = json.dumps(cfg, sort_keys=True)

    Generator(data_generator_config_file=cfg).generate()

    after = json.dumps(cfg, sort_keys=True)
    assert before == after, "generate() mutated the caller's post-processor config"


def test_two_generators_same_dict_are_identical(tmp_path):
    """Two generators built from the same dict object produce identical output.

    Before the immutability fix this failed: the first run popped `scale`, so
    the second generator saw scale=1.0 and produced different row counts.
    """
    cfg = _rng_config()
    out = tmp_path / "out"
    cfg["default_parameters"]["output_folder_path"] = str(out)

    Generator(data_generator_config_file=cfg).generate()
    h1 = _hash_tree(out)

    # a second generator built from the SAME (still-pristine) dict object
    Generator(data_generator_config_file=cfg).generate()
    h2 = _hash_tree(out)

    assert h1, "no parquet files were produced"
    assert h1 == h2


def test_repeated_generate_same_object_is_identical(tmp_path):
    """Calling generate() twice on the same Generator yields identical output."""
    cfg = _rng_config()
    out = tmp_path / "out"
    cfg["default_parameters"]["output_folder_path"] = str(out)
    gen = Generator(data_generator_config_file=cfg)

    gen.generate()
    h1 = _hash_tree(out)
    gen.generate()
    h2 = _hash_tree(out)

    assert h1, "no parquet files were produced"
    assert h1 == h2


# --------------------------------------------------------------------------- #
# Framework routing & data handoff (instrumented via postprocessors.safety_net_probe)
# --------------------------------------------------------------------------- #
def _routing_config(single_date=False):
    """A source table (t_rng) plus a probe table that references it.

    1 batch, 3 business days -> date1 uses call(), later dates use recall().
    """
    rows = 50
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": "",
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "t_rng",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    _col("t_rng", "id", "int", True, 1, rows),
                    _col("t_rng", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "probe",
                "number_of_rows": rows,
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "RecordingPostProcessor",
                    "udf_args": {"_input_tables": ["t_rng"]},
                },
                "columns": [
                    _col("probe", "id", "int", True, 1, rows),
                    _col("probe", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": single_date,
                "dump_results": True,
            },
        ],
    }


def test_call_then_recall_routing(tmp_path):
    """call() on the first date, recall() on every later date — never call() again."""
    generic_udfs.RECORDS.clear()
    _run(_routing_config(), tmp_path / "out")

    methods = [(r["method"], r["date"]) for r in generic_udfs.RECORDS]
    assert methods[0] == ("call", "2024-01-01")
    assert all(m == "recall" for m, _ in methods[1:])
    assert [d for _, d in methods] == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_single_date_invokes_call_once(tmp_path):
    """A single_date table invokes the postprocessor exactly once (date 1 only)."""
    generic_udfs.RECORDS.clear()
    out = tmp_path / "out"
    _run(_routing_config(single_date=True), out)

    assert [r["method"] for r in generic_udfs.RECORDS] == ["call"]
    # only the first date is written to disk
    date_dirs = sorted(p.name for p in (out / "probe").iterdir())
    assert date_dirs == ["2024-01-01"]


def test_table_arg_contains_generator_output(tmp_path):
    """The postprocessor's _table holds this table's generated columns, non-empty."""
    generic_udfs.RECORDS.clear()
    _run(_routing_config(), tmp_path / "out")

    first = generic_udfs.RECORDS[0]
    assert first["table_cols"] == ["id", "val"]
    assert first["table_len"] > 0


def test_tables_handoff_has_referenced_data(tmp_path):
    """_tables carries the referenced table's actual generated data (DAG ordering)."""
    generic_udfs.RECORDS.clear()
    _run(_routing_config(), tmp_path / "out")

    for r in generic_udfs.RECORDS:
        assert "t_rng" in r["tables_keys"]
        assert r["tables_lens"]["t_rng"] > 0


def test_postprocessor_gets_exactly_declared_tables(tmp_path):
    """_tables holds exactly udf_args.tables (no global injection) and no raw `tables` kwarg."""
    generic_udfs.RECORDS.clear()
    rows = 30
    cfg = {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(tmp_path / "out"),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "number_of_batches": 1,
            "offset_batches": 0,
            # a global tables list that must NOT be injected into the postprocessor
            "postprocessor_args": {"_input_tables": ["b"]},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "a",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [_col("a", "id", "int", True, 1, rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "b",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [_col("b", "id", "int", True, 1, rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "probe",
                "number_of_rows": rows,
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "RecordingPostProcessor",
                    "udf_args": {"_input_tables": ["a"]},  # declares ONLY 'a'
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": False,
            },
        ],
    }
    Generator(data_generator_config_file=cfg).generate()

    rec = generic_udfs.RECORDS[0]
    assert rec["tables_keys"] == ["a"], f"expected only 'a', got {rec['tables_keys']}"
    assert rec["raw_tables_kwarg"] is False, "raw 'tables' kwarg must not reach the postprocessor"


# --------------------------------------------------------------------------- #
# Output structure
# --------------------------------------------------------------------------- #
def test_dump_results_false_writes_no_parquet(tmp_path):
    """Tables with dump_results:false produce no files; true tables do."""
    with open(POSTPROC_CONFIG) as f:
        cfg = json.load(f)
    out = tmp_path / "out"
    cfg["default_parameters"]["output_folder_path"] = str(out)
    Generator(data_generator_config_file=cfg).generate()

    # table1 & table3 are dump_results:false; table2 & table4 are true
    assert not (out / "table1").exists()
    assert not (out / "table3").exists()
    assert (out / "table2").exists()
    assert (out / "table4").exists()


def test_parquet_path_and_batch_numbering(tmp_path):
    """Files live at {table}/{date}/{table}-{batch}.parquet with contiguous batches."""
    cfg = _rng_config()
    cfg["default_parameters"].pop("scale", None)
    cfg["default_parameters"]["number_of_batches"] = 3
    out = tmp_path / "out"
    _run(cfg, out)

    for date_dir in (out / "t_rng").iterdir():
        files = sorted(p.name for p in date_dir.glob("*.parquet"))
        # framework numbers batches 1..N (offset_batches + 1 .. + number_of_batches)
        assert files == [f"t_rng-{i}.parquet" for i in range(1, 4)]
    # nothing is written outside the {table}/{date} structure
    stray = [p for p in out.rglob("*.parquet") if len(p.relative_to(out).parts) != 3]
    assert not stray


def test_scale_application_halves_rows(tmp_path):
    """scale=0.5 produces ~half the rows of scale=1.0 (within 10%)."""

    def total_rows(out):
        return sum(len(pd.read_parquet(p)) for p in (out / "t_rng").rglob("*.parquet"))

    full = _rng_config()
    full["default_parameters"]["scale"] = 1.0
    half = _rng_config()
    half["default_parameters"]["scale"] = 0.5

    a, b = tmp_path / "full", tmp_path / "half"
    _run(full, a)
    _run(half, b)

    n_full, n_half = total_rows(a), total_rows(b)
    assert n_full > 0
    assert abs(n_half - n_full / 2) <= 0.1 * n_full


def test_zero_rows_does_not_crash(tmp_path):
    """A table whose row count is 0 generates without crashing (empty output)."""
    cfg = _rng_config()
    cfg["default_parameters"].pop("scale", None)
    cfg["tables"][0]["number_of_rows"] = 0
    out = tmp_path / "out"
    _run(cfg, out)

    files = list((out / "t_rng").rglob("*.parquet"))
    assert files, "an (empty) parquet should still be written"
    assert all(len(pd.read_parquet(p)) == 0 for p in files)


# --------------------------------------------------------------------------- #
# Regression guard — composite unicity keys (cardinality=None bug)
# --------------------------------------------------------------------------- #
def test_composite_unicity_keys_generate(tmp_path):
    """Composite string unicity keys (default cardinality None) must generate.

    Before the fix, `generate()` did `x / None` on these columns -> TypeError.
    The framework should auto-distribute their cardinality and produce unique
    key tuples.
    """
    rows = 60
    cfg = {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(tmp_path / "out"),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "composite",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    _col("composite", "k1", "string", True, 2, rows),
                    _col("composite", "k2", "string", True, 2, rows),
                    _col("composite", "v", "float", False, 2, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }
    # the two key columns carry the cardinality=None default that crashed
    assert cfg["tables"][0]["columns"][0]["column_generator"]["args"]["cardinality"] is None

    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "composite").rglob("*.parquet"))
    assert files
    for p in files:
        df = pd.read_parquet(p)
        assert len(df) > 0
        # within a table partition the framework dedups on the key set
        assert not df.duplicated(subset=["k1", "k2"]).any()


# --------------------------------------------------------------------------- #
# G1 — golden manifest (cross-commit byte-diff)
# --------------------------------------------------------------------------- #
def test_golden_manifest_matches(tmp_path):
    """Regenerating the reference fixtures matches the committed baseline manifest.

    Capture/refresh the baseline with `python tests/golden/capture_baseline.py`.
    Skipped until a baseline is committed.
    """
    manifest_path = _GOLDEN_DIR / "baseline_manifest.json"
    if not manifest_path.exists():
        pytest.skip("no golden baseline committed; run tests/golden/capture_baseline.py")

    expected = json.loads(manifest_path.read_text())
    for name, cfg in golden_fixtures.fixtures().items():
        out = tmp_path / name
        golden_fixtures.run(cfg, out)
        actual = golden_fixtures.hash_tree(out)
        assert actual == expected.get(name), (
            f"golden output for fixture '{name}' changed; if intentional, "
            f"re-run tests/golden/capture_baseline.py"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
