"""G3 — data-correctness consistency suite.

Runs the full sample-DMS column-generator config once (the path unblocked by the
cardinality fix) and asserts correctness properties on the written parquet:

  - unicity      : key tuples are unique within each table partition
  - array sizes  : array columns have their configured length, cells non-null
  - nulls        : scalar generated columns contain no nulls
  - temporal     : the per-date `Date` column equals its partition date
  - volume       : per-partition row counts are positive and bounded by the batch size

Everything is derived from the generator config, so the suite works for any DMS,
not just `solutions_dms`.

Also covers foreign-key consistency: Configs now wires a KeyEnforcer onto fact
tables (enforce_fk=True), so fact-table FK columns draw from their referenced
tables' key pools. test_fk_columns_subset_of_reference_pools verifies this on the
full DMS fixture.
"""

import math
from pathlib import Path

import golden_fixtures as gf  # noqa: E402
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dms_run(tmp_path_factory):
    """Generate the full sample-DMS dataset once for the whole module."""
    out = tmp_path_factory.mktemp("dms_consistency")
    cfg = gf._dms_config()
    gf.run(cfg, out)
    return cfg, out


def _tables(cfg):
    return cfg["tables"]


def _partitions(out, table_name):
    """Yield (date_str, parquet_path) for every written partition of a table."""
    base = Path(out) / table_name
    for date_dir in sorted(base.iterdir()):
        for parquet in sorted(date_dir.glob("*.parquet")):
            yield date_dir.name, parquet


def _key_cols(table):
    return [c["col_name"] for c in table["columns"] if c["is_unicity_key"]]


def _array_cols(table):
    return [c for c in table["columns"] if "array" in c["col_type"].lower()]


def _scalar_cols(table):
    return [c["col_name"] for c in table["columns"] if "array" not in c["col_type"].lower()]


def test_dataset_is_non_empty(dms_run):
    """Sanity: every table produced at least one partition with rows."""
    cfg, out = dms_run
    for table in _tables(cfg):
        files = list((Path(out) / table["table_name"]).rglob("*.parquet"))
        assert files, f"{table['table_name']}: no parquet written"


def test_unicity_keys_unique_per_partition(dms_run):
    """Within each partition, the unicity-key tuple has no duplicates."""
    cfg, out = dms_run
    for table in _tables(cfg):
        keys = _key_cols(table)
        if not keys:
            continue
        for date, parquet in _partitions(out, table["table_name"]):
            df = pd.read_parquet(parquet)
            dups = df.duplicated(subset=keys).sum()
            assert dups == 0, (
                f"{table['table_name']} {date} {parquet.name}: {dups} duplicate rows on keys {keys}"
            )


def test_array_columns_have_configured_size(dms_run):
    """Array columns hold a list/array of exactly size_of_arrays, never null."""
    cfg, out = dms_run
    for table in _tables(cfg):
        arr_cols = _array_cols(table)
        if not arr_cols:
            continue
        for date, parquet in _partitions(out, table["table_name"]):
            df = pd.read_parquet(parquet)
            for col in arr_cols:
                name = col["col_name"]
                expected = col["column_generator"]["args"]["size_of_arrays"]
                lengths = df[name].map(lambda v: len(v) if v is not None else -1)
                assert (lengths == expected).all(), (
                    f"{table['table_name']} {date} col '{name}': "
                    f"expected every array length {expected}, got {sorted(set(lengths))}"
                )


def test_scalar_columns_have_no_nulls(dms_run):
    """Generated scalar columns are complete (generators do not emit nulls)."""
    cfg, out = dms_run
    for table in _tables(cfg):
        cols = _scalar_cols(table)
        for date, parquet in _partitions(out, table["table_name"]):
            df = pd.read_parquet(parquet)
            for name in cols:
                assert df[name].notna().all(), (
                    f"{table['table_name']} {date} col '{name}': contains nulls"
                )


def test_date_column_matches_partition(dms_run):
    """A table's `Date` column equals the partition date on every row."""
    cfg, out = dms_run
    checked = 0
    for table in _tables(cfg):
        if not any(c["col_name"] == "Date" for c in table["columns"]):
            continue
        for date, parquet in _partitions(out, table["table_name"]):
            df = pd.read_parquet(parquet)
            got = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d").unique()
            assert list(got) == [date], (
                f"{table['table_name']} {parquet.name}: Date column {list(got)} != partition {date}"
            )
            checked += 1
    assert checked > 0, "expected at least one table with a Date column"


def test_fk_columns_subset_of_reference_pools(dms_run):
    """Every FK column enforced by a KeyEnforcer draws only from its reference pool."""
    cfg, out = dms_run

    def _col_values(table_name, col):
        vals = set()
        for _, parquet in _partitions(out, table_name):
            vals |= set(pd.read_parquet(parquet)[col].tolist())
        return vals

    checked = 0
    for table in _tables(cfg):
        pp = table.get("post_processor") or {}
        if pp.get("udf_name") != "KeyEnforcer":
            continue
        for rule in pp["udf_args"]["rules"]:
            refs = rule["from"] if isinstance(rule["from"], list) else [rule["from"]]
            pool = set()
            for ref in refs:
                ref_table, ref_col = ref.rsplit(".", 1)
                pool |= _col_values(ref_table, ref_col)
            fk_vals = _col_values(table["table_name"], rule["col"])
            assert fk_vals <= pool, (
                f"{table['table_name']}.{rule['col']} has values outside "
                f"{rule['from']}: {sorted(fk_vals - pool)[:5]}"
            )
            checked += 1
    assert checked > 0, "expected at least one FK-enforced column in the DMS fixture"


def test_volume_within_bounds(dms_run):
    """Per-partition row counts are positive and bounded by the per-batch size."""
    cfg, out = dms_run
    n_batches = cfg["default_parameters"]["number_of_batches"]
    for table in _tables(cfg):
        nr = table["number_of_rows"]
        if not nr:
            continue
        per_batch = math.ceil(nr / n_batches)
        total = 0
        for date, parquet in _partitions(out, table["table_name"]):
            n = len(pd.read_parquet(parquet))
            total += n
            assert 0 < n <= per_batch, (
                f"{table['table_name']} {date} {parquet.name}: {n} rows outside (0, {per_batch}]"
            )
        assert total > 0, f"{table['table_name']}: produced no rows"


# --------------------------------------------------------------------------- #
# FK consistency — KeyEnforcer (datagenic.postprocessors)
# --------------------------------------------------------------------------- #
def _fk_config(out):
    """A reference dict + a fact table whose `fk` column is FK-enforced to it."""
    rows_dict, rows_fact = 20, 80
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "number_of_batches": 2,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "ref_dict",
                "number_of_rows": rows_dict,
                "post_processor": {},
                "columns": [gf._col("ref_dict", "dkey", "string", True, 1, rows_dict)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "fact",
                "number_of_rows": rows_fact,
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "KeyEnforcer",
                    "udf_args": {
                        "_input_tables": ["ref_dict"],
                        "rules": [{"col": "fk", "from": "ref_dict.dkey"}],
                    },
                },
                "columns": [
                    gf._col("fact", "id", "int", True, 1, rows_fact),
                    gf._col("fact", "fk", "string", False, 1, rows_fact),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def test_fk_enforcement_constrains_to_reference_pool(tmp_path):
    """Every enforced FK value exists in the referenced table's key pool.

    Exercises datagenic.postprocessors.KeyEnforcer (full Polars) end to end, including
    the eager-Polars write path in storages.dump_dataframe. Without enforcement
    the `fk` column would hold random "fk_*" values absent from the pool.
    """
    from datagenic import Generator

    out = tmp_path / "out"
    Generator(data_generator_config_file=_fk_config(out)).generate()

    pool = set()
    for _, parquet in _partitions(out, "ref_dict"):
        pool |= set(pd.read_parquet(parquet)["dkey"].tolist())
    assert pool, "reference pool is empty"

    fk_values = set()
    for _, parquet in _partitions(out, "fact"):
        fk_values |= set(pd.read_parquet(parquet)["fk"].tolist())

    assert fk_values, "no fk values produced"
    assert fk_values <= pool, f"FK values outside reference pool: {sorted(fk_values - pool)}"


# --------------------------------------------------------------------------- #
# Dynamic row count — number_of_rows: {from_table, multiplier}
# --------------------------------------------------------------------------- #
def _dynamic_rows_config(out, multiplier):
    rows = 20
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "ref",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [gf._col("ref", "rkey", "string", True, 1, rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "dyn",
                "number_of_rows": {"from_table": "ref", "multiplier": multiplier},
                "post_processor": {},
                "columns": [gf._col("dyn", "id", "int", True, 1, 2 * rows)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def test_dynamic_row_count_from_table(tmp_path):
    """A table sized as {from_table, multiplier} gets round(len(ref) * multiplier) rows."""
    from datagenic import Generator

    for mult, expected in [(1.0, 20), (2.0, 40), (0.5, 10)]:
        out = tmp_path / f"m_{mult}"
        Generator(data_generator_config_file=_dynamic_rows_config(out, mult)).generate()

        ref_rows = sum(len(pd.read_parquet(p)) for _, p in _partitions(out, "ref"))
        dyn_rows = sum(len(pd.read_parquet(p)) for _, p in _partitions(out, "dyn"))
        assert ref_rows == 20
        assert dyn_rows == expected, f"multiplier {mult}: expected {expected}, got {dyn_rows}"


# --------------------------------------------------------------------------- #
# Cardinality semantics — total distinct count, independent of batches
# --------------------------------------------------------------------------- #
def _base_params(out, n_batches):
    return {
        "output_format": "parquet",
        "storage_type": "local",
        "output_folder_path": str(out),
        "tables_prefix": "",
        "start_date": "2024-01-01",
        "end_date": "2024-01-01",
        "number_of_batches": n_batches,
        "offset_batches": 0,
        "postprocessor_args": {},
        "seed": 0,
    }


def test_non_key_cardinality_is_total_across_batches(tmp_path):
    """A non-key column's `cardinality` is the TOTAL distinct count (not x n_batches)."""
    from datagenic import Generator

    C, N = 6, 4
    cat = gf._col("t", "cat", "string", False, 1, 400)
    cat["column_generator"]["args"]["cardinality"] = C
    cfg = {
        "default_parameters": _base_params(tmp_path / "out", N),
        "tables": [
            {
                "table_name": "t",
                "number_of_rows": 400,
                "post_processor": {},
                "columns": [gf._col("t", "id", "int", True, 1, 400), cat],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }
    Generator(data_generator_config_file=cfg).generate()

    distinct = set()
    for _, parquet in _partitions(tmp_path / "out", "t"):
        distinct |= set(pd.read_parquet(parquet)["cat"].tolist())
    # fixed pool of C values shared across all N batches — not C * N
    assert len(distinct) <= C, f"expected <= {C} distinct, got {len(distinct)}"
    assert len(distinct) == C, f"expected exactly {C} distinct, got {len(distinct)}"


def test_composite_key_unique_across_batches(tmp_path):
    """Composite unicity keys stay globally unique across batches (disjoint ranges)."""
    from datagenic import Generator

    rows, N = 40, 2
    cfg = {
        "default_parameters": _base_params(tmp_path / "out", N),
        "tables": [
            {
                "table_name": "ck",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    gf._col("ck", "k1", "string", True, 2, rows),
                    gf._col("ck", "k2", "string", True, 2, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }
    Generator(data_generator_config_file=cfg).generate()

    frames = [pd.read_parquet(p) for _, p in _partitions(tmp_path / "out", "ck")]
    allrows = pd.concat(frames, ignore_index=True)
    assert len(frames) == N
    assert not allrows.duplicated(subset=["k1", "k2"]).any(), (
        "composite keys collided across batches"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
