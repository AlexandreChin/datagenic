"""Scalability Step 5 (Part 3) — uniqueness by construction, drop the global sort.

The per-unit ``df.unique(subset).sort(keys)`` bundles two ops:
- ``unique`` — a per-unit (hash) dedup of composite key tuples; it does NOT span
  units, and disjoint per-batch key ranges (``_cardinality_range`` offset) already
  keep batches/shards unique *by construction*.
- ``sort`` — a whole-frame O(n log n) ordering that does not scale and blocks
  sub-unit row-group chunking (you cannot sort across chunks).

``default_parameters.scale_skip_global_key_sort = True`` drops the global sort
while keeping per-unit dedup. Row *order* changes (re-baseline territory), but the
set of rows and key uniqueness are identical — proven here before any re-capture.
Default is False, so G1 stays byte-identical.
"""

from pathlib import Path

import golden_fixtures as gf  # noqa: E402
import polars as pl

from datagenic import Generator


def _composite_key_cfg(out, *, batches=4, skip_sort=False, by_construction=False, seed=0):
    """A fact-like table with a COMPOSITE unicity key (k1, k2) — the case the
    global unique()/sort() exists for (single keys are already incremental)."""
    rows = 200
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "number_of_batches": batches,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": seed,
            "scale_skip_global_key_sort": skip_sort,
            "scale_unique_by_construction": by_construction,
        },
        "tables": [
            {
                "table_name": "fact",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    gf._col("fact", "k1", "int", True, 2, rows),
                    gf._col("fact", "k2", "int", True, 2, rows),
                    gf._col("fact", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _date_key_cfg(out, *, batches=2, by_construction=True, seed=0):
    """Composite key (Date, k1) spanning multiple dates. Date is a unicity key
    (CurrentAsofDates), so tuples must stay unique across dates by construction."""
    rows = 120
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "number_of_batches": batches,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": seed,
            "scale_unique_by_construction": by_construction,
        },
        "tables": [
            {
                "table_name": "fact",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    gf._col("fact", "Date", "date", True, 2, rows),
                    gf._col("fact", "k1", "int", True, 2, rows),
                    gf._col("fact", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _postprocessor_cfg(out):
    """A postprocessor-only table that emits duplicate (k1, k2) tuples, with the
    scale uniqueness flag ON — the framework must NOT touch its output."""
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
            "scale_unique_by_construction": True,
            "scale_skip_global_key_sort": True,
        },
        "tables": [
            {
                "table_name": "fact",
                "number_of_rows": 3,
                "columns": [],
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "DuplicateKeyPostProcessor",
                    "udf_args": {},
                },
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _read_all(out, table="fact"):
    files = sorted((Path(out) / table).rglob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()


def _run(cfg):
    out = cfg["default_parameters"]["output_folder_path"]
    Generator(data_generator_config_file=cfg).generate()
    return _read_all(out)


def test_keys_globally_unique_without_global_sort(tmp_path):
    """Union of all batch/shard outputs has no duplicate key tuples — with the
    global sort disabled. Disjoint per-batch ranges guarantee it by construction."""
    df = _run(_composite_key_cfg(tmp_path / "out", batches=4, skip_sort=True))

    assert df.height > 0
    keys = df.select(["k1", "k2"])
    assert keys.n_unique() == keys.height  # no duplicate (k1, k2) across all shards


def test_value_distribution_unchanged(tmp_path):
    """Dropping the global sort changes row order only: identical set of rows and
    identical per-key value coverage vs the sorted (current) path."""
    sorted_df = _run(_composite_key_cfg(tmp_path / "sorted", batches=4, skip_sort=False))
    unsorted_df = _run(_composite_key_cfg(tmp_path / "unsorted", batches=4, skip_sort=True))

    assert sorted_df.height == unsorted_df.height
    # Same multiset of rows regardless of order -> sort both sides and compare.
    cols = sorted_df.columns
    assert sorted_df.sort(cols).equals(unsorted_df.sort(cols))


def test_global_sort_actually_dropped(tmp_path):
    """The skip flag changes on-disk row order (otherwise it would be a no-op)."""
    sorted_df = _run(_composite_key_cfg(tmp_path / "sorted", batches=4, skip_sort=False))
    unsorted_df = _run(_composite_key_cfg(tmp_path / "unsorted", batches=4, skip_sort=True))

    # Same rows, but the sorted path is sorted by (k1, k2) and the skip path is not
    # (generation order), so the row sequences differ.
    assert not sorted_df.equals(unsorted_df)
    assert sorted_df.equals(sorted_df.sort(["k1", "k2"]))


def test_default_keeps_sort(tmp_path):
    """Default (flag absent) preserves the current sorted behaviour -> G1 safe."""
    cfg = _composite_key_cfg(tmp_path / "out", batches=4)
    cfg["default_parameters"].pop("scale_skip_global_key_sort")
    cfg["default_parameters"].pop("scale_unique_by_construction")
    df = _run(cfg)

    assert df.equals(df.sort(["k1", "k2"]))


# --- uniqueness by construction (eliminates the per-unit unique()) --------- #
def test_by_construction_keys_unique_without_dedup(tmp_path):
    """Composite keys are unique across all shards with no unique() pass at all."""
    df = _run(_composite_key_cfg(tmp_path / "out", batches=4, by_construction=True))

    assert df.height > 0
    keys = df.select(["k1", "k2"])
    assert keys.n_unique() == keys.height


def test_by_construction_values_within_generated_domain(tmp_path):
    """Enumeration only uses values the generators actually produced (no fabricated
    values), and stays disjoint per batch so the union is unique."""
    random_df = _run(_composite_key_cfg(tmp_path / "random", batches=4))
    constructed_df = _run(
        _composite_key_cfg(tmp_path / "constructed", batches=4, by_construction=True)
    )

    for key in ("k1", "k2"):
        produced = set(random_df[key].to_list())
        used = set(constructed_df[key].to_list())
        assert used <= produced


def test_by_construction_is_positional_and_deterministic(tmp_path):
    """Key assignment is positional and reproducible: same seed -> identical key
    tuples. This is the property a row-chunk needs *given the key domains*; making
    domains derivable without the full column (true chunk-independence) is the
    remaining generator-contract work (Step 8)."""
    df = _run(_composite_key_cfg(tmp_path / "out", batches=4, by_construction=True))

    again = _run(_composite_key_cfg(tmp_path / "again", batches=4, by_construction=True))
    assert (
        df.sort(["k1", "k2"])
        .select(["k1", "k2"])
        .equals(again.sort(["k1", "k2"]).select(["k1", "k2"]))
    )


# --- cross-date guarantee -------------------------------------------------- #
def test_by_construction_unique_across_dates_with_date_key(tmp_path):
    """With a Date column in the composite key, tuples are unique across BOTH
    dates and batches. (Without a date component the offset alone would not
    separate dates — that is the documented limit; a Date key is the usual case.)"""
    df = _run(_date_key_cfg(tmp_path / "out", batches=2))

    # more than one date actually generated
    assert df["Date"].n_unique() > 1
    keys = df.select(["Date", "k1"])
    assert keys.n_unique() == keys.height  # globally unique across dates x batches


# --- postprocessor bypass -------------------------------------------------- #
def test_postprocessor_output_bypasses_uniqueness(tmp_path):
    """Uniqueness machinery lives in the column stage only: a postprocessor's
    output is written as-is, so its intentional duplicate keys survive even with
    scale_unique_by_construction + scale_skip_global_key_sort enabled."""
    df = _run(_postprocessor_cfg(tmp_path / "out"))

    keys = df.select(["k1", "k2"])
    assert keys.height == 3
    assert keys.n_unique() == 2  # the (1, 9) duplicate was NOT removed
