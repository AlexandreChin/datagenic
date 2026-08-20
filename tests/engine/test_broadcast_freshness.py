"""A date-varying broadcast table must stay fresh on every batch.

``single_batch`` tables are generated once (on the first batch) and broadcast to
the others, so the shared ``_result`` store used to be left holding the LAST
date's frame when the first batch's date loop ended — and every later batch, at
every date, was fed that stale frame through ``_input_tables`` / ``_table``.

The engine now snapshots each broadcast table per date on the first batch and
replays those snapshots on the following ones, so a consumer sees the same data
on batch N as it did on batch 1.
"""

from pathlib import Path

import polars as pl

from datagenic import Generator

DATES = ["2024-01-01", "2024-01-02", "2024-01-03"]


def _cfg(out, *, batches=2):
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": DATES[0],
            "end_date": DATES[-1],
            "number_of_batches": batches,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "dim",
                "number_of_rows": 2,
                "columns": [],
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "DateStampingPostProcessor",
                    "udf_args": {},
                },
                "single_batch": True,  # broadcast...
                "single_date": False,  # ...but its content varies by date
                "dump_results": False,
            },
            {
                "table_name": "fact",
                "number_of_rows": 1,
                "columns": [],
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "DateEchoPostProcessor",
                    "udf_args": {"_input_tables": ["dim"]},
                },
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def _echoes(out):
    frames = [pl.read_parquet(p) for p in sorted(Path(out).rglob("*.parquet"))]
    return pl.concat(frames).sort("batch", "date")


def test_broadcast_table_is_fresh_on_every_batch(tmp_path):
    Generator(_cfg(tmp_path)).generate(force_regenerate=True)
    echoes = _echoes(tmp_path)

    assert len(echoes) == 2 * len(DATES)
    stale = echoes.filter(pl.col("date") != pl.col("seen"))
    assert stale.is_empty(), f"consumer fed a stale broadcast frame:\n{stale}"


def test_later_batches_see_what_the_first_batch_saw(tmp_path):
    """Batch-to-batch equality is the invariant: a broadcast table is the same
    data for every batch, date by date."""
    Generator(_cfg(tmp_path, batches=3)).generate(force_regenerate=True)
    echoes = _echoes(tmp_path)

    per_batch = {
        b[0]: dict(zip(g["date"].to_list(), g["seen"].to_list(), strict=True))
        for b, g in echoes.group_by("batch")
    }
    assert sorted(per_batch) == [1, 2, 3]
    first = per_batch[1]
    assert all(seen == first for seen in per_batch.values())


def test_frozen_broadcast_table_replays_from_a_single_snapshot(tmp_path):
    """A single_date broadcast table keeps the same frame on every date, so the
    cache holds one entry and the later dates simply carry it forward — batch 2
    must still see start_date's data, not a gap."""
    cfg = _cfg(tmp_path)
    cfg["tables"][0]["single_date"] = True  # frozen at start_date
    Generator(cfg).generate(force_regenerate=True)

    echoes = _echoes(tmp_path)
    assert echoes["seen"].to_list() == [DATES[0]] * (2 * len(DATES))


def test_only_broadcast_tables_someone_reads_are_snapshotted():
    """The cache covers broadcast tables a postprocessor or a dynamic row count
    actually reads — not fact tables, and not broadcast tables nobody looks at."""
    tables = [
        {"table_name": "read_dim", "single_batch": True, "post_processor": {}},
        {"table_name": "ignored_dim", "single_batch": True, "post_processor": {}},
        {"table_name": "sized_dim", "single_batch": True, "post_processor": {}},
        {"table_name": "bulk", "single_batch": False, "post_processor": {}},
        {
            "table_name": "fact",
            "single_batch": False,
            "number_of_rows": {"from_table": "sized_dim", "multiplier": 2},
            "post_processor": {"udf_args": {"_input_tables": ["read_dim", "bulk"]}},
        },
    ]
    assert Generator._broadcast_consumed(tables) == {"read_dim", "sized_dim"}
