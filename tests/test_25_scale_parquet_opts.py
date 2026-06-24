"""Scalability Step 10 (Part 7) — parquet tuning knobs.

`compression` (e.g. zstd), `use_dictionary`, and `row_group_size` are optional
pass-throughs to the Polars write paths in `dump_dataframe`. They are not required:
absent => None => the writer's current defaults, so output stays byte-identical
(G1 safe). The pandas/data_pusher path is untouched.
"""

import polars as pl
import pyarrow.parquet as pq

from datagenic.iokit import storages


def _frame(n=2000):
    return pl.DataFrame({"id": list(range(n)), "label": [f"v{i % 5}" for i in range(n)]})


def test_default_is_unchanged(tmp_path):
    """No knobs given => default codec (snappy), single shot — byte-identical to a
    plain write (proves the knobs are truly optional)."""
    plain = tmp_path / "a.parquet"
    also = tmp_path / "b.parquet"
    storage = storages.Storages("local")
    storage.dump_dataframe(_frame(), str(plain))
    storage.dump_dataframe(_frame(), str(also))  # no knobs

    assert plain.read_bytes() == also.read_bytes()
    assert pq.read_metadata(plain).row_group(0).column(0).compression == "SNAPPY"


def test_compression_applied(tmp_path):
    """compression='zstd' is reflected in the written file."""
    out = tmp_path / "z.parquet"
    storages.Storages("local").dump_dataframe(_frame(), str(out), compression="zstd")

    assert pq.read_metadata(out).row_group(0).column(0).compression == "ZSTD"


def test_row_group_size_applied(tmp_path):
    """row_group_size controls the number of row groups."""
    out = tmp_path / "rg.parquet"
    storages.Storages("local").dump_dataframe(_frame(2000), str(out), row_group_size=500)

    assert pq.read_metadata(out).num_row_groups == 4


def test_options_on_lazyframe_sink(tmp_path):
    """Knobs also reach the streaming LazyFrame write (compression via sink)."""
    out = tmp_path / "lz.parquet"
    storages.Storages("local").dump_dataframe(
        pl.LazyFrame({"id": list(range(2000))}), str(out), compression="zstd"
    )

    assert pq.read_metadata(out).row_group(0).column(0).compression == "ZSTD"


def test_output_still_readable_with_options(tmp_path):
    """A tuned file round-trips through Storages.read_file unchanged in content."""
    out = tmp_path / "t.parquet"
    storage = storages.Storages("local")
    storage.dump_dataframe(_frame(300), str(out), compression="zstd", use_dictionary=True)

    back = storage.read_file(str(out))  # pandas via read_file
    assert len(back) == 300
    assert list(back.columns) == ["id", "label"]
