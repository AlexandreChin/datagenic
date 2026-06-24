"""Scalability Step 3 (Part 1) — streaming row-group parquet writer.

`storages.dump_dataframe(..., stream=True, row_group_rows=N)` writes a Polars
frame to local parquet one row group at a time via `pyarrow.parquet.ParquetWriter`,
instead of materialising one Arrow table for the whole frame. The default
(`stream=False`) is unchanged and byte-identical, so G1 stays green.
"""

import polars as pl
import pyarrow.parquet as pq

from datagenic.iokit import storages


def _frame(n_rows):
    return pl.DataFrame(
        {
            "id": list(range(n_rows)),
            "amount": [float(i) * 1.5 for i in range(n_rows)],
            "label": [f"row-{i}" for i in range(n_rows)],
        }
    )


def test_streamed_equals_materialized(tmp_path):
    """One row group streamed == the non-streamed write, byte-for-byte."""
    df = _frame(5_000)
    storage = storages.Storages("local")

    materialized = tmp_path / "materialized.parquet"
    streamed = tmp_path / "streamed.parquet"
    storage.dump_dataframe(df, str(materialized))
    storage.dump_dataframe(df, str(streamed), stream=True)  # row_group_rows=None -> 1 group

    assert streamed.read_bytes() == materialized.read_bytes()


def test_streamed_equals_materialized_content_when_chunked(tmp_path):
    """Chunked row groups change layout but not content (re-read is identical)."""
    df = _frame(5_000)
    storage = storages.Storages("local")

    materialized = tmp_path / "materialized.parquet"
    streamed = tmp_path / "streamed.parquet"
    storage.dump_dataframe(df, str(materialized))
    storage.dump_dataframe(df, str(streamed), stream=True, row_group_rows=1_000)

    assert pl.read_parquet(streamed).equals(pl.read_parquet(materialized))


def test_streamed_write_is_chunked(tmp_path):
    """row_group_rows < n produces more than one row group."""
    df = _frame(5_000)
    out = tmp_path / "streamed.parquet"
    storages.Storages("local").dump_dataframe(df, str(out), stream=True, row_group_rows=1_000)

    md = pq.read_metadata(out)
    assert md.num_row_groups == 5
    assert md.num_rows == 5_000


def test_empty_frame_streams_to_valid_parquet(tmp_path):
    """A zero-row frame still writes a readable parquet with the right schema."""
    df = _frame(0)
    out = tmp_path / "empty.parquet"
    storages.Storages("local").dump_dataframe(df, str(out), stream=True, row_group_rows=1_000)

    back = pl.read_parquet(out)
    assert back.height == 0
    assert back.columns == ["id", "amount", "label"]


def test_arrow_conversion_bounded_by_row_group(tmp_path, monkeypatch):
    """Streaming converts at most one row group to Arrow at a time.

    This is the writer's actual memory guarantee, asserted structurally (max rows
    handed to the Arrow conversion) rather than via process RSS: an eager input
    frame stays resident, and polars→arrow is largely zero-copy for primitives, so
    a whole-process RSS drop only appears once the input itself is produced
    incrementally (Step 4 / Step 8). Here we prove the bound at the writer.
    """
    df = _frame(400_000)
    storage = storages.Storages("local")

    original = storages.Storages._polars_to_arrow_table
    max_rows_converted = []

    def _spy(frame, metadata):
        max_rows_converted.append(frame.height)
        return original(frame, metadata)

    monkeypatch.setattr(storages.Storages, "_polars_to_arrow_table", staticmethod(_spy))

    # Materialised: the whole frame is converted in one Arrow table.
    storage.dump_dataframe(df, str(tmp_path / "materialized.parquet"))
    assert max(max_rows_converted) == 400_000

    # Streamed: never more than row_group_rows converted at once.
    max_rows_converted.clear()
    storage.dump_dataframe(
        df, str(tmp_path / "streamed.parquet"), stream=True, row_group_rows=20_000
    )
    assert max(max_rows_converted) == 20_000
