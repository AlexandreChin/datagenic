"""Scalability Step 11 (Part 8, minimal) — streaming LazyFrame dump.

The framework's only type responsibility is to dump any of the 3 supported types
correctly: pandas DataFrame, Polars DataFrame, Polars LazyFrame. A LazyFrame is
streamed to disk via ``sink_parquet`` — the framework never ``.collect()``s it (a
post-processor that produced it owns any materialisation). Eager paths are
unchanged, so G1 stays byte-identical.
"""

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from datagenic.iokit import storages


def _frame_dict(n):
    return {"id": list(range(n)), "v": [float(i) * 1.5 for i in range(n)]}


def test_lazyframe_streamed_to_disk(tmp_path):
    """A LazyFrame is written correctly (valid parquet, all rows)."""
    out = tmp_path / "lazy.parquet"
    storages.Storages("local").dump_dataframe(pl.LazyFrame(_frame_dict(500)), str(out))

    back = pl.read_parquet(out)
    assert back.height == 500
    assert back.columns == ["id", "v"]


def test_lazy_equals_eager_content(tmp_path):
    """Lazy and eager writes of the same data produce identical content."""
    data = _frame_dict(500)
    eager = tmp_path / "eager.parquet"
    lazy = tmp_path / "lazy.parquet"
    storage = storages.Storages("local")
    storage.dump_dataframe(pl.DataFrame(data), str(eager))
    storage.dump_dataframe(pl.LazyFrame(data), str(lazy))

    assert pl.read_parquet(lazy).equals(pl.read_parquet(eager))


def test_lazy_write_embeds_checkpoint_metadata(tmp_path):
    """Streamed LazyFrame still carries the config-hash metadata (checkpointing)."""
    out = tmp_path / "lazy.parquet"
    meta = {"datagenic_config_hash": "deadbeef", "datagenic_seed": "0"}
    storages.Storages("local").dump_dataframe(
        pl.LazyFrame(_frame_dict(10)), str(out), metadata=meta
    )

    kv = pq.read_metadata(out).metadata
    assert kv.get(b"datagenic_config_hash") == b"deadbeef"
    assert kv.get(b"datagenic_seed") == b"0"


def test_framework_streams_lazyframe_via_sink(tmp_path, monkeypatch):
    """The framework routes a LazyFrame to sink_parquet (streaming) rather than
    materialising it through the eager Arrow path. (Whether Polars collects
    internally for an in-memory frame is its own concern; the framework's contract
    is: use the streaming sink, don't build the whole Arrow table.)"""
    sink_calls = []
    arrow_calls = []
    original_sink = pl.LazyFrame.sink_parquet
    original_to_arrow = storages.Storages._polars_to_arrow_table

    def _sink_spy(self, *a, **k):
        sink_calls.append(True)
        return original_sink(self, *a, **k)

    def _arrow_spy(df, metadata):
        arrow_calls.append(True)
        return original_to_arrow(df, metadata)

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", _sink_spy)
    monkeypatch.setattr(storages.Storages, "_polars_to_arrow_table", staticmethod(_arrow_spy))
    storages.Storages("local").dump_dataframe(
        pl.LazyFrame(_frame_dict(50)), str(tmp_path / "lazy.parquet")
    )

    assert sink_calls == [True]  # streamed via sink
    assert arrow_calls == []  # never went through the eager materialise path


def test_three_types_all_dump(tmp_path):
    """pandas / Polars DataFrame / Polars LazyFrame all write readable parquet."""
    storage = storages.Storages("local")
    data = _frame_dict(20)
    paths = {
        "pandas": tmp_path / "pandas.parquet",
        "polars": tmp_path / "polars.parquet",
        "lazy": tmp_path / "lazy.parquet",
    }
    storage.dump_dataframe(pd.DataFrame(data), str(paths["pandas"]))
    storage.dump_dataframe(pl.DataFrame(data), str(paths["polars"]))
    storage.dump_dataframe(pl.LazyFrame(data), str(paths["lazy"]))

    for path in paths.values():
        assert pl.read_parquet(path).height == 20
