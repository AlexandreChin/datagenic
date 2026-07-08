"""Output contract for the platform reclaim (generate → consume → reclaim).

`opensee_core`'s ChunkedRun reclaims each generated chunk between ingests. To delete
the right files on local *or* cloud — without reconstructing datagenic's output layout
in the (separate) opensee-core repo — it relies on three public seams owned here:

  - ``Generator.resolved_config`` — the resolved, read-only config (no private access).
  - ``Generator.batch_output_paths(table, batch)`` — the objects written for one batch,
    enumerated through the storage layer (so it works for local and every cloud backend).
  - ``Storages.delete(path)`` — storage-agnostic single-object delete.

This test pins that contract, so a change to the output layout / naming breaks here,
in datagenic's own suite, instead of silently breaking reclaim in opensee-core.
"""

from pathlib import Path

from datagenic import Configs, Generator
from datagenic.iokit.storages import Storages

IR = {
    "name": "shop",
    "tables": [
        {
            "type": "fact",
            "name": "orders",
            "keys": ["order_id"],
            "columns": [
                {"name": "order_id", "type": "int"},
                {"name": "amount", "type": "float"},
            ],
        }
    ],
}


def _config(out, *, batches=3):
    return Configs(
        input=IR,
        storage_type="local",
        output_format="parquet",
        output_folder_path=str(out),
        number_of_rows_in_fact_tables=30,
        number_of_batches=batches,
        start_date="2024-01-01",
        end_date="2024-01-02",
        seed=0,
    ).config_template


# --------------------------------------------------------------------------- #
# resolved_config
# --------------------------------------------------------------------------- #
def test_resolved_config_is_public_and_isolated(tmp_path):
    """`resolved_config` exposes the source config and hands back a fresh copy each
    time, so a caller mutating it cannot corrupt the generator's state."""
    gen = Generator(_config(tmp_path / "out"))
    cfg = gen.resolved_config
    assert "default_parameters" in cfg and "tables" in cfg

    cfg["default_parameters"]["number_of_batches"] = 999
    assert gen.resolved_config["default_parameters"]["number_of_batches"] == 3


# --------------------------------------------------------------------------- #
# batch_output_paths + delete
# --------------------------------------------------------------------------- #
def test_batch_output_paths_finds_a_batchs_files(tmp_path):
    out = tmp_path / "out"
    gen = Generator(_config(out, batches=3))
    gen.generate()

    paths = gen.batch_output_paths("orders", 1)
    assert paths, "expected batch 1 to have written files"
    # every returned path is this table+batch's parquet, and actually exists
    for p in paths:
        assert Path(p).name == "orders-1.parquet"
        assert Path(p).exists()
    # the layout spans dates: 2 business days -> 2 files for the batch
    assert len(paths) == 2


def test_batch_output_paths_empty_when_nothing_written(tmp_path):
    gen = Generator(_config(tmp_path / "out"))
    # no generate() call -> the table dir does not exist -> [], not an error
    assert gen.batch_output_paths("orders", 1) == []


def test_delete_removes_only_that_file(tmp_path):
    out = tmp_path / "out"
    gen = Generator(_config(out, batches=3))
    gen.generate()
    storage = Storages("local")

    batch1 = gen.batch_output_paths("orders", 1)
    batch2 = gen.batch_output_paths("orders", 2)
    assert batch1 and batch2

    for p in batch1:
        storage.delete(p)

    assert all(not Path(p).exists() for p in batch1), "batch 1 should be reclaimed"
    assert all(Path(p).exists() for p in batch2), "batch 2 must be untouched"


def test_delete_missing_is_noop(tmp_path):
    Storages("local").delete(str(tmp_path / "does_not_exist.parquet"))  # must not raise


def test_reclaim_loop_matches_chunkedrun_usage(tmp_path):
    """Exactly how ChunkedRun reclaims: for each batch in a chunk, list the bulk
    table's files via the contract and delete them through the storage layer."""
    out = tmp_path / "out"
    gen = Generator(_config(out, batches=4))
    gen.generate()
    storage = Storages("local")

    # reclaim the first chunk (batches 1-2); keep batches 3-4
    for batch in (1, 2):
        for path in gen.batch_output_paths("orders", batch):
            storage.delete(path)

    assert gen.batch_output_paths("orders", 1) == []
    assert gen.batch_output_paths("orders", 2) == []
    assert gen.batch_output_paths("orders", 3)
    assert gen.batch_output_paths("orders", 4)
