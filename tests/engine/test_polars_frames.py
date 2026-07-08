"""Phase 2, Step 0 — Polars as the default internal format.

Asserts the framework now builds column-generated tables as Polars DataFrames and
hands Polars to postprocessors, and that the Polars dedup/sort produces the same
data as the previous pandas drop_duplicates/sort_values (the proof that Step 0
changed only the internal format/serialization, not the data).
"""

import golden_fixtures as gf  # noqa: E402
import numpy as np
import pandas as pd
import polars as pl

from datagenic import Generator
from datagenic.dtypes import _as_pl_series
from postprocessors_udfs import generic_udfs


def test_dedup_sort_parity_pandas_vs_polars():
    """Polars unique(keep=first)+sort equals pandas drop_duplicates+sort_values."""
    rng = np.random.default_rng(0)
    n = 500
    data = {
        "k1": rng.integers(0, 20, n),
        "k2": rng.integers(0, 20, n),
        "v": rng.normal(size=n),
    }
    keys = ["k1", "k2"]

    pdf = pd.DataFrame(data).drop_duplicates(subset=keys, ignore_index=True)
    pdf = pdf.sort_values(by=keys, ignore_index=True)

    pldf = pl.DataFrame(data).unique(subset=keys, keep="first", maintain_order=True).sort(keys)

    # same surviving (k1,k2) rows in the same order, same associated value
    assert pldf.to_pandas().reset_index(drop=True).equals(pdf)


def test_postprocessor_receives_polars():
    """_table and referenced _tables handed to a postprocessor are Polars frames."""
    generic_udfs.RECORDS.clear()
    cfg = _routing_config()
    import tempfile

    with tempfile.TemporaryDirectory() as out:
        cfg["default_parameters"]["output_folder_path"] = out
        Generator(data_generator_config_file=cfg).generate()

    first = generic_udfs.RECORDS[0]
    assert first["table_mod"].startswith("polars"), first["table_mod"]
    assert first["tables_mods"]["t_rng"].startswith("polars"), first["tables_mods"]


def test_generator_series_have_expected_dtypes():
    """Wrapping generator output as pl.Series yields the expected Polars dtypes."""
    rows = 10
    specs = [
        ("id", "int", True, pl.Int64),
        ("s", "string", False, pl.String),
        ("f", "float", False, pl.Float64),
        ("d", "date", False, pl.Datetime),
    ]
    for name, col_type, is_key, expected in specs:
        col = gf._col("t", name, col_type, is_key, 1, rows)
        gen = getattr(
            __import__("datagenic.generators", fromlist=["x"]), col["column_generator"]["udf_name"]
        )()
        args = {
            "_date": "2024-01-01",
            "_previous_date": "2024-01-01",
            "_start_date": "2024-01-01",
            "_end_date": "2024-01-01",
            "_batch_number": 1,
            "_number_of_batches": 1,
            "_number_of_rows_per_batch": rows,
            **col["column_generator"]["args"],
            "cardinality": rows,
        }
        values = gen.call(**args)
        s = _as_pl_series(name, values)
        assert s.dtype == expected or s.dtype.base_type() == expected, (
            f"{name}: got {s.dtype}, expected {expected}"
        )
        assert len(s) == rows


# --------------------------------------------------------------------------- #
# Step 1 — flexible postprocessor return types
# --------------------------------------------------------------------------- #
def _dict_intermediate_config(out, dump_results):
    """tableA returns a dict; a probe table reads it via _tables."""
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
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
                "table_name": "intermediate",
                "number_of_rows": 10,
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "DictReturning",
                    "udf_args": {},
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": dump_results,
            },
            {
                "table_name": "reader",
                "number_of_rows": 10,
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "RecordingPostProcessor",
                    "udf_args": {"_input_tables": ["intermediate"]},
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": False,
            },
        ],
    }


def test_dump_results_false_passes_dict_downstream(tmp_path):
    """A dict from a dump_results:false postprocessor reaches the next via _tables."""
    generic_udfs.RECORDS.clear()
    out = tmp_path / "out"
    Generator(
        data_generator_config_file=_dict_intermediate_config(out, dump_results=False)
    ).generate()

    # nothing written for the intermediate (dump_results:false)
    assert not (out / "intermediate").exists()
    # the reader saw the dict in _tables (a builtins type, not a DataFrame)
    rec = generic_udfs.RECORDS[0]
    assert rec["tables_mods"]["intermediate"] == "builtins"


def test_dump_results_true_nondataframe_raises(tmp_path):
    """A non-DataFrame return with dump_results:true raises TypeError at write."""
    import pytest

    out = tmp_path / "out"
    with pytest.raises(TypeError):
        Generator(
            data_generator_config_file=_dict_intermediate_config(out, dump_results=True)
        ).generate()


def _routing_config():
    rows = 50
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": "",
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
                "table_name": "t_rng",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    gf._col("t_rng", "id", "int", True, 1, rows),
                    gf._col("t_rng", "val", "float", False, 1, rows),
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
                    gf._col("probe", "id", "int", True, 1, rows),
                    gf._col("probe", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
