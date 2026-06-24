"""Phase 3 — partial table injection into column generators.

A column generator receives ``_table`` holding the columns generated *before* it
in the same table, enabling correlated columns. Column order is load-bearing.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator


def _cfg(out, columns):
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
                "table_name": "t",
                "number_of_rows": 30,
                "post_processor": {},
                "columns": columns,
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _derived(source):
    return {
        "col_name": "derived",
        "col_type": "string",
        "is_unicity_key": False,
        "column_generator": {
            "udf_name": "DerivedColumn",
            "udf_module": "postprocessors_udfs.generic_udfs",
            "args": {"source": source},
        },
    }


def _one_partition(out):
    files = list((Path(out) / "t").rglob("*.parquet"))
    assert len(files) == 1
    return pd.read_parquet(files[0])


def test_generator_reads_earlier_column(tmp_path):
    """A generator derives its values from a column defined earlier (via _table)."""
    out = tmp_path / "out"
    cols = [
        gf._col("t", "id", "int", True, 1, 30),
        gf._col("t", "base", "string", False, 1, 30),
        _derived("base"),  # 'base' is generated before 'derived'
    ]
    Generator(data_generator_config_file=_cfg(out, cols)).generate()

    df = _one_partition(out)
    assert (df["derived"] == df["base"] + "_d").all()
    assert "NO_SOURCE" not in set(df["derived"])


def test_column_order_is_load_bearing(tmp_path):
    """Referencing a column defined LATER yields the fallback (not yet in _table)."""
    out = tmp_path / "out"
    cols = [
        gf._col("t", "id", "int", True, 1, 30),
        _derived("base"),  # 'derived' comes BEFORE 'base'
        gf._col("t", "base", "string", False, 1, 30),
    ]
    Generator(data_generator_config_file=_cfg(out, cols)).generate()

    df = _one_partition(out)
    assert (df["derived"] == "NO_SOURCE").all()


def test_first_column_gets_empty_partial_table(tmp_path):
    """The first column's generator sees an empty _table (no earlier columns)."""
    out = tmp_path / "out"
    # 'derived' is the FIRST column and references nothing available yet
    cols = [
        _derived("base"),
        gf._col("t", "base", "string", False, 1, 30),
    ]
    # 'derived' is also the (only sensible) unicity key here; mark it so the table is valid
    cols[0]["is_unicity_key"] = True
    Generator(data_generator_config_file=_cfg(out, cols)).generate()

    df = _one_partition(out)
    assert (df["derived"] == "NO_SOURCE").all()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
