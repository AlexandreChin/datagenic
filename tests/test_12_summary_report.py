"""Phase 3 / file 07 — post-generation summary report.

`generation_summary.json` is derived from the journal: per-table rows/units/
errors + overall timing. Written incrementally so a crash leaves a valid
partial summary.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator

_PROBE = "postprocessors_udfs.generic_udfs"


def _params(out):
    return {
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
    }


def _summary(out):
    return json.loads((Path(out) / "generation_summary.json").read_text())


def _two_table_cfg(out):
    return {
        "default_parameters": _params(out),
        "tables": [
            {
                "table_name": "a",
                "number_of_rows": 10,
                "post_processor": {},
                "columns": [gf._col("a", "id", "int", True, 1, 10)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "b",
                "number_of_rows": 7,
                "post_processor": {},
                "columns": [gf._col("b", "id", "int", True, 1, 7)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def test_summary_has_entry_per_table(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_two_table_cfg(out)).generate()
    s = _summary(out)
    assert set(s["tables"]) == {"a", "b"}
    assert s["errors"] == []
    assert "started_at" in s and "completed_at" in s and "elapsed_s" in s


def test_total_rows_matches_parquet(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_two_table_cfg(out)).generate()
    s = _summary(out)
    for name in ("a", "b"):
        actual = sum(len(pd.read_parquet(p)) for p in (out / name).rglob("*.parquet"))
        assert s["tables"][name]["total_rows"] == actual


def test_error_table_listed_in_errors(tmp_path):
    out = tmp_path / "out"
    cfg = {
        "default_parameters": _params(out),
        "tables": [
            {
                "table_name": "boom",
                "number_of_rows": 5,
                "post_processor": {
                    "udf_module": _PROBE,
                    "udf_name": "BoomPostProcessor",
                    "udf_args": {},
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }
    with pytest.raises(RuntimeError):
        Generator(data_generator_config_file=cfg).generate()

    # partial summary still valid JSON, and records the error
    s = _summary(out)
    assert any(e["table"] == "boom" for e in s["errors"])
    assert s["tables"]["boom"]["errors"] >= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
