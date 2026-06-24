"""Phase 3 — _result access guard for ungenerated tables.

Referencing a table before it is generated raises a clear error instead of
silently handing over an empty DataFrame.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator
from datagenic.dtypes import _Ungenerated

_PROBE = "postprocessors_udfs.generic_udfs"


def test_sentinel_raises_on_access():
    s = _Ungenerated("scenario_types")
    with pytest.raises(RuntimeError, match="scenario_types"):
        len(s)
    with pytest.raises(RuntimeError, match="scenario_types"):
        s["col"]
    with pytest.raises(RuntimeError, match="scenario_types"):
        _ = s.columns


def test_reference_to_later_table_raises(tmp_path):
    """A postprocessor referencing a table defined later hits the guard at runtime."""
    cfg = {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(tmp_path / "out"),
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
                "table_name": "tableA",
                "number_of_rows": 10,
                "post_processor": {
                    "udf_module": _PROBE,
                    "udf_name": "RecordingPostProcessor",
                    "udf_args": {"_input_tables": ["tableB"]},  # tableB is defined AFTER tableA
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": False,
            },
            {
                "table_name": "tableB",
                "number_of_rows": 10,
                "post_processor": {},
                "columns": [gf._col("tableB", "id", "int", True, 1, 10)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }

    with pytest.raises(RuntimeError) as ei:
        Generator(data_generator_config_file=cfg).generate()
    msg = str(ei.value)
    assert "tableB" in msg
    assert "before it was generated" in msg


def test_full_run_triggers_no_guard(tmp_path):
    """A correctly-ordered config generates without ever tripping the guard."""
    cfg = gf._rng_config()
    cfg["default_parameters"]["output_folder_path"] = str(tmp_path / "out")
    Generator(data_generator_config_file=cfg).generate()  # no RuntimeError


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
