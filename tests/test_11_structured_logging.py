"""Phase 3 / file 07 — structured logging (progress journal).

A JSON line is written before (START) and after (DONE) each table/date/batch
unit, flushed immediately; a failing unit gets START + ERROR (no DONE). The log
is appended across runs.
"""

import json
import sys
from pathlib import Path

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


def _read_log(out):
    lines = (Path(out) / "generation.log").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _ok_cfg(out):
    return {
        "default_parameters": _params(out),
        "tables": [
            {
                "table_name": "t",
                "number_of_rows": 10,
                "post_processor": {},
                "columns": [gf._col("t", "id", "int", True, 1, 10)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _boom_cfg(out):
    return {
        "default_parameters": _params(out),
        "tables": [
            {
                "table_name": "t",
                "number_of_rows": 10,
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


def test_success_writes_start_and_done(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_ok_cfg(out)).generate()

    recs = _read_log(out)
    events = [(r["event"], r["table"]) for r in recs]
    assert ("START", "t") in events
    assert ("DONE", "t") in events
    done = next(r for r in recs if r["event"] == "DONE")
    assert done["rows"] == 10 and done["date"] == "2024-01-01" and done["batch"] == 1


def test_failure_writes_start_and_error_no_done(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(RuntimeError):
        Generator(data_generator_config_file=_boom_cfg(out)).generate()

    recs = _read_log(out)
    events = {r["event"] for r in recs if r["table"] == "t"}
    assert "START" in events
    assert "ERROR" in events
    assert "DONE" not in events
    err = next(r for r in recs if r["event"] == "ERROR")
    assert "boom-pp" in err["error"]


def test_runs_append_to_the_log(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_ok_cfg(out)).generate()
    Generator(data_generator_config_file=_ok_cfg(out)).generate()

    recs = _read_log(out)
    starts = [r for r in recs if r["event"] == "START" and r["table"] == "t"]
    assert len(starts) == 2  # both runs recorded, not overwritten


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
