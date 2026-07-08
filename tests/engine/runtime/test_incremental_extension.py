"""Phase 3 / file 07 — incremental date-range extension.

Extending end_date reuses existing per-date files (date-partitioned tables) and
generates only the new dates, while single_date tables regenerate (they span the
whole range). `extend=True` guards against treating a fresh run as an extension.
"""

import os
from pathlib import Path

import golden_fixtures as gf  # noqa: E402
import pytest

from datagenic import Generator


def _cfg(out, end):
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": end,
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "dp",  # date-partitioned
                "number_of_rows": 20,
                "post_processor": {},
                "columns": [gf._col("dp", "id", "int", True, 1, 20)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "sd",  # single_date (spans the whole range)
                "number_of_rows": 20,
                "post_processor": {},
                "columns": [gf._col("sd", "id", "int", True, 1, 20)],
                "single_batch": False,
                "single_date": True,
                "dump_results": True,
            },
        ],
    }


def test_extend_reuses_old_dates_and_regenerates_single_date(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_cfg(out, "2024-01-02")).generate()

    old = 1_000_000.0
    for p in Path(out).rglob("*.parquet"):
        os.utime(p, (old, old))

    # extend the range by one business day
    Generator(data_generator_config_file=_cfg(out, "2024-01-03")).generate(extend=True)

    # date-partitioned: old dates reused, new date generated
    assert os.path.getmtime(out / "dp" / "2024-01-01" / "dp-1.parquet") == old
    assert os.path.getmtime(out / "dp" / "2024-01-02" / "dp-1.parquet") == old
    assert (out / "dp" / "2024-01-03" / "dp-1.parquet").exists()
    assert os.path.getmtime(out / "dp" / "2024-01-03" / "dp-1.parquet") > old

    # single_date: regenerated because end_date changed (still only the first date)
    assert os.path.getmtime(out / "sd" / "2024-01-01" / "sd-1.parquet") > old
    assert sorted(p.name for p in (out / "sd").iterdir()) == ["2024-01-01"]


def test_extend_on_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Generator(data_generator_config_file=_cfg(tmp_path / "empty", "2024-01-02")).generate(
            extend=True
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
