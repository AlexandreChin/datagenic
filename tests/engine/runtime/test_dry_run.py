"""Phase 3 / file 07 — dry-run mode.

generate(dry_run=True) validates the config and prints the generation plan
without producing any data or invoking any generator/postprocessor.
"""

import golden_fixtures as gf  # noqa: E402
import pytest

from datagenic import Generator


def _cfg(out):
    cfg = gf._dms_config()
    cfg["default_parameters"]["output_folder_path"] = str(out)
    return cfg


def test_dry_run_produces_no_output(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_cfg(out)).generate(dry_run=True)
    # nothing written — not even the output directory
    assert not out.exists() or not any(out.rglob("*.parquet"))


def test_dry_run_prints_plan_with_tables_and_rows(tmp_path, capsys):
    cfg = _cfg(tmp_path / "out")
    Generator(data_generator_config_file=cfg).generate(dry_run=True)
    plan = capsys.readouterr().out

    assert "[dry-run]" in plan
    assert "generation units" in plan
    for table in cfg["tables"]:
        assert table["table_name"] in plan
    assert "rows/batch" in plan


def test_dry_run_estimates_dynamic_row_counts(tmp_path, capsys):
    """A {from_table, multiplier} table shows a concrete estimated rows/batch,
    resolved through the dependency chain."""
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
                "table_name": "ref",
                "number_of_rows": 20,
                "post_processor": {},
                "columns": [gf._col("ref", "id", "int", True, 1, 20)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
            {
                "table_name": "dyn",
                "number_of_rows": {"from_table": "ref", "multiplier": 2.0},
                "post_processor": {},
                "columns": [gf._col("dyn", "id", "int", True, 1, 40)],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }
    Generator(data_generator_config_file=cfg).generate(dry_run=True)
    plan = capsys.readouterr().out
    # ref = 20 rows/batch, dyn = ~40 (20 × 2.0), referencing 'ref'
    assert "~40 rows/batch" in plan
    assert "from ref" in plan


def test_dry_run_raises_on_invalid_config(tmp_path):
    cfg = _cfg(tmp_path / "out")
    cfg["tables"][0]["columns"][0]["column_generator"]["udf_module"] = "datagenic.nope"
    with pytest.raises(ImportError):
        Generator(data_generator_config_file=cfg).generate(dry_run=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
