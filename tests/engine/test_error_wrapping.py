"""Phase 3 — contextual error wrapping.

A generator/postprocessor exception is re-raised as RuntimeError annotated with
table/date/batch, while the original exception is preserved as __cause__.
"""

import pytest

from datagenic import Generator

_PROBE = "postprocessors_udfs.generic_udfs"


def _base(out):
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
        "tables": [],
    }


def test_generator_exception_wrapped_with_context(tmp_path):
    cfg = _base(tmp_path / "out")
    cfg["tables"] = [
        {
            "table_name": "t",
            "number_of_rows": 10,
            "post_processor": {},
            "columns": [
                {
                    "col_name": "id",
                    "col_type": "int",
                    "is_unicity_key": True,
                    "column_generator": {
                        "udf_name": "UniqueIncrementalIntegers",
                        "udf_module": "datagenic.generators",
                        "args": {},
                    },
                },
                {
                    "col_name": "boom",
                    "col_type": "string",
                    "is_unicity_key": False,
                    "column_generator": {
                        "udf_name": "BoomGenerator",
                        "udf_module": _PROBE,
                        "args": {},
                    },
                },
            ],
            "single_batch": False,
            "single_date": False,
            "dump_results": True,
        }
    ]

    with pytest.raises(RuntimeError) as ei:
        Generator(data_generator_config_file=cfg).generate()

    msg = str(ei.value)
    assert "table 't'" in msg
    assert "boom" in msg  # the column name
    assert "2024-01-01" in msg and "batch 1" in msg
    assert isinstance(ei.value.__cause__, ValueError)
    assert "boom-gen" in str(ei.value.__cause__)


def test_postprocessor_exception_wrapped_with_context(tmp_path):
    cfg = _base(tmp_path / "out")
    cfg["tables"] = [
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
    ]

    with pytest.raises(RuntimeError) as ei:
        Generator(data_generator_config_file=cfg).generate()

    msg = str(ei.value)
    assert "postprocessor" in msg and "table 't'" in msg and "batch 1" in msg
    assert isinstance(ei.value.__cause__, ValueError)
    assert "boom-pp" in str(ei.value.__cause__)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
