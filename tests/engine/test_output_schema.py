"""Phase 3 — output schema validation.

Before writing, a postprocessor's returned DataFrame is validated against the
table's declared schema (explicit ``output_schema``, else ``columns``): missing
columns raise ValueError, incompatible dtypes raise TypeError, extra columns warn,
and ``dump_results: false`` tables are exempt.
"""

import pytest

from datagenic import Generator

_PROBE = "postprocessors_udfs.generic_udfs"


def _cfg(out, output_schema, cols, dtype="string", dump_results=True):
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
                "number_of_rows": 3,
                "output_schema": output_schema,
                "post_processor": {
                    "udf_module": _PROBE,
                    "udf_name": "ReturnsFrame",
                    "udf_args": {"cols": cols, "dtype": dtype},
                },
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": dump_results,
            }
        ],
    }


_SCHEMA_AB = [
    {"col_name": "a", "col_type": "string"},
    {"col_name": "b", "col_type": "string"},
]


def test_missing_column_raises(tmp_path):
    cfg = _cfg(tmp_path / "out", _SCHEMA_AB, cols=["a"])  # missing 'b'
    with pytest.raises(ValueError, match="b"):
        Generator(data_generator_config_file=cfg).generate()


def test_wrong_dtype_raises(tmp_path):
    schema = [{"col_name": "a", "col_type": "float"}]
    cfg = _cfg(tmp_path / "out", schema, cols=["a"], dtype="string")  # string, expected float
    with pytest.raises(TypeError, match="a"):
        Generator(data_generator_config_file=cfg).generate()


def test_allowed_coercion_passes(tmp_path):
    # expected int, returned int -> ok (int32/int64 differences are within category)
    schema = [{"col_name": "a", "col_type": "int64"}]
    cfg = _cfg(tmp_path / "out", schema, cols=["a"], dtype="int")
    Generator(data_generator_config_file=cfg).generate()  # no raise


def test_extra_column_is_warning_not_error(tmp_path):
    schema = [{"col_name": "a", "col_type": "string"}]
    out = tmp_path / "out"
    cfg = _cfg(out, schema, cols=["a", "b"])  # 'b' is extra
    Generator(data_generator_config_file=cfg).generate()  # no raise

    import pandas as pd

    files = list((out / "t").rglob("*.parquet"))
    assert files
    cols = set(pd.read_parquet(files[0]).columns)
    assert {"a", "b"} <= cols  # extra column is kept, just warned about


def test_dump_results_false_is_exempt(tmp_path):
    # wrong schema, but dump_results:false -> no validation, no error, no file
    cfg = _cfg(tmp_path / "out", _SCHEMA_AB, cols=["a"], dump_results=False)
    Generator(data_generator_config_file=cfg).generate()
    assert not (tmp_path / "out" / "t").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
