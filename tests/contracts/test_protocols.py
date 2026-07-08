"""Phase 3 / file 08 — generator & postprocessor protocol enforcement.

validate() checks every generator/postprocessor honours the call()/recall()
contract up front, so a misspelled/missing call() or a non-callable recall fails
at validation time instead of mid-run on date 2.
"""

import golden_fixtures as gf  # noqa: E402
import pytest

from datagenic import Generator
from datagenic.protocols import (
    ColumnGenerator,
    PostProcessor,
    StatelessColumnGenerator,
    StatelessPostProcessor,
)

_UDFS = "postprocessors_udfs.generic_udfs"


class _Both:
    def call(self, **k): ...
    def recall(self, **k): ...


class _CallOnly:
    def call(self, **k): ...


def test_protocols_isinstance():
    assert isinstance(_Both(), PostProcessor)
    assert isinstance(_Both(), StatelessPostProcessor)
    assert isinstance(_CallOnly(), StatelessPostProcessor)
    assert not isinstance(_CallOnly(), PostProcessor)  # no recall

    assert isinstance(_Both(), ColumnGenerator)
    assert isinstance(_CallOnly(), StatelessColumnGenerator)
    assert not isinstance(_CallOnly(), ColumnGenerator)


def _pp_cfg(out, udf_name):
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
                "number_of_rows": 5,
                "post_processor": {"udf_module": _UDFS, "udf_name": udf_name, "udf_args": {}},
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def test_validate_rejects_noncallable_recall_postprocessor(tmp_path):
    cfg = _pp_cfg(tmp_path / "out", "NonCallableRecall")
    with pytest.raises(TypeError, match="recall"):
        Generator(data_generator_config_file=cfg).validate()


def test_validate_rejects_noncallable_call_generator(tmp_path):
    cfg = _pp_cfg(tmp_path / "out", "ImportFile")  # placeholder postprocessor (valid)
    cfg["tables"][0]["post_processor"] = {}
    cfg["tables"][0]["columns"] = [
        {
            "col_name": "x",
            "col_type": "string",
            "is_unicity_key": True,
            "column_generator": {
                "udf_name": "NonCallableCallGenerator",
                "udf_module": _UDFS,
                "args": {},
            },
        }
    ]
    with pytest.raises(TypeError, match="call"):
        Generator(data_generator_config_file=cfg).validate()


def test_valid_udfs_pass_validation():
    # existing generators (rng/dms) and KeyEnforcer-style postprocessors validate cleanly
    Generator(data_generator_config_file=gf._rng_config()).validate()
    Generator(data_generator_config_file=gf._dms_config()).validate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
