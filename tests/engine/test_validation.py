"""Phase 3 — pre-flight validation (`Generator.validate()`).

Structural errors are caught up front, before any data is produced.
"""

import copy

import golden_fixtures as gf  # noqa: E402
import pytest

from datagenic import Generator

_PROBE = "postprocessors_udfs.generic_udfs"


def _g(cfg):
    return Generator(data_generator_config_file=cfg)


def test_valid_configs_pass():
    _g(gf._rng_config()).validate()
    _g(gf._dms_config()).validate()


def test_unknown_referenced_table_raises():
    cfg = gf._rng_config()
    cfg["tables"][0]["post_processor"] = {
        "udf_module": _PROBE,
        "udf_name": "RecordingPostProcessor",
        "udf_args": {"_input_tables": ["does_not_exist"]},
    }
    with pytest.raises(ValueError, match="does_not_exist"):
        _g(cfg).validate()


def test_bad_udf_module_raises():
    cfg = gf._rng_config()
    cfg["tables"][0]["columns"][0]["column_generator"]["udf_module"] = "datagenic.nope"
    with pytest.raises(ImportError):
        _g(cfg).validate()


def test_bad_udf_name_raises():
    cfg = gf._rng_config()
    cfg["tables"][0]["columns"][0]["column_generator"]["udf_name"] = "NoSuchGenerator"
    with pytest.raises(ImportError, match="NoSuchGenerator"):
        _g(cfg).validate()


def test_underscore_prefixed_udf_arg_raises():
    cfg = gf._rng_config()
    cfg["tables"][0]["post_processor"] = {
        "udf_module": _PROBE,
        "udf_name": "RecordingPostProcessor",
        "udf_args": {"_date": "2024-01-01"},
    }
    with pytest.raises(ValueError, match="_date"):
        _g(cfg).validate()


def test_required_args_enforced():
    cfg = gf._rng_config()
    cfg["tables"][0]["post_processor"] = {
        "udf_module": _PROBE,
        "udf_name": "RequiresArgs",
        "udf_args": {},
    }
    with pytest.raises(ValueError, match="needed"):
        _g(cfg).validate()

    cfg["tables"][0]["post_processor"]["udf_args"]["needed"] = 1
    _g(cfg).validate()  # present -> passes


def test_dynamic_from_table_order():
    cfg = gf._rng_config()
    t2 = copy.deepcopy(cfg["tables"][0])
    t2["table_name"] = "t2"
    t2["number_of_rows"] = {"from_table": "t_rng"}

    ok = copy.deepcopy(cfg)
    ok["tables"].append(t2)
    _g(ok).validate()  # t_rng appears earlier -> valid

    reversed_cfg = copy.deepcopy(cfg)
    reversed_cfg["tables"] = [t2, reversed_cfg["tables"][0]]
    with pytest.raises(ValueError, match="earlier"):
        _g(reversed_cfg).validate()


def test_validate_runs_before_any_io(tmp_path):
    cfg = gf._rng_config()
    out = tmp_path / "out"
    cfg["default_parameters"]["output_folder_path"] = str(out)
    cfg["tables"][0]["columns"][0]["column_generator"]["udf_module"] = "datagenic.nope"

    with pytest.raises(ImportError):
        _g(cfg).generate()
    assert not out.exists(), "no output should be written when validation fails"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
