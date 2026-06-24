"""Phase 3 / file 07 — config inheritance via `extends`.

A child config can `extends` a base: default_parameters deep-merge (child wins),
tables merge by table_name (matching deep-merged, base-only inherited, new
appended). Resolved before generation; cycles raise.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator


def _table(name, rows):
    return {
        "table_name": name,
        "number_of_rows": rows,
        "post_processor": {},
        "columns": [gf._col(name, "id", "int", True, 1, rows)],
        "single_batch": False,
        "single_date": False,
        "dump_results": True,
    }


def _write(path, cfg):
    path.write_text(json.dumps(cfg))
    return str(path)


def _base_cfg(out):
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
        "tables": [_table("a", 10), _table("b", 20)],
    }


def _gen(path):
    return Generator(data_generator_config_file=path)


def test_inherits_all_base_tables(tmp_path):
    _write(tmp_path / "base.json", _base_cfg(tmp_path / "out"))
    child = _write(
        tmp_path / "child.json", {"extends": "base.json", "default_parameters": {"seed": 1}}
    )
    src = _gen(child)._source_config
    assert [t["table_name"] for t in src["tables"]] == ["a", "b"]


def test_default_parameters_deep_merge(tmp_path):
    _write(tmp_path / "base.json", _base_cfg(tmp_path / "out"))
    child = _write(
        tmp_path / "child.json",
        {"extends": "base.json", "default_parameters": {"seed": 99}},
    )
    dp = _gen(child)._source_config["default_parameters"]
    assert dp["seed"] == 99  # child wins
    assert dp["start_date"] == "2024-01-01"  # base retained
    assert dp["number_of_batches"] == 1


def test_overrides_one_table_others_inherited(tmp_path):
    _write(tmp_path / "base.json", _base_cfg(tmp_path / "out"))
    child = _write(
        tmp_path / "child.json",
        {
            "extends": "base.json",
            "tables": [{"table_name": "a", "number_of_rows": 500}, _table("c", 7)],
        },
    )
    tables = {t["table_name"]: t for t in _gen(child)._source_config["tables"]}
    assert list(tables) == ["a", "b", "c"]  # a overridden in place, b inherited, c appended
    assert tables["a"]["number_of_rows"] == 500  # child override
    assert tables["a"]["columns"]  # base detail retained (not wiped by partial override)
    assert tables["b"]["number_of_rows"] == 20  # untouched


def test_circular_extends_raises(tmp_path):
    _write(tmp_path / "A.json", {"extends": "B.json", "default_parameters": {}, "tables": []})
    _write(tmp_path / "B.json", {"extends": "A.json", "default_parameters": {}, "tables": []})
    with pytest.raises(ValueError, match="circular"):
        _gen(str(tmp_path / "A.json"))


def test_extended_config_generates(tmp_path):
    out = tmp_path / "out"
    _write(tmp_path / "base.json", _base_cfg(out))
    child = _write(
        tmp_path / "child.json",
        {"extends": "base.json", "tables": [{"table_name": "a", "number_of_rows": 5}]},
    )
    Generator(data_generator_config_file=child).generate()
    assert (out / "a").exists() and (out / "b").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
