"""Package split Part 1 — Configs scaffolds from a neutral IR, not from a DMS.

`Configs` used to inherit `dms.Dms`; it now scaffolds from a neutral IR, so
`datagenic` ships without any platform (`opensee-core`) dependency. An Opensee DMS
is mapped in through the platform-side `dms_to_ir` adaptor (tested in opensee-core);
here we drive the native IR path directly. Scaffolding output is unchanged (also
guarded byte-for-byte by the golden suite, test_2).
"""

import json
import subprocess
import sys
from pathlib import Path

from datagenic import Configs

REPO_ROOT = Path(__file__).resolve().parents[1]
IR_PATH = "samples/ir/solutions_ir.json"


def _scaffold(ir_input):
    return Configs(
        input=ir_input,
        output_format="parquet",
        storage_type="local",
        number_of_rows_in_fact_tables=100,
        number_of_rows_in_non_fact_tables=30,
        number_of_batches=2,
        start_date="2024-01-01",
        end_date="2024-01-02",
        seed=0,
    ).config_template


def test_configs_from_dict_equals_from_path():
    """An IR dict and its JSON path scaffold the identical config."""
    from_path = _scaffold(IR_PATH)
    from_dict = _scaffold(json.loads(Path(REPO_ROOT / IR_PATH).read_text()))

    assert from_dict == from_path


def test_scaffold_has_expected_structure():
    """Sanity: scaffolding still produces the fact table + a populated config
    (byte-level 'unchanged' is enforced by the golden suite, test_2)."""
    template = _scaffold(IR_PATH)

    assert "default_parameters" in template
    table_names = [t["table_name"] for t in template["tables"]]
    assert any("FactTable" in n for n in table_names)
    assert len(template["tables"]) > 1


def test_configs_has_no_dms_import():
    """Importing datagenic.scaffolding (where Configs lives) must NOT pull in any
    platform DMS module, and the module must not expose a Dms symbol (static
    decoupling check — even stronger here, where opensee_core isn't installed)."""
    code = (
        "import datagenic.scaffolding as m, sys; "
        "print('opensee_core.dms' in sys.modules, hasattr(m, 'Dms'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False False"


def test_configs_rejects_bad_input():
    """A non-dict / non-str input is a clear error (was an implicit Dms failure)."""
    import pytest

    with pytest.raises(TypeError, match="must be a dict or a string"):
        Configs(input=12345)


# --------------------------------------------------------------------------- #
# dump() path handling (regression)
# --------------------------------------------------------------------------- #
_DUMP_IR = {
    "name": "shop",
    "tables": [
        {
            "type": "fact",
            "name": "orders",
            "keys": ["id"],
            "columns": [{"name": "id", "type": "int"}],
        }
    ],
}


def test_dump_bare_filename_writes_file_not_junk_dir(tmp_path, monkeypatch):
    """Regression: dump('x.json') with no directory must write x.json to the cwd, not
    mkdir a truncated 'x.jso' directory (the old `file_path[:rfind('/')]` slice bug)."""
    monkeypatch.chdir(tmp_path)
    Configs(input=_DUMP_IR, output_format="parquet").dump(file_path="gen_config.json")

    assert (tmp_path / "gen_config.json").is_file()
    assert not (tmp_path / "gen_config.jso").exists()  # the bug created this junk dir


def test_dump_creates_nested_parent_dirs(tmp_path):
    """A path with directories that don't exist yet is created."""
    out = tmp_path / "a" / "b" / "cfg.json"
    Configs(input=_DUMP_IR, output_format="parquet").dump(file_path=str(out))
    assert out.is_file()
