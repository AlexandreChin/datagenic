"""Configs scaffolds from the neutral IR; an adaptor maps any other JSON shape in.

The IR (a flat ``{name, tables:[{kind,name,keys,columns:[{name,type}]}]}``) is the
*native* input — passed straight through with no adaptor. To scaffold from a different
schema (e.g. a bespoke JSON, or an Opensee DMS) pass ``adaptor=fn`` where
``fn(source) -> IR``. The DMS→IR adaptor ships in ``opensee_core.dms_adaptor.dms_to_ir``
and its end-to-end path is tested in the opensee-core suite; here we cover the native
IR and a generic custom adaptor (no platform dependency).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from datagenic import Configs, Generator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _native_ir():
    """A small, valid IR — what Configs consumes by default (no adaptor)."""
    return {
        "name": "shop",
        "tables": [
            {
                "kind": "fact",
                "name": "orders",
                "keys": ["order_id"],
                "columns": [
                    {"name": "order_id", "type": "int"},
                    {"name": "amount", "type": "float"},
                ],
            },
            {
                "kind": "dimension",
                "name": "products",
                "keys": ["sku"],
                "columns": [{"name": "sku", "type": "string"}],
            },
        ],
    }


def _alt_adaptor(source):
    """Map a small bespoke schema -> the datagenic IR."""
    return {
        "name": source["modelName"],
        "tables": [
            {
                "kind": "dimension",
                "name": e["id"],
                "keys": e["key"],
                "columns": [{"name": f["name"], "type": f["type"]} for f in e["fields"]],
            }
            for e in source["entities"]
        ],
    }


def test_scaffold_from_native_ir():
    """The IR is the native input — no adaptor needed."""
    template = Configs(
        input=_native_ir(), output_format="parquet", number_of_batches=1
    ).config_template
    assert sorted(t["table_name"] for t in template["tables"]) == ["orders", "products"]


def test_scaffold_from_arbitrary_json_via_adaptor():
    """A non-IR JSON + adaptor scaffolds a usable config."""
    source = {
        "modelName": "alt_model",
        "entities": [
            {
                "id": "customer",
                "key": ["cust_id"],
                "fields": [
                    {"name": "cust_id", "type": "int"},
                    {"name": "name", "type": "string"},
                ],
            }
        ],
    }
    template = Configs(
        input=source, adaptor=_alt_adaptor, output_format="parquet", number_of_batches=1
    ).config_template

    names = [t["table_name"] for t in template["tables"]]
    assert names == ["customer"]
    cols = {c["col_name"]: c for c in template["tables"][0]["columns"]}
    assert cols["cust_id"]["is_unicity_key"] is True
    assert cols["name"]["is_unicity_key"] is False
    # the adaptor's columns were turned into real generators
    assert cols["cust_id"]["column_generator"]["udf_module"] == "datagenic.generators"


def test_identity_adaptor_matches_no_adaptor():
    """An identity adaptor on an IR yields the same config as no adaptor — i.e. the
    adaptor seam doesn't change the native (IR) path."""
    ir = _native_ir()
    kw = dict(output_format="parquet", number_of_rows_in_fact_tables=100, number_of_batches=2)

    no_adaptor = Configs(input=ir, **kw).config_template
    identity = Configs(input=ir, adaptor=lambda s: s, **kw).config_template

    assert no_adaptor == identity


def test_scaffold_from_frozen_sample_ir():
    """The frozen sample IR (ex-DMS) scaffolds a multi-table config with a fact table."""
    ir = json.loads((REPO_ROOT / "samples" / "ir" / "solutions_ir.json").read_text())
    template = Configs(input=ir, output_format="parquet", number_of_batches=2).config_template
    names = [t["table_name"] for t in template["tables"]]
    assert any("FactTable" in n for n in names)
    assert len(template["tables"]) > 1


def test_generated_config_actually_generates(tmp_path):
    """The adaptor-scaffolded config runs end-to-end through the generator."""
    source = {
        "modelName": "alt",
        "entities": [{"id": "t", "key": ["id"], "fields": [{"name": "id", "type": "int"}]}],
    }
    cfg = Configs(
        input=source,
        adaptor=_alt_adaptor,
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_non_fact_tables=20,
        number_of_batches=1,
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    assert list((tmp_path / "out" / "t").rglob("*.parquet"))


def test_invalid_ir_raises_clear_error():
    """A malformed IR (adaptor not returning the IR shape) fails with a clear error,
    not a deep KeyError mid-scaffold."""
    bad = lambda s: {"name": "x"}  # noqa: E731  — missing 'tables'
    with pytest.raises(ValueError, match=r"'tables' list"):
        Configs(input={}, adaptor=bad)


def test_cli_generate_data_config_native_ir(tmp_path):
    """generate-data-config scaffolds from a native IR with no adaptor."""
    source = json.dumps(
        {
            "name": "m",
            "tables": [
                {
                    "kind": "fact",
                    "name": "f",
                    "keys": ["id"],
                    "columns": [{"name": "id", "type": "int"}],
                }
            ],
        }
    )
    out = tmp_path / "config.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datagenic",
            "generate-data-config",
            source,
            "-o",
            str(out),
            "--output-folder",
            str(tmp_path / "data"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    cfg = json.loads(out.read_text())
    assert [t["table_name"] for t in cfg["tables"]] == ["f"]
