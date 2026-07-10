"""Visualize a datagenic IR as a Mermaid ER diagram (``IRDiagram``).

The IR is the neutral input schema (tables, columns+types, keys, foreign_keys).
``IRDiagram`` renders it as a Mermaid ``erDiagram`` — the schema counterpart to
``GraphDependencies`` (which renders a generation *config* as a lineage flowchart).
Dependency-free (stdlib only), so ``import datagenic`` stays cheap.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from datagenic.ir_diagram import IRDiagram

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ir():
    return {
        "name": "shop",
        "tables": [
            {
                "type": "fact",
                "name": "orders",
                "keys": ["order_id"],
                "columns": [
                    {"name": "order_id", "type": "int"},
                    {"name": "sku", "type": "string"},
                    {"name": "amount", "type": "float"},
                ],
                "foreign_keys": [{"column": "sku", "references": "products.sku"}],
            },
            {
                "type": "referential",
                "name": "products",
                "keys": ["sku"],
                "columns": [
                    {"name": "sku", "type": "string"},
                    {"name": "label", "type": "string"},
                ],
            },
        ],
    }


def test_mermaid_starts_with_erdiagram_and_lists_entities():
    m = IRDiagram(_ir()).to_mermaid()
    assert m.lstrip().startswith("erDiagram")
    assert "orders {" in m
    assert "products {" in m


def test_columns_carry_type_and_pk_fk_markers():
    m = IRDiagram(_ir()).to_mermaid()
    assert "int order_id PK" in m  # a unicity key -> PK
    assert "string sku FK" in m  # a foreign key -> FK
    assert "float amount" in m  # a plain column, no marker
    assert "string sku PK" in m  # products.sku is that table's key


def test_column_that_is_both_key_and_fk_is_marked_pk_fk():
    ir = {
        "name": "m",
        "tables": [
            {
                "type": "fact",
                "name": "link",
                "keys": ["a_id"],
                "columns": [{"name": "a_id", "type": "int"}],
                "foreign_keys": [{"column": "a_id", "references": "a.id"}],
            },
            {
                "type": "referential",
                "name": "a",
                "keys": ["id"],
                "columns": [{"name": "id", "type": "int"}],
            },
        ],
    }
    assert "int a_id PK,FK" in IRDiagram(ir).to_mermaid()


def test_foreign_key_becomes_a_many_to_one_relationship():
    m = IRDiagram(_ir()).to_mermaid()
    assert "orders }o--|| products" in m  # fact -> referential, many-to-one
    assert "sku" in m.split("orders }o--|| products")[1].splitlines()[0]  # label names the column


def test_array_type_is_sanitised_to_a_mermaid_safe_token():
    ir = {
        "name": "m",
        "tables": [
            {
                "type": "fact",
                "name": "t",
                "keys": ["id"],
                "columns": [
                    {"name": "id", "type": "int"},
                    {"name": "pnl", "type": "array(float32)"},
                ],
            }
        ],
    }
    m = IRDiagram(ir).to_mermaid()
    assert "array(float32)" not in m  # raw parens would break mermaid
    assert "array_float32 pnl" in m


def test_accepts_a_json_path():
    p = REPO_ROOT / "samples" / "ir" / "solutions_ir.json"
    m = IRDiagram(str(p)).to_mermaid()
    assert m.lstrip().startswith("erDiagram")


def test_dump_file_writes_fenced_mermaid(tmp_path):
    out = tmp_path / "ir.md"
    IRDiagram(_ir()).dump_file(str(out))
    text = out.read_text()
    assert "```mermaid" in text
    assert "erDiagram" in text
    assert "orders }o--|| products" in text


def test_dump_html_is_self_contained_and_embeds_the_diagram(tmp_path):
    out = tmp_path / "ir.html"
    IRDiagram(_ir()).dump_html(str(out))
    html = out.read_text().lower()
    assert "<html" in html
    assert "mermaid" in html
    assert "orders {" in html  # the diagram source is embedded


def test_invalid_ir_raises():
    with pytest.raises(ValueError):
        IRDiagram({"name": "x", "tables": "not-a-list"}).to_mermaid()


def test_cli_visualize_ir_prints_mermaid_to_stdout(tmp_path):
    p = tmp_path / "ir.json"
    p.write_text(json.dumps(_ir()))
    result = subprocess.run(
        [sys.executable, "-m", "datagenic", "visualize-ir", str(p)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "erDiagram" in result.stdout
    assert "orders }o--|| products" in result.stdout


def test_cli_visualize_ir_writes_html_file(tmp_path):
    p = tmp_path / "ir.json"
    p.write_text(json.dumps(_ir()))
    out = tmp_path / "diagram.html"
    result = subprocess.run(
        [sys.executable, "-m", "datagenic", "visualize-ir", str(p), "-o", str(out), "--html"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "mermaid" in out.read_text().lower()
