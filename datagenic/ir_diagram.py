"""Visualize a datagenic **IR** (the neutral input schema) as a Mermaid ER diagram.

``IRDiagram`` is the schema counterpart to ``GraphDependencies`` (which renders a
generation *config* as a lineage flowchart): it takes an IR — a dict or a path to IR
JSON — and renders an ``erDiagram`` where each table is an entity (its columns with
types, unicity keys marked ``PK``, foreign-key columns marked ``FK``) and each declared
foreign key is a many-to-one relationship edge to the referenced table.

Dependency-free (stdlib only: ``json`` / ``pathlib`` / ``re``) so a bare
``import datagenic`` — and importing this module — pulls in no polars/pandas/numpy.

Usage::

    from datagenic import IRDiagram
    print(IRDiagram("samples/ir/solutions_ir.json").to_mermaid())   # paste into mermaid.live
    IRDiagram(ir_dict).dump_file("ir.md")     # fenced ```mermaid``` markdown
    IRDiagram(ir_dict).dump_html("ir.html")   # standalone page (mermaid via CDN)
"""

import json
import re
from pathlib import Path

# Mermaid relationship: a fact row references exactly one referential row, and a
# referential row is referenced by zero-or-more fact rows -> many-to-one.
_REL = "}o--||"


def _safe(token: str) -> str:
    """A Mermaid-safe identifier: keep word chars, fold everything else to ``_``.

    So ``array(float32)`` -> ``array_float32`` (raw parens break the erDiagram parser),
    ``float32`` stays ``float32``. Empty input -> ``unknown``."""
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(token)).strip("_")
    return cleaned or "unknown"


class IRDiagram:
    """Render a datagenic IR as a Mermaid ``erDiagram`` (+ markdown / HTML dumps)."""

    def __init__(self, ir):
        """``ir`` is an IR dict, or a path (str/Path) to IR JSON."""
        if isinstance(ir, dict):
            self._ir = ir
        else:
            self._ir = json.loads(Path(ir).read_text())
        self._validate(self._ir)

    @staticmethod
    def _validate(ir) -> None:
        if not isinstance(ir, dict) or not isinstance(ir.get("tables"), list):
            raise ValueError("IRDiagram: IR must be a dict with a 'tables' list")
        for i, table in enumerate(ir["tables"]):
            if (
                not isinstance(table, dict)
                or "name" not in table
                or not isinstance(table.get("columns"), list)
            ):
                raise ValueError(f"IRDiagram: tables[{i}] needs a 'name' and a 'columns' list")

    def to_mermaid(self) -> str:
        """The Mermaid ``erDiagram`` source for this IR."""
        lines = ["erDiagram"]
        rels: list[str] = []
        for table in self._ir["tables"]:
            name = _safe(table["name"])
            keys = set(table.get("keys", []))
            fk_cols = {fk["column"] for fk in table.get("foreign_keys", [])}
            lines.append(f"  %% {table['name']} — {table.get('type', 'referential')}")
            lines.append(f"  {name} {{")
            for col in table["columns"]:
                col_name = _safe(col["name"])
                col_type = _safe(col.get("type", "unknown"))
                marker = ",".join(
                    m
                    for m, hit in (("PK", col["name"] in keys), ("FK", col["name"] in fk_cols))
                    if hit
                )
                lines.append(f"    {col_type} {col_name} {marker}".rstrip())
            lines.append("  }")

            for fk in table.get("foreign_keys", []):
                refs = fk["references"]
                for ref in refs if isinstance(refs, list) else [refs]:
                    ref_table, _, ref_col = ref.partition(".")
                    label = f"{fk['column']} → {ref_col}" if ref_col else fk["column"]
                    rels.append(f'  {name} {_REL} {_safe(ref_table)} : "{label}"')

        return "\n".join(lines + rels)

    def dump_file(self, file_path: str) -> None:
        """Write a markdown file with the diagram in a fenced ```mermaid``` block."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        md = f"# IR: {self._ir.get('name', 'model')}\n\n```mermaid\n{self.to_mermaid()}\n```\n"
        path.write_text(md, encoding="utf-8")
        print(f"IR diagram written to {path}")

    def dump_html(self, file_path: str) -> None:
        """Write a standalone HTML page that renders the diagram (Mermaid via CDN)."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        name = self._ir.get("name", "model")
        page = _HTML_TEMPLATE.replace("/*NAME*/", name).replace("/*DIAGRAM*/", self.to_mermaid())
        path.write_text(page, encoding="utf-8")
        print(f"IR diagram written to {path}")


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IR: /*NAME*/</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  body{margin:0;background:#0b1020;color:#eef2fb;
    font-family:system-ui,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;}
  header{padding:20px 28px;border-bottom:1px solid #1e2640;}
  header h1{margin:0;font-size:20px;font-weight:650;}
  header span{color:#97a1ba;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;}
  .wrap{padding:28px;overflow:auto;}
  .mermaid{display:flex;justify-content:center;}
</style></head>
<body>
  <header><h1>IR: /*NAME*/</h1><span>datagenic internal representation · entity-relationship diagram</span></header>
  <div class="wrap"><pre class="mermaid">
/*DIAGRAM*/
  </pre></div>
  <script>mermaid.initialize({startOnLoad:true, theme:"dark", er:{useMaxWidth:true}});</script>
</body></html>
"""
