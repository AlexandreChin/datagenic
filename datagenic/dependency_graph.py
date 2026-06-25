"""Build a mermaid dependency flowchart from a generation config
(``GraphDependencies``). Split out of ``data_generator`` (package split Part 6).
"""

import json
import re
from pathlib import Path


def _dedup(items: list) -> list:
    """Order-preserving de-duplication (deterministic, unlike ``list(set(...))``)."""
    seen: set = set()
    out: list = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


class GraphDependencies:
    _FILE_EXTENSIONS = (".csv", ".json", ".parquet", ".yaml", ".xlsx")

    @staticmethod
    def _nid(text: str) -> str:
        """Mermaid-safe node id (dots, slashes, etc. -> underscore)."""
        return re.sub(r"[^0-9A-Za-z_]", "_", text)

    @classmethod
    def _file_path_args(cls, udf_args: dict) -> list:
        """Arg values that look like input file paths (for file-dependency edges)."""
        return [
            value
            for value in udf_args.values()
            if isinstance(value, str) and value.endswith(cls._FILE_EXTENSIONS)
        ]

    def __init__(self, config_file_path: str):
        if config_file_path[-5:] != ".json":
            raise Exception("unrecognized config file")
        with open(config_file_path) as f:
            self._config = json.load(f)

        global_input_tables = self._config["default_parameters"]["postprocessor_args"].get(
            "_input_tables", []
        )

        # One subgraph per table; cross-table edges (input/reference tables, dynamic
        # row counts) reference nodes in *other* subgraphs, so they live at top level.
        self.groups: list = []
        self.cross_edges: list = []

        for table in self._config["tables"]:
            table_name = table["table_name"]
            tbl = self._nid(table_name)
            dump_results = table["dump_results"]
            table_class = "table" if dump_results else "intermediate_results"
            table_emoji = "🗄️" if dump_results else "🧪"
            has_columns = bool(table["columns"])
            post_processor = table["post_processor"]

            nodes: list = [f'{tbl}[("{table_emoji} {table_name}")]:::{table_class}']
            edges: list = []
            generator_sink = tbl  # where the column generators feed by default

            # ── post-processor ───────────────────────────────────────────────
            # When the table also has column generators, the column data is a
            # temporary result the post-processor consumes (generators -> temp
            # result -> post-processor -> table); otherwise it produces the table.
            if post_processor:
                udf_module = post_processor["udf_module"]
                udf_name = post_processor["udf_name"]
                udf_args = post_processor["udf_args"]
                pp = self._nid(f"{table_name}.{udf_module}.{udf_name}")

                nodes.append(f'{pp}{{{{"🧩 {udf_name}<br/>{udf_module}"}}}}:::postprocessor')
                edges.append(f"{pp} --> {tbl}")

                if has_columns:
                    tmp = self._nid(f"{table_name}.temp_result")
                    nodes.append(f'{tmp}[("🧪 temp result")]:::intermediate_results')
                    edges.append(f"{tmp} --> {pp}")
                    generator_sink = tmp

                # files read by the post-processor
                for path in self._file_path_args(udf_args):
                    fid = self._nid(f"file.{path}")
                    nodes.append(f'{fid}["📄 {path}"]:::file')
                    edges.append(f"{fid} --> {pp}")

                # cross-table: reference / input tables feed the post-processor
                for src in global_input_tables + udf_args.get("_input_tables", []):
                    self.cross_edges.append(f"{self._nid(src)} --> {pp}")

            # ── column generators ────────────────────────────────────────────
            # One aggregate node represents all of a table's column generators, so a
            # wide fact table doesn't explode into a node per column.
            if has_columns:
                gen = self._nid(f"{table_name}.column_generators")
                nodes.append(f'{gen}(["⚙️ column generators"]):::generator')
                edges.append(f"{gen} --> {generator_sink}")

                # files read by any column generator (e.g. ValuesFromFile)
                for column in table["columns"]:
                    for path in self._file_path_args(
                        column.get("column_generator", {}).get("args", {})
                    ):
                        fid = self._nid(f"file.{path}")
                        nodes.append(f'{fid}["📄 {path}"]:::file')
                        edges.append(f"{fid} --> {gen}")

                # dynamic row count: this table's size derives from another table
                number_of_rows = table.get("number_of_rows")
                if isinstance(number_of_rows, dict) and "from_table" in number_of_rows:
                    self.cross_edges.append(f"{self._nid(number_of_rows['from_table'])} --> {tbl}")

            self.groups.append(
                {
                    "id": tbl,
                    "title": table_name,
                    "nodes": _dedup(nodes),
                    "edges": _dedup(edges),
                }
            )

        self.cross_edges = _dedup(self.cross_edges)

    def dump_file(self, folder_path: str = "samples/graphs", mermaid_header: bool = True) -> None:
        """Write the dependency flowchart as a mermaid markdown file.

        Each table's pipeline is grouped in its own left-to-right subgraph
        (column generators -> [temp result] -> [post-processor] -> table). Node type
        is shown in both colour and shape: column generators are amber stadiums,
        post-processors pink hexagons, temp/intermediate results grey, tables green,
        input files blue. Cross-table edges (input tables, dynamic row counts) are
        drawn between subgraphs.
        """
        md_file_path = f"{folder_path}/data_generator_dependencies.md"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        lines: list = []
        if mermaid_header:
            lines.append("```mermaid")
        lines += [
            "---",
            "title: Data Generator Workflow",
            "---",
            "%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 45, 'rankSpacing': 70}}}%%",
            "flowchart LR",
            "",
            "classDef table fill:#bbf7d0,stroke:#16a34a,stroke-width:2.5px,color:#052e16",
            "classDef generator fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#451a03",
            "classDef postprocessor fill:#fce7f3,stroke:#db2777,stroke-width:1.5px,color:#500724",
            "classDef intermediate_results fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,"
            "stroke-dasharray:4 3,color:#0f172a",
            "classDef file fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554",
            "",
        ]

        # one subgraph per table (each laid out left-to-right)
        for g in self.groups:
            lines.append(f'subgraph {g["id"]}_pipe["{g["title"]}"]')
            lines.append("    direction LR")
            lines += [f"    {n}" for n in g["nodes"]]
            lines += [f"    {e}" for e in g["edges"]]
            lines.append("end")
            lines.append("")

        # cross-table edges (between subgraphs)
        if self.cross_edges:
            lines += self.cross_edges
            lines.append("")

        # legend — a horizontal row, ordered with invisible links (~~~)
        lines += [
            'subgraph legend["legend"]',
            "    direction LR",
            '    lf["📄 file"]:::file ~~~ lg(["⚙️ generator"]):::generator ~~~ '
            'lp{{"🧩 post-processor"}}:::postprocessor ~~~ '
            'lt[("🧪 temp / intermediate result")]:::intermediate_results ~~~ '
            'ltab[("🗄️ table")]:::table',
            "end",
            "",
        ]

        # no fill on the pipeline subgraphs; only the legend is filled
        for g in self.groups:
            lines.append(f"style {g['id']}_pipe fill:transparent")
        lines.append("style legend fill:#f1f5f9,stroke:#94a3b8")

        if mermaid_header:
            lines.append("```")

        Path(md_file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"flowchart file available here:{md_file_path}")
