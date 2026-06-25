"""Build dependency visualisations from a generation config (``GraphDependencies``).

The config is parsed once into a structured node/edge model (``self.nodes`` /
``self.edges``); from it we render either a static **mermaid** flowchart
(``dump_file``) or an interactive **HTML** visualiser (``dump_html``). Split out of
``data_generator`` (package split Part 6).
"""

import json
import re
from collections import OrderedDict
from pathlib import Path

# node type -> (mermaid open, mermaid close, mermaid class, emoji, html shape)
_NODE_STYLE = {
    "table": ('[("', '")]', "table", "🗄️", "round-rectangle"),
    "intermediate": ('[("', '")]', "intermediate_results", "🧪", "round-rectangle"),
    "temp": ('[("', '")]', "intermediate_results", "🧪", "round-rectangle"),
    "generator": ('(["', '"])', "generator", "⚙️", "round-tag"),
    "postprocessor": ('{{"', '"}}', "postprocessor", "🧩", "hexagon"),
    "file": ('["', '"]', "file", "📄", "rectangle"),
}


class GraphDependencies:
    _FILE_EXTENSIONS = (".csv", ".json", ".parquet", ".yaml", ".xlsx")

    @staticmethod
    def _nid(text: str) -> str:
        """Mermaid/HTML-safe node id (dots, slashes, etc. -> underscore)."""
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

        # Structured model: nodes keyed by id (insertion order preserved), edges as
        # {source, target}. `group` is the owning table (used to draw mermaid subgraphs).
        self._nodes: OrderedDict = OrderedDict()
        self.edges: list = []

        def add_node(node_id, ntype, label, group, sublabel=""):
            if node_id not in self._nodes:
                self._nodes[node_id] = {
                    "id": node_id,
                    "type": ntype,
                    "label": label,
                    "sublabel": sublabel,
                    "group": group,
                }

        def add_edge(source, target):
            edge = {"source": source, "target": target}
            if edge not in self.edges:
                self.edges.append(edge)

        for table in self._config["tables"]:
            table_name = table["table_name"]
            tbl = self._nid(table_name)
            dump_results = table["dump_results"]
            table_type = "table" if dump_results else "intermediate"
            has_columns = bool(table["columns"])
            post_processor = table["post_processor"]

            add_node(tbl, table_type, table_name, group=table_name)
            generator_sink = tbl  # where the column generators feed by default

            # ── post-processor ───────────────────────────────────────────────
            # With column generators present, the column data is a temporary result
            # the post-processor consumes (generators -> temp -> post-processor ->
            # table); otherwise the post-processor produces the table directly.
            if post_processor:
                udf_module = post_processor["udf_module"]
                udf_name = post_processor["udf_name"]
                udf_args = post_processor["udf_args"]
                pp = self._nid(f"{table_name}.{udf_module}.{udf_name}")

                add_node(pp, "postprocessor", udf_name, group=table_name, sublabel=udf_module)
                add_edge(pp, tbl)

                if has_columns:
                    tmp = self._nid(f"{table_name}.temp_result")
                    add_node(tmp, "temp", "temp result", group=table_name)
                    add_edge(tmp, pp)
                    generator_sink = tmp

                for path in self._file_path_args(udf_args):
                    fid = self._nid(f"file.{path}")
                    add_node(fid, "file", path, group=table_name)
                    add_edge(fid, pp)

                # cross-table: reference / input tables feed the post-processor
                for src in global_input_tables + udf_args.get("_input_tables", []):
                    add_edge(self._nid(src), pp)

            # ── column generators ────────────────────────────────────────────
            if has_columns:
                gen = self._nid(f"{table_name}.column_generators")
                add_node(gen, "generator", "column generators", group=table_name)
                add_edge(gen, generator_sink)

                for column in table["columns"]:
                    for path in self._file_path_args(
                        column.get("column_generator", {}).get("args", {})
                    ):
                        fid = self._nid(f"file.{path}")
                        add_node(fid, "file", path, group=table_name)
                        add_edge(fid, gen)

                number_of_rows = table.get("number_of_rows")
                if isinstance(number_of_rows, dict) and "from_table" in number_of_rows:
                    add_edge(self._nid(number_of_rows["from_table"]), tbl)

        # any edge endpoint without a node (e.g. an input table not in this config)
        # becomes a bare table node so renderers never reference a missing id.
        for edge in self.edges:
            for endpoint in (edge["source"], edge["target"]):
                if endpoint not in self._nodes:
                    add_node(endpoint, "table", endpoint, group=endpoint)

    @property
    def nodes(self) -> list:
        return list(self._nodes.values())

    # ── chronological ordering ───────────────────────────────────────────────
    def _topological_order(self) -> list:
        """Node ids in dependency order, stable by insertion order (Kahn's algo).

        Used by the HTML 'Chronological' layout: each node gets its own column, so
        no two nodes share a vertical line and reading left-to-right walks the DAG.
        """
        ids = list(self._nodes)
        adjacency: dict = {i: [] for i in ids}
        indegree: dict = {i: 0 for i in ids}
        for edge in self.edges:
            adjacency[edge["source"]].append(edge["target"])
            indegree[edge["target"]] += 1

        order: list = []
        placed: set = set()
        while len(order) < len(ids):
            # always take the earliest (insertion order) node with no pending deps
            nxt = next((i for i in ids if i not in placed and indegree[i] == 0), None)
            if nxt is None:  # cycle / leftover — append the rest in insertion order
                order += [i for i in ids if i not in placed]
                break
            order.append(nxt)
            placed.add(nxt)
            for tgt in adjacency[nxt]:
                indegree[tgt] -= 1
        return order

    # ── mermaid (static) ──────────────────────────────────────────────────────
    def _mermaid_node(self, node: dict) -> str:
        open_, close_, cls, emoji, _ = _NODE_STYLE[node["type"]]
        label = f"{emoji} {node['label']}"
        if node["sublabel"]:
            label += f"<br/>{node['sublabel']}"
        return f"{node['id']}{open_}{label}{close_}:::{cls}"

    def dump_file(self, folder_path: str = "samples/graphs", mermaid_header: bool = True) -> None:
        """Write the dependency flowchart as a mermaid markdown file.

        One left-to-right subgraph per table (column generators -> [temp result] ->
        [post-processor] -> table); node type shows in colour and shape. Cross-table
        edges (input tables, dynamic row counts) are drawn between subgraphs.
        """
        md_file_path = f"{folder_path}/data_generator_dependencies.md"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        id2group = {n["id"]: n["group"] for n in self.nodes}
        groups: OrderedDict = OrderedDict()
        for node in self.nodes:
            groups.setdefault(node["group"], []).append(node)
        intra: dict = {g: [] for g in groups}
        cross: list = []
        for edge in self.edges:
            line = f"{edge['source']} --> {edge['target']}"
            sg, tg = id2group.get(edge["source"]), id2group.get(edge["target"])
            (intra[sg].append(line) if sg == tg else cross.append(line))

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

        for group, group_nodes in groups.items():
            lines.append(f'subgraph {self._nid(group)}_pipe["{group}"]')
            lines.append("    direction LR")
            lines += [f"    {self._mermaid_node(n)}" for n in group_nodes]
            lines += [f"    {e}" for e in intra[group]]
            lines.append("end")
            lines.append("")

        if cross:
            lines += cross
            lines.append("")

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
        for group in groups:
            lines.append(f"style {self._nid(group)}_pipe fill:transparent")
        lines.append("style legend fill:#f1f5f9,stroke:#94a3b8")

        if mermaid_header:
            lines.append("```")

        Path(md_file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"flowchart file available here:{md_file_path}")

    # ── HTML (interactive) ─────────────────────────────────────────────────────
    def _cytoscape_elements(self) -> list:
        """Cytoscape elements for the 'Chronological' preset layout, plus edges.

        Tables (and other 'primary' nodes) step diagonally by topological rank, so
        each lands on its own row — unconnected nodes never share a row. A feeder
        node (column generators / input file) adopts the **row of the node it feeds**
        and sits to its left, so a generator pairs with its table on the same line
        instead of getting its own step. x stays unique per node (rank).
        """
        order = self._topological_order()
        rank = {nid: i for i, nid in enumerate(order)}
        px = {nid: rank[nid] * 260 for nid in rank}
        py = {nid: rank[nid] * 120 for nid in rank}

        # feeders share the y of the (first) node they feed → same row, to the left
        first_consumer: dict = {}
        for edge in self.edges:
            first_consumer.setdefault(edge["source"], edge["target"])
        for node in self.nodes:
            if node["type"] in ("generator", "file"):
                target = first_consumer.get(node["id"])
                if target in py:
                    py[node["id"]] = py[target]

        elements: list = []
        for node in self.nodes:
            emoji = _NODE_STYLE[node["type"]][3]
            display = f"{emoji} {node['label']}"
            if node["sublabel"]:
                display += f"\n{node['sublabel']}"
            nid = node["id"]
            elements.append(
                {
                    "data": {
                        "id": nid,
                        "type": node["type"],
                        "display": display,
                        "rank": rank[nid],
                        "px": px[nid],
                        "py": py[nid],
                    },
                    "position": {"x": px[nid], "y": py[nid]},
                }
            )
        for n, edge in enumerate(self.edges):
            elements.append(
                {"data": {"id": f"e{n}", "source": edge["source"], "target": edge["target"]}}
            )
        return elements

    def dump_html(self, folder_path: str = "samples/graphs") -> None:
        """Write an interactive HTML visualiser (Cytoscape.js + dagre).

        A toolbar toggles between **Chronological** (topological order, one node per
        column — nothing shares a vertical line) and **Compact** (dagre layered)
        layouts, plus fit, per-type filters, search, and click-to-highlight a node's
        upstream/downstream dependencies. Self-contained except for the CDN libraries.
        """
        html_file_path = f"{folder_path}/data_generator_dependencies.html"
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        page = _HTML_TEMPLATE.replace(
            "/*ELEMENTS*/", json.dumps(self._cytoscape_elements(), ensure_ascii=False)
        )
        Path(html_file_path).write_text(page, encoding="utf-8")
        print(f"interactive graph available here:{html_file_path}")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Generator Workflow</title>
<style>
  body { margin: 0; font-family: system-ui, -apple-system, sans-serif; }
  #bar { position: fixed; inset: 0 0 auto 0; height: 46px; display: flex; gap: 10px;
         align-items: center; padding: 0 14px; background: #0f172a; color: #e2e8f0; z-index: 10; }
  #bar strong { margin-right: 8px; }
  #bar button, #bar input { font: inherit; }
  #bar button { background: #1e293b; color: #e2e8f0; border: 1px solid #334155;
                border-radius: 6px; padding: 6px 11px; cursor: pointer; }
  #bar button:hover { background: #334155; }
  #search { padding: 6px 9px; border-radius: 6px; border: 1px solid #334155;
            background: #1e293b; color: #e2e8f0; }
  #filters { display: flex; gap: 10px; margin-left: auto; font-size: 12px; }
  #filters label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }
  #cy { position: fixed; inset: 46px 0 0 0; background: #ffffff; }
</style></head><body>
<div id="bar">
  <strong>Data Generator Workflow</strong>
  <button id="layout">Layout: Chronological</button>
  <button id="fit">Fit</button>
  <input id="search" placeholder="search nodes…">
  <span id="filters"></span>
</div>
<div id="cy"></div>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<script>
const elements = /*ELEMENTS*/;
const T = {
  table:         { bg: '#bbf7d0', line: '#16a34a', shape: 'round-rectangle' },
  intermediate:  { bg: '#f1f5f9', line: '#64748b', shape: 'round-rectangle' },
  temp:          { bg: '#f1f5f9', line: '#64748b', shape: 'round-rectangle' },
  generator:     { bg: '#fef3c7', line: '#d97706', shape: 'round-tag' },
  postprocessor: { bg: '#fce7f3', line: '#db2777', shape: 'hexagon' },
  file:          { bg: '#dbeafe', line: '#2563eb', shape: 'rectangle' },
};
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements,
  style: [
    { selector: 'node', style: {
        'label': 'data(display)', 'text-wrap': 'wrap', 'text-max-width': '170px',
        'text-valign': 'center', 'text-halign': 'center', 'font-size': '11px',
        'width': 'label', 'height': 'label', 'padding': '10px', 'color': '#0f172a',
        'background-color': (e) => T[e.data('type')].bg,
        'border-color': (e) => T[e.data('type')].line, 'border-width': 2,
        'shape': (e) => T[e.data('type')].shape } },
    { selector: 'edge', style: {
        'width': 2, 'line-color': '#94a3b8', 'target-arrow-color': '#94a3b8',
        'target-arrow-shape': 'triangle', 'curve-style': 'unbundled-bezier' } },
    { selector: '.faded', style: { 'opacity': 0.12 } },
    { selector: '.hl', style: { 'line-color': '#0ea5e9', 'target-arrow-color': '#0ea5e9', 'width': 3 } },
  ],
});
const layouts = {
  // Chronological: step each node diagonally by topological rank — unique x AND
  // unique y, so no two nodes share a column or a row (unconnected nodes are never
  // on the same line). Deterministic (positions derived from rank), so toggling back
  // always restores it.
  chrono:  { name: 'preset', fit: true, padding: 40,
             positions: (n) => ({ x: n.data('px'), y: n.data('py') }) },
  compact: { name: 'dagre', rankDir: 'LR', nodeSep: 25, rankSep: 70, edgeSep: 10, fit: true, padding: 40 },
};
let mode = 'chrono';
const runLayout = () => cy.layout(layouts[mode]).run();
runLayout();
document.getElementById('layout').onclick = (e) => {
  mode = mode === 'chrono' ? 'compact' : 'chrono';
  e.target.textContent = 'Layout: ' + (mode === 'chrono' ? 'Chronological' : 'Compact');
  runLayout();
};
document.getElementById('fit').onclick = () => cy.fit(undefined, 40);
cy.on('tap', 'node', (evt) => {
  const n = evt.target;
  cy.elements().removeClass('hl');
  cy.elements().addClass('faded');
  const up = n.predecessors(); // upstream nodes + edges only
  up.union(n).removeClass('faded');
  up.filter('edge').addClass('hl');
});
cy.on('tap', (evt) => { if (evt.target === cy) cy.elements().removeClass('faded hl'); });
document.getElementById('search').oninput = (e) => {
  const q = e.target.value.trim().toLowerCase();
  cy.nodes().forEach((n) => {
    const hit = q && n.data('display').toLowerCase().includes(q);
    n.style('border-width', hit ? 5 : 2);
    n.style('border-color', hit ? '#0ea5e9' : T[n.data('type')].line);
  });
};
const filters = document.getElementById('filters');
[...new Set(cy.nodes().map((n) => n.data('type')))].sort().forEach((t) => {
  const l = document.createElement('label');
  l.innerHTML = '<input type="checkbox" checked data-t="' + t + '"> ' + t;
  filters.appendChild(l);
});
filters.onchange = (e) => {
  const t = e.target.dataset.t;
  cy.nodes('[type = "' + t + '"]').style('display', e.target.checked ? 'element' : 'none');
};
</script></body></html>
"""
