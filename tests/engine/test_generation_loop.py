import json
from pathlib import Path

from datagenic import Configs, Generator, GraphDependencies

# The native input to Configs is the neutral IR. This sample IR is the frozen
# output the Opensee `dms_to_ir` adaptor produced for the sample DMS; feeding it
# directly keeps datagenic standalone (the adaptor path is tested in opensee-core).
SOLUTIONS_IR = "samples/ir/solutions_ir.json"


def test_data_generator_column_generator():
    ######## parameters ###########
    storage_type = "local"
    output_format = "parquet"
    output_folder_path = "samples/data_generator/solutions_dms_fake_data"
    generator_config_file_path = (
        "samples/data_generation_config/data_generator_config_columns_generator.json"
    )
    ###############################

    ir = json.loads(Path(SOLUTIONS_IR).read_text())

    # create config file from the native IR
    Configs(
        input=ir,
        storage_type=storage_type,
        output_format=output_format,
        output_folder_path=output_folder_path,
    ).dump(file_path=generator_config_file_path)

    # generate data
    Generator(data_generator_config_file=generator_config_file_path).generate()


def test_data_generator_post_processor():
    # generate data
    Generator(
        data_generator_config_file="samples/data_generation_config/data_generator_config_post_processor.json"
    ).generate()


def test_data_generator_graph_dependencies():
    GraphDependencies(
        "samples/data_generation_config/data_generator_config_post_processor.json"
    ).dump_file(mermaid_header=False)


def test_data_generator_graph_html(tmp_path):
    """The interactive HTML visualiser renders, embeds the elements, and lays nodes
    out chronologically with a unique x per node (none share a vertical line)."""
    gd = GraphDependencies(
        "samples/data_generation_config/data_generator_config_post_processor.json"
    )
    gd.dump_html(folder_path=str(tmp_path))
    html = (tmp_path / "data_generator_dependencies.html").read_text()
    assert "cytoscape" in html and "Layout: Chronological" in html
    assert "const elements = [" in html

    els = gd._cytoscape_elements()
    positioned = [e for e in els if "position" in e]
    xs = [e["position"]["x"] for e in positioned]
    assert len(xs) == len(set(xs)), "chronological layout must give every node a unique x"
    # each table-type node is on its own row (unconnected nodes never share a y); a
    # feeder (generator/file) shares the row of the node it feeds.
    table_ys = [
        e["position"]["y"] for e in positioned if e["data"]["type"] in ("table", "intermediate")
    ]
    assert len(table_ys) == len(set(table_ys)), "each table must be on its own row"


def test_table_relations_contracts_through_internals(tmp_path):
    """'Tables only' lineage: FK / intermediate chains collapse to direct dumped-table
    edges; an un-dumped intermediate is passed through, never an endpoint."""
    config = {
        "default_parameters": {"postprocessor_args": {"_input_tables": []}},
        "tables": [
            {
                "table_name": "products",
                "dump_results": True,
                "columns": [{"column_name": "sku", "column_generator": {"args": {}}}],
                "post_processor": None,
            },
            {
                "table_name": "staging",  # un-dumped intermediate
                "dump_results": False,
                "columns": [],
                "post_processor": {
                    "udf_module": "m",
                    "udf_name": "u",
                    "udf_args": {"_input_tables": ["products"]},
                },
            },
            {
                "table_name": "orders",
                "dump_results": True,
                "columns": [],
                "post_processor": {
                    "udf_module": "m",
                    "udf_name": "u",
                    "udf_args": {"_input_tables": ["staging"]},
                },
            },
        ],
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(config))
    gd = GraphDependencies(str(path))

    rels = {(r["source"], r["target"]) for r in gd.table_relations()}
    # products feeds orders through an intermediate + a post-processor -> one direct edge
    assert ("products", "orders") in rels
    # the un-dumped intermediate is contracted away — never a relation endpoint
    assert all("staging" not in pair for pair in rels)

    # contracted edges are emitted (hidden) and the toggle ships in the HTML
    els = gd._cytoscape_elements()
    assert any(e["data"].get("kind") == "rel" for e in els)
    gd.dump_html(folder_path=str(tmp_path))
    assert "Tables only: off" in (tmp_path / "data_generator_dependencies.html").read_text()


test_data_generator_post_processor()
