"""Build a mermaid dependency flowchart from a generation config
(``GraphDependencies``). Split out of ``data_generator`` (package split Part 6).
"""

import json
from pathlib import Path


class GraphDependencies:
    _FILE_EXTENSIONS = (".csv", ".json", ".parquet", ".yaml", ".xlsx")

    @classmethod
    def _file_path_args(cls, udf_args: dict) -> list:
        """Arg values that look like input file paths (for file-dependency edges)."""
        return [
            value
            for value in udf_args.values()
            if isinstance(value, str) and value.endswith(cls._FILE_EXTENSIONS)
        ]

    def __init__(self, config_file_path: str):
        if config_file_path[-5:] == ".json":
            with open(config_file_path) as f:
                self._config_file_path = json.load(f)
        else:
            raise Exception("unrecognized config file")

        self.entities = []
        self.relationships = []

        for table in self._config_file_path["tables"]:
            table_name = table["table_name"]
            dump_results = table["dump_results"]
            table_class = "table" if dump_results else "intermediate_results"

            # ── column-generated tables ──────────────────────────────────────
            # One aggregate node represents all of a table's column generators,
            # so a wide fact table doesn't explode into a node per column.
            if table["columns"]:
                self.entities.append(f"{table_name}[({table_name})]:::{table_class}")

                column_gen = f"{table_name}.column_generators"
                self.entities.append(f"{column_gen}([column generators]):::generator")
                self.relationships.append(f"{column_gen} ---> {table_name}")

                # files read by any column generator (e.g. ValuesFromFile)
                for column in table["columns"]:
                    for path in self._file_path_args(
                        column.get("column_generator", {}).get("args", {})
                    ):
                        self.entities.append(f"{path}:::file")
                        self.relationships.append(f"{path} ---> {column_gen}")

                # dynamic row count: this table's size derives from another table
                number_of_rows = table.get("number_of_rows")
                if isinstance(number_of_rows, dict) and "from_table" in number_of_rows:
                    self.relationships.append(f"{number_of_rows['from_table']} ---> {table_name}")

            if table["post_processor"]:
                udf_module = table["post_processor"]["udf_module"]
                udf_name = table["post_processor"]["udf_name"]
                udf_args = table["post_processor"]["udf_args"]

                ##### entities #####
                # add generator to entities
                self.entities.append(
                    f"{table_name}.{udf_module}.{udf_name}([{udf_module}.{udf_name}]):::generator"
                )

                # add table to entities
                self.entities.append(f"{table_name}[({table_name})]:::{table_class}")

                ##### files (entities + dependencies) #####
                for path in self._file_path_args(udf_args):
                    self.entities.append(f"{path}:::file")
                    self.relationships.append(f"{path} ---> {table_name}.{udf_module}.{udf_name}")

                self.relationships.append(f"{table_name}.{udf_module}.{udf_name} ---> {table_name}")

                # add input-table dependencies
                if (
                    "_input_tables"
                    in self._config_file_path["default_parameters"]["postprocessor_args"]
                ):
                    for table in self._config_file_path["default_parameters"]["postprocessor_args"][
                        "_input_tables"
                    ]:
                        self.relationships.append(
                            f"{table} ---> {table_name}.{udf_module}.{udf_name}"
                        )
                        self.relationships.append(
                            f"{table_name}.{udf_module}.{udf_name} ---> {table_name}"
                        )

                if "_input_tables" in udf_args:
                    for table in udf_args["_input_tables"]:
                        self.relationships.append(
                            f"{table} ---> {table_name}.{udf_module}.{udf_name}"
                        )
                        self.relationships.append(
                            f"{table_name}.{udf_module}.{udf_name} ---> {table_name}"
                        )

        # remove duplicates
        self.entities = list(set(self.entities))
        self.relationships = list(set(self.relationships))

    def dump_file(self, folder_path: str = "samples/graphs", mermaid_header: bool = True) -> None:
        """
        create a mermaid ErDiagram markdown file
        """

        md_file_path = f"{folder_path}/data_generator_dependencies.md"

        # create directory
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        # clear md file
        open(md_file_path, "w").close()

        with open(md_file_path, "a") as f:
            # headers & title
            if mermaid_header:
                f.write("```mermaid\n")
            f.write("---\n")
            f.write("title: Data Generator Workflow\n")
            f.write("---\n\n")

            f.write("flowchart LR\n\n")

            f.write("""%% Colors %%
            classDef table fill:lightgreen,stroke:#000,stroke-width:3px
            classDef intermediate_results fill:lightgrey,stroke:#000,stroke-dasharray: 5
            classDef file fill:ligntblue,stroke:#000,stroke-width:1px
            classDef generator fill:lightpink,stroke:#000,stroke-width:1px\n\n""")

            # legend
            f.write("""%% Legend
                    subgraph legend
                    direction TB
                        file:::file
                        generator([generator]):::generator
                        intermediate_results[(intermediate_results)]:::intermediate_results
                        tables[(table)]:::table
                    end\n\n""")

            # entities
            f.write("%% Entities\n")
            for elt in self.entities:
                f.write(f"{elt}\n\n")
            f.write("\n")

            # relationships
            f.write("%% Relationships\n")
            for elt in self.relationships:
                f.write(f"{elt}\n\n")

            # eof
            if mermaid_header:
                f.write("\n```")
            f.close()

        print(f"flowchart file available here:{md_file_path}")
