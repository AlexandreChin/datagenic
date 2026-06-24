"""Scaffold a generation config from a model's neutral IR: ``Columns`` (per-column
default-generator selection) and ``Configs`` (IR → full generation config). Split
out of ``data_generator`` (package split Part 6).
"""

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import datagenic.iokit.func as func
from datagenic import generators


class Columns:
    """
    class for creating a column and generating values
    """

    def __init__(
        self,
        table_name: str,
        col_name: str,
        col_type: str,
        is_unicity_key: bool,
        number_of_unicity_keys_in_table: int,
        number_of_rows: int = 1000,
        _number_of_rows_per_batch: int = 1000,
        joins: list | None = None,
        default_cardinality: int = 50,
        default_size_of_arrays: int = 100,
    ) -> None:
        if joins is None:
            joins = []
        self.table_name = table_name
        self.col_name = col_name
        self.col_type = col_type.lower()
        self.is_unicity_key = is_unicity_key
        self.number_of_unicity_keys_in_table = number_of_unicity_keys_in_table
        self.number_of_rows = number_of_rows
        self._number_of_rows_per_batch = min(_number_of_rows_per_batch, number_of_rows)
        self.default_cardinality = min(default_cardinality, number_of_rows)
        self.default_size_of_arrays = default_size_of_arrays
        self.joins = joins

        # dealing with joins column which do not have the same names for generating consistent data
        if len(self.joins) == 0:
            self._default_prefix = f"{self.col_name}_"
        else:
            self._default_prefix = f"{self.joins[0]['col_name']}_"

        if "array" in self.col_type:
            self._is_scalar = False
        else:
            self._is_scalar = True

        ###### default generator mapping #######
        self._default_generator_obj: Any
        # scalar type column
        if self._is_scalar:
            # unicity key column
            if self.is_unicity_key:
                # unique unicity key
                if self.number_of_unicity_keys_in_table == 1:
                    match self.col_type:
                        case "string":
                            self._default_generator_obj = generators.UniqueIncrementalStrings()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                            self.default_arguments = self.default_arguments | {
                                "prefix": self._default_prefix
                            }
                        case x if "int" in x:
                            self._default_generator_obj = generators.UniqueIncrementalIntegers()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                        case "date":
                            if self.col_name == "Date":
                                self._default_generator_obj = generators.CurrentAsofDates()
                            else:
                                self._default_generator_obj = generators.UniqueIncrementalDates()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )

                        case _:
                            raise Exception(
                                f"{self.col_name}({self.col_type}) in table {self.table_name}: unknown type for defining a generator for this column as it is a unique unicity key"
                            )
                # composite unicity keys
                else:
                    match self.col_type:
                        case "string":
                            self._default_generator_obj = generators.RandomStrings()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                            self.default_arguments = self.default_arguments | {
                                "prefix": self._default_prefix
                            }
                        case x if "int" in x:
                            self._default_generator_obj = generators.RandomIntegers()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                        case "date":
                            if self.col_name == "Date":
                                self._default_generator_obj = generators.CurrentAsofDates()
                            else:
                                self._default_generator_obj = generators.RandomDates()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                            self.default_arguments = self.default_arguments
                        case x if "datetime" in x:
                            self._default_generator_obj = generators.RandomTimeStamps()
                            self.default_arguments = func.get_default_args(
                                self._default_generator_obj.call
                            )
                            self.default_arguments = self.default_arguments
                        case _:
                            raise Exception(
                                f"{self.col_name}({self.col_type}) in table {self.table_name}: unknown type for defining a generator for this column as it is a unicity key"
                            )
            # non-unicity key column
            else:
                match self.col_type:
                    case "string":
                        self._default_generator_obj = generators.RandomStrings()
                        self.default_arguments = func.get_default_args(
                            self._default_generator_obj.call
                        )
                        self.default_arguments = self.default_arguments | {
                            "prefix": self._default_prefix,
                            "cardinality": self.default_cardinality,
                        }
                    case x if "int" in x:
                        self._default_generator_obj = generators.RandomIntegers()
                        self.default_arguments = func.get_default_args(
                            self._default_generator_obj.call
                        )
                        self.default_arguments = self.default_arguments | {
                            "cardinality": self.default_cardinality
                        }
                    case "date":
                        if self.col_name == "Date":
                            self._default_generator_obj = generators.CurrentAsofDates()
                        else:
                            self._default_generator_obj = generators.RandomDates()
                        self.default_arguments = func.get_default_args(
                            self._default_generator_obj.call
                        )
                        self.default_arguments = self.default_arguments | {
                            "cardinality": self.default_cardinality
                        }
                    case x if "datetime" in x:
                        self._default_generator_obj = generators.RandomTimeStamps()
                        self.default_arguments = func.get_default_args(
                            self._default_generator_obj.call
                        )
                        self.default_arguments = self.default_arguments
                    case x if "float" in x:
                        self._default_generator_obj = generators.BrownianFloats()
                        self.default_arguments = func.get_default_args(
                            self._default_generator_obj.call
                        )
                        # | {"mu":100,"sigma":50}
                        self.default_arguments = self.default_arguments
                    case _:
                        raise Exception(
                            f"{self.col_name}({self.col_type}) in table {self.table_name}: unknown type for defining a generator for this column"
                        )
        # array type column
        else:
            match self.col_type:
                case x if "float" in x:
                    self._default_generator_obj = generators.BrownianArrayFloats()
                    self.default_arguments = func.get_default_args(self._default_generator_obj.call)
                    self.default_arguments = self.default_arguments | {
                        "size_of_arrays": self.default_size_of_arrays,
                        "mu": 1000,
                        "sigma": 500,
                    }
                case x if "string" in x:
                    self._default_generator_obj = generators.RandomArrayStrings()
                    self.default_arguments = func.get_default_args(self._default_generator_obj.call)
                    self.default_arguments = self.default_arguments | {
                        "prefix": self._default_prefix,
                        "size_of_arrays": self.default_size_of_arrays,
                        "cardinality": self.default_cardinality,
                    }
                case x if "date" in x:
                    self._default_generator_obj = generators.IncrementalArrayDates()
                    self.default_arguments = func.get_default_args(self._default_generator_obj.call)
                    self.default_arguments = self.default_arguments | {
                        "prefix": self._default_prefix,
                        "size_of_arrays": self.default_size_of_arrays,
                        "cardinality": self.default_cardinality,
                    }

                case _:
                    raise Exception(
                        f"{self.col_name}({self.col_type}) in table {self.table_name}: unknown type for defining a generator for this column"
                    )

        self.default_generator_name = self._default_generator_obj.__class__.__name__
        self.default_generator_description = self._default_generator_obj._generator_description


class Configs:
    """
    Scaffold a generation config from a model's **internal representation (IR)** — a
    flat, source-agnostic description of the tables to generate. The IR is the native
    input; other schemas are mapped in via an ``adaptor`` callable.

    Native input (the IR)::

        {
          "name": "<model name>",
          "tables": [
            {
              "name": "trades",
              "kind": "fact",                  # "fact" | "dimension"
              "keys": ["trade_id"],             # unicity-key column names
              "columns": [{"name": "trade_id", "type": "int"}, …],
              "foreign_keys": [                 # optional, fact tables
                {"column": "ccy", "references": "fx_dim.ccy"}   # references: str | [str]
              ],
            },
            …
          ],
        }

    To scaffold from another schema pass ``adaptor=fn`` where ``fn(source) -> IR`` —
    a small one-time mapper from your source format to the IR.
    """

    @staticmethod
    def _validate_model_shape(ir):
        """Fail loudly (clear error) if the IR — post-adaptor, or raw input — is
        malformed, instead of a deep KeyError mid-scaffold."""
        if not isinstance(ir, dict) or not isinstance(ir.get("name"), str):
            raise ValueError(
                "Configs IR must be a dict with a string 'name' "
                "(an adaptor must return the IR shape; see Configs docstring)."
            )
        if not isinstance(ir.get("tables"), list):
            raise ValueError("Configs IR must have a 'tables' list (see Configs docstring).")
        for i, table in enumerate(ir["tables"]):
            if (
                not isinstance(table, dict)
                or "name" not in table
                or not isinstance(table.get("columns"), list)
            ):
                raise ValueError(f"Configs IR tables[{i}] needs a 'name' and a 'columns' list.")
            for col in table["columns"]:
                if not (isinstance(col, dict) and "name" in col and "type" in col):
                    raise ValueError(
                        f"Configs IR table '{table.get('name')}' columns each need 'name' and 'type'."
                    )

    @staticmethod
    def _ir_to_internal(ir):
        """Expand the flat IR into the structured shape the scaffolder reads: fact
        tables + composite dicts, with foreign_keys mapped to dict-composite joins so
        FK enforcement is built. (All dimensions collapse to composite dicts.)"""
        internal = {
            "name": ir["name"],
            "tables": {
                "fact": [],
                "dict": {"composite": [], "denormalized": []},
                "zip": {"external": [], "internal": []},
            },
            "joins": {"dict": {"composite": [], "denormalized": []}, "zip": []},
        }
        for table in ir["tables"]:
            entry = {
                "name": table["name"],
                "unicityKey": list(table.get("keys", [])),
                "columns": [{"name": c["name"], "dataType": c["type"]} for c in table["columns"]],
            }
            if table.get("kind") == "fact":
                internal["tables"]["fact"].append(entry)
                for fk in table.get("foreign_keys", []):
                    refs = fk["references"]
                    for ref in refs if isinstance(refs, list) else [refs]:
                        ref_table, _, ref_col = ref.partition(".")
                        internal["joins"]["dict"]["composite"].append(
                            {
                                "factTable": table["name"],
                                "dictionary": ref_table,
                                "joinKeys": [{"tableColumn": fk["column"], "dictColumn": ref_col}],
                            }
                        )
            else:
                internal["tables"]["dict"]["composite"].append(entry)
        return internal

    def __init__(
        self,
        input: object = None,
        server: str | None = None,
        output_format: str = "csv",
        storage_type: str = "local",
        output_folder_path: str = "samples/data_generator/fake_data",
        number_of_batches: int = 2,
        offset_batches: int = 0,
        number_of_rows_in_fact_tables: int = 2500,
        number_of_rows_in_non_fact_tables: int = 100,
        start_date: str = "2024-01-01",
        end_date: str = "2024-01-02",
        seed: int = 0,
        enforce_fk: bool = True,
        template_postprocessors: bool = False,
        adaptor: Callable[[dict], dict] | None = None,
    ):
        # Load the source (dict, JSON path, or a plain string → empty IR), then run
        # the optional ``adaptor`` to map it into the IR. The IR is the native input;
        # to scaffold from another schema pass ``adaptor=fn`` where ``fn(source) -> IR``.
        self.input = input
        self.adaptor = adaptor
        if isinstance(input, dict):
            source = input
        elif isinstance(input, str) and input[-5:] == ".json":
            with open(input) as f:
                source = json.load(f)
        elif isinstance(input, str):
            source = {"name": input, "tables": []}  # empty IR skeleton
        else:
            raise TypeError(f"Configs input must be a dict or a string, got {type(input).__name__}")
        self._ir = adaptor(source) if adaptor is not None else source
        self._validate_model_shape(self._ir)
        # expand the IR into the structured shape the scaffolder reads
        self._internal_model = self._ir_to_internal(self._ir)
        self._module_name = self._internal_model["name"]

        self.enforce_fk = enforce_fk
        self.template_postprocessors = template_postprocessors
        self.server = server
        self.module_name = self._module_name
        self.output_format = output_format
        self.storage_type = storage_type
        self.output_folder_path = output_folder_path
        self.number_of_rows_in_fact_tables = number_of_rows_in_fact_tables
        self.number_of_rows_in_non_fact_tables = number_of_rows_in_non_fact_tables
        self.number_of_batches = number_of_batches
        self.offset_batches = offset_batches
        self.start_date = start_date
        self.end_date = end_date
        self.seed = seed

        self.config_template: dict[str, Any] = {
            "default_parameters": {
                "output_format": self.output_format,
                "storage_type": self.storage_type,
                "output_folder_path": self.output_folder_path,
                "tables_prefix": "",
                "start_date": self.start_date,
                "end_date": self.end_date,
                "number_of_batches": self.number_of_batches,
                "offset_batches": self.offset_batches,
                "postprocessor_args": {"fk_enforcement": self._build_fk_enforcement()},
                "seed": self.seed,
            },
            "tables": [],
        }

        # external ziptables
        for table in self._internal_model["tables"]["zip"]["external"]:
            number_of_unicity_keys_in_table = len(table["technical"]["orderingKey"])

            temp_table: dict[str, Any] = {
                "table_name": table["name"],
                "number_of_rows": self.number_of_rows_in_non_fact_tables,
                "post_processor": {},
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }

            for col in table["columns"]:
                temp_col = {
                    "col_name": col["name"],
                    "col_type": col["dataType"],
                    "column_generator": {},
                }

                # unicity key
                if col["name"] in table["technical"]["orderingKey"]:
                    temp_col["is_unicity_key"] = True
                else:
                    temp_col["is_unicity_key"] = False

                # generator
                # TO DO: joins
                temp_generator = Columns(
                    table_name=temp_table["table_name"],
                    col_name=temp_col["col_name"],
                    col_type=temp_col["col_type"],
                    is_unicity_key=temp_col["is_unicity_key"],
                    number_of_unicity_keys_in_table=number_of_unicity_keys_in_table,
                    _number_of_rows_per_batch=math.ceil(
                        self.number_of_rows_in_non_fact_tables / self.number_of_batches
                    ),
                )
                temp_col["column_generator"] = {}
                temp_col["column_generator"]["udf_name"] = temp_generator.default_generator_name
                temp_col["column_generator"]["udf_module"] = "datagenic.generators"
                temp_col["column_generator"]["args"] = temp_generator.default_arguments
                temp_col["column_generator"]["description"] = (
                    temp_generator.default_generator_description
                )

                temp_table["columns"].append(temp_col)

            self.config_template["tables"].append(temp_table)

        # composite dictionaries
        for table in self._internal_model["tables"]["dict"]["composite"]:
            number_of_unicity_keys_in_table = len(table["unicityKey"])
            temp_table: dict[str, Any] = {
                "table_name": table["name"],
                "number_of_rows": self.number_of_rows_in_non_fact_tables,
                "post_processor": {},
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }

            for col in table["columns"]:
                temp_col = {"col_name": col["name"], "col_type": col["dataType"]}

                # unicity key
                if col["name"] in table["unicityKey"]:
                    temp_col["is_unicity_key"] = True
                else:
                    temp_col["is_unicity_key"] = False

                # generator
                # TO DO: joins
                temp_generator = Columns(
                    table_name=temp_table["table_name"],
                    col_name=temp_col["col_name"],
                    col_type=temp_col["col_type"],
                    is_unicity_key=temp_col["is_unicity_key"],
                    number_of_unicity_keys_in_table=number_of_unicity_keys_in_table,
                    _number_of_rows_per_batch=math.ceil(
                        self.number_of_rows_in_non_fact_tables / self.number_of_batches
                    ),
                )

                temp_col["column_generator"] = {}
                temp_col["column_generator"]["udf_name"] = temp_generator.default_generator_name
                temp_col["column_generator"]["udf_module"] = "datagenic.generators"
                temp_col["column_generator"]["args"] = temp_generator.default_arguments
                temp_col["column_generator"]["description"] = (
                    temp_generator.default_generator_description
                )

                temp_table["columns"].append(temp_col)

            self.config_template["tables"].append(temp_table)

        # hashed dictionaries
        for table in self._internal_model["tables"]["dict"]["denormalized"]:
            number_of_unicity_keys_in_table = len(table["joinKey"])
            temp_table: dict[str, Any] = {
                "table_name": table["name"],
                "number_of_rows": self.number_of_rows_in_non_fact_tables,
                "post_processor": {},
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }

            for col in table["columns"]:
                temp_col = {"col_name": col["name"], "col_type": col["dataType"]}

                # unicity key
                if col["name"] in table["joinKey"]:
                    temp_col["is_unicity_key"] = True
                else:
                    temp_col["is_unicity_key"] = False

                # generator
                # TO DO: joins
                temp_generator = Columns(
                    table_name=temp_table["table_name"],
                    col_name=temp_col["col_name"],
                    col_type=temp_col["col_type"],
                    is_unicity_key=temp_col["is_unicity_key"],
                    number_of_unicity_keys_in_table=number_of_unicity_keys_in_table,
                    _number_of_rows_per_batch=math.ceil(
                        self.number_of_rows_in_non_fact_tables / self.number_of_batches
                    ),
                )
                temp_col["column_generator"] = {}
                temp_col["column_generator"]["udf_name"] = temp_generator.default_generator_name
                temp_col["column_generator"]["udf_module"] = "datagenic.generators"
                temp_col["column_generator"]["args"] = temp_generator.default_arguments
                temp_col["column_generator"]["description"] = (
                    temp_generator.default_generator_description
                )

                temp_table["columns"].append(temp_col)

            self.config_template["tables"].append(temp_table)

        # facttables
        for table in self._internal_model["tables"]["fact"]:
            number_of_unicity_keys_in_table = len(table["unicityKey"])
            temp_table: dict[str, Any] = {
                "table_name": table["name"],
                "number_of_rows": self.number_of_rows_in_fact_tables,
                "post_processor": {},
                "columns": [],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }

            # internal zip labels
            for internal_zip in self._internal_model["tables"]["zip"]["internal"]:
                if internal_zip["factTable"]["name"] == table["name"]:
                    table["columns"].append(
                        {
                            "name": internal_zip["zipTable"]["label"],
                            "dataType": internal_zip["zipTable"]["dataType"],
                        }
                    )

            for col in table["columns"]:
                temp_col = {"col_name": col["name"], "col_type": col["dataType"]}

                # unicity key
                if col["name"] in table["unicityKey"]:
                    temp_col["is_unicity_key"] = True
                else:
                    temp_col["is_unicity_key"] = False

                # generator
                temp_generator = Columns(
                    table_name=temp_table["table_name"],
                    col_name=temp_col["col_name"],
                    col_type=temp_col["col_type"],
                    is_unicity_key=temp_col["is_unicity_key"],
                    number_of_unicity_keys_in_table=number_of_unicity_keys_in_table,
                    _number_of_rows_per_batch=math.ceil(
                        self.number_of_rows_in_non_fact_tables / self.number_of_batches
                    ),
                )
                temp_col["column_generator"] = {}
                temp_col["column_generator"]["udf_name"] = temp_generator.default_generator_name
                temp_col["column_generator"]["udf_module"] = "datagenic.generators"
                temp_col["column_generator"]["args"] = temp_generator.default_arguments
                temp_col["column_generator"]["description"] = (
                    temp_generator.default_generator_description
                )

                temp_table["columns"].append(temp_col)

            # wire a KeyEnforcer postprocessor so fact-table FK columns draw from
            # their referenced tables' key pools (referential integrity)
            if self.enforce_fk:
                rules, input_tables = self._fk_rules(temp_table["table_name"])
                if rules:
                    temp_table["post_processor"] = {
                        "udf_module": "datagenic.postprocessors",
                        "udf_name": "KeyEnforcer",
                        "udf_args": {"_input_tables": input_tables, "rules": rules},
                    }

            self.config_template["tables"].append(temp_table)

        if self.template_postprocessors:
            self._wire_template_postprocessors()

    def _wire_template_postprocessors(self) -> None:
        """Give every table a no-op postprocessor template (in place).

        Only fills tables whose ``post_processor`` is empty, so a fact table's
        auto-wired KeyEnforcer (FK enforcement) is preserved. Each filled table
        points at the shipped ``datagenic.postprocessors.TemplatePostProcessor`` —
        copy and rename it to customize a specific table.
        """
        for table in self.config_template["tables"]:
            if not table.get("post_processor"):
                table["post_processor"] = {
                    "udf_module": "datagenic.postprocessors",
                    "udf_name": "TemplatePostProcessor",
                    "udf_args": {},
                }

    def _fk_rules(self, fact_table_name: str):
        """Translate the fk_enforcement map for a fact table into KeyEnforcer rules.

        Each FK column becomes a {"col", "from"} sample rule drawing from the
        referenced table's key pool (a list of refs => intersection). The asof
        `Date` column is excluded — it is the partition date, not a foreign key.
        """
        fk = self.config_template["default_parameters"]["postprocessor_args"].get(
            "fk_enforcement", {}
        )
        cols = {k: v for k, v in fk.get(fact_table_name, {}).items() if k != "Date"}
        rules = []
        referenced: list[str] = []
        for col, ref_list in cols.items():
            rules.append({"col": col, "from": ref_list if len(ref_list) > 1 else ref_list[0]})
            referenced.extend(ref.split(".")[0] for ref in ref_list)
        return rules, sorted(set(referenced))

    def _build_fk_enforcement(self) -> dict:
        """Build fk_enforcement from the IR's join definitions.

        Covers composite dict joins (all join keys including Date), denormalized
        dict joins (resolved via the dict's joinKey), and external zip joins.
        If the same fact column joins multiple reference tables, refs are
        appended — ForeignKeyEnforcer will use the intersection of all pools.
        """
        fk: dict = {"enabled": True}

        # Dict composite joins
        for join in self._internal_model.get("joins", {}).get("dict", {}).get("composite", []):
            fact_table = join["factTable"]
            dict_table = join["dictionary"]
            for jk in join.get("joinKeys", []):
                ref = f"{dict_table}.{jk['dictColumn']}"
                col_refs = fk.setdefault(fact_table, {}).setdefault(jk["tableColumn"], [])
                if ref not in col_refs:
                    col_refs.append(ref)

        # Dict denormalized joins — join key is shared by both tables under the dict's joinKey
        denorm_dicts = {
            t["name"]: t
            for t in self._internal_model.get("tables", {}).get("dict", {}).get("denormalized", [])
        }
        for join in self._internal_model.get("joins", {}).get("dict", {}).get("denormalized", []):
            fact_table = join["factTable"]
            dict_table = join["dictionary"]
            for join_key in denorm_dicts.get(dict_table, {}).get("joinKey", []):
                ref = f"{dict_table}.{join_key}"
                col_refs = fk.setdefault(fact_table, {}).setdefault(join_key, [])
                if ref not in col_refs:
                    col_refs.append(ref)

        # External zip tables only (internal zips have no standalone table)
        external_zip_names = {
            t["name"]
            for t in self._internal_model.get("tables", {}).get("zip", {}).get("external", [])
        }
        for join in self._internal_model.get("joins", {}).get("zip", []):
            zip_table_name = join["zipTable"]["name"]
            if zip_table_name not in external_zip_names:
                continue
            fact_table = join["factTable"]["name"]
            for jk in join.get("joinKeys", []):
                ref = f"{zip_table_name}.{jk['zip']}"
                col_refs = fk.setdefault(fact_table, {}).setdefault(jk["table"], [])
                if ref not in col_refs:
                    col_refs.append(ref)

        return fk

    def dump(
        self, file_path: str = "samples/data_generation_config/data_generation_config.json"
    ) -> None:
        # create directory
        Path(file_path[: file_path.rfind("/")]).mkdir(parents=True, exist_ok=True)

        # clear file
        open(file_path, "w").close()

        with open(file_path, "w") as f:
            f.write(json.dumps(self.config_template, sort_keys=False, indent=4))
        print(f"config file for data generation available here:{file_path}")
