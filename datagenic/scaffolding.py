"""Scaffold a generation config from a model's neutral IR: ``Columns`` (per-column
default-generator selection) and ``Configs`` (IR → full generation config). Split
out of ``data_generator`` (package split Part 6).
"""

import importlib
import json
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import datagenic.iokit.func as func
from datagenic import generators
from datagenic.semantics import SEMANTIC_GENERATORS, build_generator, resolve_inferrer


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
              "type": "fact",                  # "fact" or "referential" (optional, default "referential")
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
            table_type = table.get("type", "referential")
            if table_type not in ("fact", "referential"):
                raise ValueError(
                    f"Configs IR table '{table['name']}': type must be 'fact' or 'referential' "
                    f"(got {table_type!r}); omit it to default to 'referential'."
                )
            for col in table["columns"]:
                if not (isinstance(col, dict) and "name" in col and "type" in col):
                    raise ValueError(
                        f"Configs IR table '{table.get('name')}' columns each need 'name' and 'type'."
                    )

    @staticmethod
    def _ir_to_internal(ir):
        """Expand the flat IR into the structured shape the scaffolder reads: fact
        tables + composite dicts, with foreign_keys mapped to dict-composite joins so
        FK enforcement is built. (All referential tables collapse to composite dicts.)"""
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
                "columns": [
                    {
                        "name": c["name"],
                        "dataType": c["type"],
                        # optional null-injection hints, carried through to null_spec
                        "null_rate": c.get("null_rate"),
                        "null_when": c.get("null_when"),
                    }
                    for c in table["columns"]
                ],
            }
            if table.get("type") == "fact":
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
        semantic_inferrer=None,
        samples=None,
        rolling_book: bool = False,
        rolling_scenarios: int = 1,
        reference_rows: str | int | None = None,
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

        # Semantic inference (scaffold-time only): map each column to a domain
        # generator automatically, so the user need not hand-tag the IR. The
        # resulting config is fully hand-editable; see _inferred annotations.
        # ``samples`` (optional) lets a value-based inferrer read real values.
        self._inferrer = resolve_inferrer(semantic_inferrer)
        self._sample_data = self._normalize_samples(samples)
        self.rolling_book = rolling_book
        self.reference_rows = reference_rows
        self._needs_unique_by_construction = False
        self.rolling_scenarios = rolling_scenarios
        self._semantics: dict = {}
        self._compute_semantics()
        if self.rolling_book:
            self._apply_rolling_book()
        # per-(fact, fk-column) KeyEnforcer rule options (null_rate / fanout) from the IR
        self._fk_options = self._compute_fk_options()

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

        # Rolling-book generators key off row index == slot; the post-generation key
        # sort would scramble that mapping across dates (recall regenerates row by row),
        # so skip it. Row *order* changes only; the row set and key uniqueness do not.
        if self.rolling_book:
            self.config_template["default_parameters"]["scale_skip_global_key_sort"] = True

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

                self._apply_semantic(temp_col, table["name"])
                self._apply_null_spec(temp_col, col)
                temp_table["columns"].append(temp_col)

            self._order_for_coherence(temp_table["columns"])
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

                self._apply_semantic(temp_col, table["name"])
                self._apply_null_spec(temp_col, col)
                temp_table["columns"].append(temp_col)

            self._order_for_coherence(temp_table["columns"])
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

                self._apply_semantic(temp_col, table["name"])
                self._apply_null_spec(temp_col, col)
                temp_table["columns"].append(temp_col)

            self._order_for_coherence(temp_table["columns"])
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

                self._apply_semantic(temp_col, table["name"])
                self._apply_null_spec(temp_col, col)
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

            self._order_for_coherence(temp_table["columns"])
            self.config_template["tables"].append(temp_table)

        # Size reference (non-fact) tables to their strict minimal grain when
        # reference_rows="auto": number_of_rows = ∏ key cardinalities, and make them
        # broadcast dimensions (single_batch/single_date) so the full grain is produced
        # once and reused by every fact FK draw. Facts keep their explicit row count.
        fact_names = {t["name"] for t in self._internal_model["tables"]["fact"]}
        for temp_table in self.config_template["tables"]:
            if temp_table["table_name"] not in fact_names:
                self._size_reference_table(temp_table)
        if self._needs_unique_by_construction:
            self.config_template["default_parameters"]["scale_unique_by_construction"] = True

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

    def _normalize_samples(self, samples) -> dict:
        """Normalise the optional ``samples`` input to ``{table: {col: [values]}}``
        for the inferrer. ``samples`` may be a dict (``table -> frame | path``) or
        a directory string (matched to ``<table>.parquet`` / ``<table>.csv`` per
        IR table). Frames are read frame-agnostically; values are capped per column.
        """
        if not samples:
            return {}
        sources = self._resolve_sample_sources(samples)
        return {
            tname: self._frame_to_coldict(self._read_frame(src) if isinstance(src, str) else src)
            for tname, src in sources.items()
        }

    def _resolve_sample_sources(self, samples) -> dict:
        if isinstance(samples, dict):
            return samples
        # a directory: match <table>.parquet / <table>.csv for each IR table
        import os

        out: dict = {}
        for table in self._ir["tables"]:
            for ext in ("parquet", "csv"):
                path = os.path.join(samples, f"{table['name']}.{ext}")
                if os.path.exists(path):
                    out[table["name"]] = path
                    break
        return out

    @staticmethod
    def _read_frame(path: str):
        import polars as pl

        if path.endswith(".parquet"):
            return pl.read_parquet(path)
        if path.endswith(".csv"):
            return pl.read_csv(path)
        raise ValueError(f"unsupported sample file '{path}' (expected .parquet or .csv)")

    @staticmethod
    def _frame_to_coldict(frame, cap: int = 2000) -> dict:
        """Frame-agnostic {col: [values]} (polars or pandas), capped to ``cap`` rows."""
        columns = getattr(frame, "columns", None)
        if columns is None:
            return {}
        out: dict = {}
        for col in columns:
            series = frame[col]
            to_list = getattr(series, "to_list", None) or getattr(series, "tolist", None)
            values = to_list() if to_list is not None else list(series)
            out[col] = values[:cap]
        return out

    def _compute_semantics(self) -> None:
        """Infer (or read explicit) per-column semantics keyed by (table, column).

        Runs the configured inferrer over each table's columns; an explicit IR
        ``semantic`` on a column overrides inference (confidence 1.0). Only
        columns whose semantic maps to a known generator are kept. No-op when no
        inferrer is set and no column carries an explicit ``semantic``.
        """
        has_explicit = any(c.get("semantic") for t in self._ir["tables"] for c in t["columns"])
        if self._inferrer is None and not has_explicit:
            return
        for table in self._ir["tables"]:
            tname = table["name"]
            cols = [{"name": c["name"], "type": c.get("type")} for c in table["columns"]]
            inferred = (
                self._inferrer(tname, cols, samples=self._sample_data.get(tname))
                if self._inferrer
                else {}
            )
            for col in table["columns"]:
                rec = inferred.get(col["name"])
                if col.get("semantic"):  # explicit IR hint wins over inference
                    rec = {
                        "semantic": col["semantic"],
                        "confidence": 1.0,
                        "parent": col.get("parent"),
                        "alternatives": [],
                        "by": "explicit",
                    }
                if rec and rec["semantic"] in SEMANTIC_GENERATORS:
                    self._semantics[(tname, col["name"])] = rec

    @staticmethod
    def _key_cardinality(temp_col: dict):
        """Distinct values a unicity-key column will produce, and whether that count is
        *bounded* (a real domain: fixed-vocabulary generator / list_of_values) or *free*
        (a synthetic id whose cardinality is a convention). Mirrors the engine's
        budgeting resolution so scaffold-time sizing agrees with generation."""
        cg = temp_col["column_generator"]
        args = cg.get("args", {})
        try:
            cls = getattr(importlib.import_module(cg["udf_module"]), cg["udf_name"])
            inst = cls()
        except Exception:
            inst = None
        if inst is not None and hasattr(inst, "cardinality"):
            try:
                return int(inst.cardinality(**args)), "bounded"
            except TypeError:
                return int(inst.cardinality()), "bounded"
        if args.get("list_of_values"):
            return len(args["list_of_values"]), "bounded"
        return None, "free"

    def _resolve_grain(self, keys: list[dict]) -> tuple[int, dict, bool]:
        """Strict minimal grain for a reference table: one row per distinct key tuple.

        ``rows = ∏ cardinality(key)``. Bounded keys use their real domain; free keys use
        the ``number_of_rows_in_non_fact_tables`` budget — and when *two or more* keys are
        free (the only case cardinality is equalized) that budget is split geometrically so
        the product stays ≈ the budget instead of blowing up as budget^k. Returns the row
        count, the per-key cardinality to stamp on each generator, and whether the table
        needs unique-by-construction enumeration (any composite key)."""
        budget = self.number_of_rows_in_non_fact_tables
        cards: dict = {}
        bounded_product = 1
        free = []
        for col in keys:
            card, kind = self._key_cardinality(col)
            if kind == "bounded":
                cards[col["col_name"]] = card
                bounded_product *= card
            else:
                free.append(col)
        if not free:
            rows = bounded_product
        elif len(free) == 1:
            cards[free[0]["col_name"]] = budget
            rows = bounded_product * budget
        else:
            per = max(1, round(budget ** (1 / len(free))))
            for col in free:
                cards[col["col_name"]] = per
            rows = bounded_product * per ** len(free)
        return rows, cards, len(keys) > 1

    def _size_reference_table(self, temp_table: dict) -> None:
        """Set a reference table's row count from its key cardinalities (``reference_rows``
        = "auto"), stamp each key's cardinality, and mark it a broadcast dimension. An
        explicit ``reference_rows`` int sets a flat count; ``None`` (default) leaves the
        legacy flat ``number_of_rows_in_non_fact_tables`` untouched. No keys → flat default."""
        if self.reference_rows is None:
            return  # opt-in only: preserve legacy behaviour (golden-safe)
        if self.reference_rows != "auto":
            temp_table["number_of_rows"] = self.reference_rows
            return
        keys = [c for c in temp_table["columns"] if c.get("is_unicity_key")]
        if not keys:
            return  # nothing to infer from — leave number_of_rows_in_non_fact_tables
        rows, cards, needs_enum = self._resolve_grain(keys)
        for col in keys:
            if col["col_name"] in cards:
                col["column_generator"].setdefault("args", {})["cardinality"] = cards[
                    col["col_name"]
                ]
        temp_table["number_of_rows"] = rows
        temp_table["single_batch"] = True
        temp_table["single_date"] = True
        if needs_enum:
            self._needs_unique_by_construction = True
        # A lone bounded-vocabulary key is sampled (not enumerated), so its distinct count
        # can fall short of the domain — the one case the exact grain isn't guaranteed.
        if len(keys) == 1 and self._key_cardinality(keys[0])[1] == "bounded":
            warnings.warn(
                f"table '{temp_table['table_name']}': sole unicity key is a bounded "
                f"vocabulary — generated distinct values may fall below its cardinality "
                f"({rows}); use a composite key or an explicit number_of_rows for an exact grain.",
                stacklevel=2,
            )

    def _apply_rolling_book(self) -> None:
        """Turn every trade table into a rolling book: remap the inferred trade_date /
        tenor / maturity_date semantics to their Rolling* equivalents and map the single
        unicity key to a rolling trade id. Requires an inferrer (heuristic) to have tagged
        those columns. On each date the book is a constant-size set of live trades; matured
        trades roll to freshly-booked ones with new ids. See RollingTradeDate & co."""
        remap = {
            "trade_date": "rolling_trade_date",
            "tenor": "rolling_tenor",
            "maturity_date": "rolling_maturity_date",
        }
        for table in self._ir["tables"]:
            tname = table["name"]
            # the column carrying the (rolling) trade date, used as the coherence parent
            trade_date_col = next(
                (
                    c["name"]
                    for c in table["columns"]
                    if self._semantics.get((tname, c["name"]), {}).get("semantic") == "trade_date"
                ),
                None,
            )
            if trade_date_col is None:  # not a trade table (or inference is off) -> leave it
                continue
            for col in table["columns"]:
                rec = self._semantics.get((tname, col["name"]))
                if rec and rec["semantic"] in remap:
                    rec["semantic"] = remap[rec["semantic"]]
                    rec["by"] = "rolling_book"
                    # trade_date & tenor are schedule roots; maturity reads trade_date
                    rec["parent"] = (
                        trade_date_col if rec["semantic"] == "rolling_maturity_date" else None
                    )
            # The book's identity is a per-slot trade id that rolls (new id on each
            # roll). Support composite keys, not just a single key: map an id-like key
            # column (or the first key that isn't a rolling-schedule column) to
            # rolling_trade_id. It is unique per slot, so the composite key tuple is
            # unique BY CONSTRUCTION -> the engine's per-unit dedup is a no-op and the
            # book keeps its constant size (a shrunk frame would desync recall()).
            keys = table.get("keys", [])
            schedule_cols = {trade_date_col} | {
                c["name"]
                for c in table["columns"]
                if self._semantics.get((tname, c["name"]), {}).get("semantic")
                in ("rolling_tenor", "rolling_maturity_date")
            }
            free_keys = [k for k in keys if k not in schedule_cols]
            id_key = next(
                (k for k in free_keys if k.lower().endswith("id") or "ref" in k.lower()),
                free_keys[0] if free_keys else None,
            )
            scenario_keys = [k for k in free_keys if k != id_key]

            # Scenario dimension (rolling_scenarios=m): repeat each position across m
            # scenario rows so the trade id recurs while (rolling_trade_id, scenario)
            # stays unique. Needs exactly one non-id key to carry the scenario labels;
            # otherwise fall back to one row per position (m == 1).
            m = max(1, int(getattr(self, "rolling_scenarios", 1) or 1))
            if m > 1 and len(scenario_keys) != 1:
                m = 1
            if m > 1:
                # tile the schedule roots across the scenario rows (maturity reads the
                # tiled trade_date/tenor from the frame, so it needs no repeat itself)
                for col in table["columns"]:
                    rec = self._semantics.get((tname, col["name"]))
                    if rec and rec.get("semantic") in ("rolling_trade_date", "rolling_tenor"):
                        rec["args"] = {**rec.get("args", {}), "repeat": m}
                self._semantics[(tname, scenario_keys[0])] = {
                    "semantic": "rolling_scenario",
                    "confidence": 1.0,
                    "parent": None,
                    "alternatives": [],
                    "by": "rolling_book",
                    "args": {"count": m},
                }
            if id_key is not None:
                self._semantics[(tname, id_key)] = {
                    "semantic": "rolling_trade_id",
                    "confidence": 1.0,
                    "parent": trade_date_col,
                    "alternatives": [],
                    "by": "rolling_book",
                    "args": {"repeat": m} if m > 1 else {},
                }

    def _apply_semantic(self, temp_col: dict, table_name: str) -> None:
        """Override a column's generator with the one inferred for its semantic,
        recording the inference under ``_inferred`` for manual review/fine-tuning.
        ``_inferred`` sits beside ``column_generator`` and is ignored by the engine.
        """
        rec = self._semantics.get((table_name, temp_col["col_name"]))
        if not rec:
            return
        cg = build_generator(rec["semantic"], rec.get("parent"), rec.get("args"))
        cg["description"] = (
            f"inferred semantic '{rec['semantic']}' "
            f"(confidence {rec['confidence']:.2f}, by {rec['by']})"
        )
        temp_col["column_generator"] = cg
        temp_col["_inferred"] = {
            "semantic": rec["semantic"],
            "confidence": rec["confidence"],
            "parent": rec.get("parent"),
            "alternatives": rec.get("alternatives", []),
            "by": rec["by"],
        }

    @staticmethod
    def _apply_null_spec(temp_col: dict, col: dict) -> None:
        """Wire a column's ``null_spec`` from IR hints (``null_rate`` / ``null_when``).

        No-op when neither hint is present (so existing configs are unchanged). The
        engine applies the spec at generation time; nulls on a unicity key are
        rejected by ``Generator.validate()``.
        """
        rate = col.get("null_rate")
        when = col.get("null_when")
        if rate is None and not when:
            return
        spec: dict = {}
        if rate is not None:
            spec["rate"] = rate
        if when:
            spec["when"] = when
        temp_col["null_spec"] = spec

    @staticmethod
    def _order_for_coherence(columns: list[dict]) -> None:
        """Stably reorder so any column whose inferred parent is in the same table
        comes after that parent: the engine builds columns top-to-bottom and a
        hierarchy generator reads only the columns built before it. Depth follows the
        full parent chain, so multi-level chains (tenor -> trade_date -> maturity_date)
        order correctly, not just one level. No-op when no column has an ``_inferred``
        parent (stable sort with equal keys)."""
        parent_of = {c["col_name"]: (c.get("_inferred") or {}).get("parent") for c in columns}

        def _depth(col: dict) -> int:
            depth, seen = 0, set()
            parent = parent_of.get(col["col_name"])
            while parent and parent in parent_of and parent not in seen:
                seen.add(parent)  # guard against a cyclic parent link
                depth += 1
                parent = parent_of.get(parent)
            return depth

        columns.sort(key=_depth)

    def _fk_rules(self, fact_table_name: str):
        """Translate the fk_enforcement map for a fact table into KeyEnforcer rules.

        Each FK column becomes a {"col", "from"} sample rule drawing from the
        referenced table's key pool (a list of refs => intersection). The asof
        `Date` column is excluded — it is the partition date, not a foreign key.
        Optional per-FK ``null_rate`` / ``fanout`` hints (from the IR) are attached to
        the rule so KeyEnforcer can make the FK nullable or shape its multiplicity.
        """
        fk = self.config_template["default_parameters"]["postprocessor_args"].get(
            "fk_enforcement", {}
        )
        cols = {k: v for k, v in fk.get(fact_table_name, {}).items() if k != "Date"}
        rules = []
        referenced: list[str] = []
        for col, ref_list in cols.items():
            rule = {"col": col, "from": ref_list if len(ref_list) > 1 else ref_list[0]}
            rule.update(self._fk_options.get((fact_table_name, col), {}))
            rules.append(rule)
            referenced.extend(ref.split(".")[0] for ref in ref_list)
        return rules, sorted(set(referenced))

    def _compute_fk_options(self) -> dict:
        """Per-(fact table, FK column) extra KeyEnforcer rule options read from the IR
        ``foreign_keys`` hints (``null_rate``, ``fanout``). Empty when none are set."""
        opts: dict = {}
        for table in self._ir["tables"]:
            if table.get("type") != "fact":
                continue
            for fk in table.get("foreign_keys", []):
                option = {}
                if fk.get("null_rate") is not None:
                    option["null_rate"] = fk["null_rate"]
                if fk.get("fanout"):
                    option["fanout"] = fk["fanout"]
                if option:
                    opts[(table["name"], fk["column"])] = option
        return opts

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
        # create the parent directory (Path.parent is "." for a bare filename, so a
        # name with no directory writes to cwd instead of mkdir-ing a truncated junk
        # dir — the old `file_path[:rfind('/')]` produced e.g. "my_config.jso").
        parent = Path(file_path).parent
        parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(json.dumps(self.config_template, sort_keys=False, indent=4))
        print(f"config file for data generation available here:{file_path}")
