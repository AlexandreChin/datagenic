"""The generation engine: ``_Journal`` (append-only run journal + derived summary)
and ``Generator`` (execute a config → data files). Split out of ``data_generator``
(package split Part 6).
"""

import contextlib
import copy
import datetime
import hashlib
import importlib
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy
import pandas as pd
import polars as pl

from datagenic.dtypes import (
    _as_pl_series,
    _categories_compatible,
    _expected_category,
    _frame_categories,
    _materialized,
    _safe_len,
    _ts_diff,
    _Ungenerated,
)
from datagenic.iokit import storages
from datagenic.iokit.utils import print_message


class _Journal:
    """Append-only JSON-lines journal of generation units, flushed immediately.

    One START before and one DONE after each table/date/batch unit (ERROR instead
    of DONE on failure). Because each line is flushed, a hard kill (e.g. OOM
    SIGKILL, which raises no Python exception) leaves the in-progress unit as a
    START with no matching DONE — pinpointing where the process died. Appended
    across runs so crash history is preserved; doubles as an audit trail of writes.

    Also maintains a derived ``generation_summary.json`` (per-table rows/units/
    errors + overall timing), rewritten atomically after each DONE/ERROR so a
    partial summary survives a crash.
    """

    def __init__(self, path: str, summary_path: str | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a")  # noqa: SIM115 (kept open for the run, closed in close())
        self._summary_path = summary_path
        self._records: list[dict] = []
        self._started = datetime.datetime.now()

    def _emit(self, event, table, date, batch, **extra):
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "table": table,
            "date": date,
            "batch": batch,
            **extra,
        }
        self._records.append(rec)
        self._file.write(json.dumps(rec) + "\n")
        self._file.flush()

    def start(self, table, date, batch):
        self._emit("START", table, date, batch)

    def done(self, table, date, batch, rows=None):
        self._emit("DONE", table, date, batch, rows=rows)
        self._write_summary()

    def error(self, table, date, batch, error):
        self._emit("ERROR", table, date, batch, error=error)
        self._write_summary()

    def _write_summary(self):
        if not self._summary_path:
            return
        tables: dict = {}
        errors: list = []
        starts: dict = {}
        for r in self._records:
            entry = tables.setdefault(
                r["table"], {"total_rows": 0, "units": 0, "errors": 0, "elapsed_s": 0.0}
            )
            key = (r["table"], r["date"], r["batch"])
            if r["event"] == "START":
                starts[key] = r["ts"]
            elif r["event"] == "DONE":
                entry["units"] += 1
                entry["total_rows"] += r.get("rows") or 0
                if key in starts:
                    entry["elapsed_s"] = round(
                        entry["elapsed_s"] + _ts_diff(starts[key], r["ts"]), 3
                    )
            elif r["event"] == "ERROR":
                entry["errors"] += 1
                errors.append({k: r.get(k) for k in ("table", "date", "batch", "error")})
        completed = datetime.datetime.now()
        summary = {
            "started_at": self._started.isoformat(timespec="seconds"),
            "completed_at": completed.isoformat(timespec="seconds"),
            "elapsed_s": round((completed - self._started).total_seconds(), 3),
            "tables": tables,
            "errors": errors,
        }
        # atomic write so the summary is always valid JSON, even on crash
        tmp = f"{self._summary_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, self._summary_path)

    def close(self):
        self._write_summary()
        with contextlib.suppress(Exception):
            self._file.close()


class Generator:
    """
    Class for generating data based on a json config file
    """

    def __init__(self, data_generator_config_file):
        """
        data_generator_config_file: dict, json or path to json
        """
        self.data_generator_config_file = data_generator_config_file
        if isinstance(self.data_generator_config_file, dict):
            loaded = self.data_generator_config_file
            base_dir = os.getcwd()  # in-memory config: resolve `extends` relative to cwd
        elif isinstance(self.data_generator_config_file, str):
            if self.data_generator_config_file[-5:] == ".json":
                with open(data_generator_config_file) as f:
                    loaded = json.load(f)
                base_dir = os.path.dirname(os.path.abspath(data_generator_config_file))
            else:
                loaded = json.loads(data_generator_config_file)
                base_dir = os.getcwd()

        # Resolve an optional `extends` chain into a single merged config before
        # anything else, treating the caller config as read-only throughout.
        loaded = self._resolve_extends(loaded, base_dir)

        # Treat the caller-provided config as read-only. Keep a pristine deep
        # copy as the single source of truth and always operate on a fresh
        # working copy, so neither __init__ nor repeated generate() calls
        # mutate the caller's dict or let state drift across runs.
        self._source_config = copy.deepcopy(loaded)
        self._build_working_config()

    def _resolve_extends(self, config, base_dir, _seen=None):
        """Resolve an optional top-level ``extends`` chain into one merged config.

        Child values win: ``default_parameters`` deep-merge over the base; ``tables``
        are merged by ``table_name`` (matching tables deep-merged, base-only tables
        inherited, new child tables appended). A base may itself ``extends`` another
        (single chain); cycles raise a clear error. The caller config is not mutated.
        """
        if not isinstance(config, dict) or "extends" not in config:
            return config
        _seen = list(_seen or [])
        base_path = os.path.normpath(os.path.join(base_dir, config["extends"]))
        if base_path in _seen:
            chain = " -> ".join(_seen + [base_path])
            raise ValueError(f"circular 'extends' chain detected: {chain}")
        with open(base_path) as f:
            base = json.load(f)
        base = self._resolve_extends(base, os.path.dirname(base_path), _seen + [base_path])
        return self._merge_configs(base, config)

    @staticmethod
    def _deep_merge(base: dict, child: dict) -> dict:
        """Recursively merge child over base (child wins on non-dict conflicts)."""
        merged = dict(base)
        for k, v in child.items():
            if isinstance(merged.get(k), dict) and isinstance(v, dict):
                merged[k] = Generator._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    @staticmethod
    def _merge_configs(base: dict, child: dict) -> dict:
        merged = Generator._deep_merge(base, child)
        # tables are merged by name rather than wholesale-replaced
        by_name = {t["table_name"]: t for t in base.get("tables", [])}
        order = [t["table_name"] for t in base.get("tables", [])]
        for ct in child.get("tables", []):
            name = ct["table_name"]
            if name in by_name:
                by_name[name] = Generator._deep_merge(by_name[name], ct)
            else:
                by_name[name] = ct
                order.append(name)
        merged["tables"] = [by_name[n] for n in order]
        merged.pop("extends", None)
        return merged

    def _build_working_config(self):
        """
        (Re)build the mutable working config from the pristine source config.
        Called at construction and at the start of every generate(), so each
        run starts from an identical, unmutated config.
        """
        self._data_generator_config_file_obj = copy.deepcopy(self._source_config)

        self._apply_scale()

        # count unicity keys for each table
        for table in self._data_generator_config_file_obj["tables"]:
            table["number_of_unicity_keys_in_table"] = 0
            for col in table["columns"]:
                if col["is_unicity_key"]:
                    table["number_of_unicity_keys_in_table"] += 1

    def _apply_scale(self):
        """
        Reads `scale` from default_parameters and applies it proportionally:
          - number_of_rows per table is multiplied by scale.
          - cardinality on non-key columns is multiplied by scale
            (preserves the cardinality/rows ratio).
        Key-column cardinality is left untouched: Generator.generate() already
        recomputes it from _number_of_rows_per_batch via pow(rows, 1/n_keys).
        `scale` is popped so the scaling is not applied twice if the same
        config dict is reused across multiple Generator instantiations.
        """
        scale = self._data_generator_config_file_obj["default_parameters"].pop("scale", 1.0)
        # Inject scale into postprocessor_args so every postprocessor receives it.
        self._data_generator_config_file_obj["default_parameters"]["postprocessor_args"][
            "scale"
        ] = scale
        if scale == 1.0:
            return

        for table in self._data_generator_config_file_obj["tables"]:
            # Only scale a static integer count. A dynamic {from_table: ...} count
            # follows its (already-scaled) reference table, so it needs no scaling.
            if isinstance(table.get("number_of_rows"), (int, float)):
                table["number_of_rows"] = max(1, round(table["number_of_rows"] * scale))

            for col in table.get("columns", []):
                col_args = col.get("column_generator", {}).get("args", {})
                if (
                    "cardinality" in col_args
                    and not col.get("is_unicity_key")
                    and "size_of_arrays" not in col_args
                ):
                    col_args["cardinality"] = max(1, round(col_args["cardinality"] * scale))

    def config_file_controls(self):
        """
        performs consistency checks on the config file
        """
        all_checks_ok = True
        # required global default parameters
        required_default_parameters = [
            "output_format",
            "storage_type",
            "output_folder_path",
            "tables_prefix",
            "start_date",
            "end_date",
            "number_of_batches",
            "offset_batches",
            "postprocessor_args",
            "seed",
        ]

        for required_default_parameter in required_default_parameters:
            if (
                required_default_parameter
                not in self._data_generator_config_file_obj["default_parameters"]
            ):
                all_checks_ok = False
                raise Exception(
                    f"Missing default parameter in config file: {required_default_parameter}"
                )

        # supported storage type
        supported_storage_type = ["local", "azure_blob"]
        if (
            self._data_generator_config_file_obj["default_parameters"]["storage_type"]
            not in supported_storage_type
        ):
            all_checks_ok = False
            raise Exception(
                f"Unknown format: {self._data_generator_config_file_obj['default_parameters']['storage_type']}. Please use one of the following format: {supported_storage_type}"
            )

        # supported format
        supported_format = ["csv", "parquet", "avro"]
        if (
            self._data_generator_config_file_obj["default_parameters"]["output_format"]
            not in supported_format
        ):
            all_checks_ok = False
            raise Exception(
                f"Unknown format: {self._data_generator_config_file_obj['default_parameters']['output_format']}. Please use one of the following format: {supported_format}"
            )

        if all_checks_ok:
            print("All checks OK on config file")

    @staticmethod
    def _derive_seed(global_seed: int, table_name: str, date: str, batch: int) -> int:
        key = f"{global_seed}|{table_name}|{date}|{batch}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    def _config_hash(self) -> str:
        """Stable 16-char hash of the *content-determining* config (seed included).

        Excludes IO/location fields (output path, storage, formatting, log paths,
        batch offset) so the same data written to a different directory hashes the
        same — what matters for checkpoint validity is the data, not where it lands.
        """
        cfg = copy.deepcopy(self._source_config)
        dp = cfg.get("default_parameters", {})
        for k in (
            "output_folder_path",
            "storage_type",
            "output_format",
            "tables_prefix",
            "logger_folder_path",
            "generation_log_path",
            "generation_summary_path",
            "offset_batches",
            "batch_range",
            # end_date is excluded so that extending the date range keeps existing
            # per-date files valid (incremental extension); single_date tables are
            # re-validated separately against the stored end_date.
            "end_date",
        ):
            dp.pop(k, None)
        payload = json.dumps(cfg, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _checkpoint_path(self, table, date, batch):
        """Parquet path/key for a unit if checkpointing applies (parquet, dumped), else None.

        Works for local *and* cloud output (Step 12): the path is the same
        ``table/date/batch`` layout the writer uses; the storage layer resolves
        existence + metadata remotely without downloading the whole object.
        """
        p = self._data_generator_config_file_obj["default_parameters"]
        if p["output_format"] != "parquet" or not table.get("dump_results"):
            return None
        name = table["table_name"]
        return f"{p['output_folder_path']}/{p['tables_prefix']}{name}/{date}/{name}-{batch}.parquet"

    def _load_checkpoint(
        self, path, config_hash, force_regenerate, storage, table=None, end_date=None
    ):
        """Return the existing parquet as a Polars frame iff it is a valid, matching
        checkpoint — for local or cloud output, via the storage layer (cloud reads
        only the footer to validate, downloading the frame only on a confirmed hit).
        """
        if force_regenerate or path is None:
            return None
        try:
            if not storage.exists(path):
                return None
            meta = storage.read_parquet_metadata(path)  # dict[str, str]
        except Exception:
            return None
        # only reuse files we wrote whose config hash matches this run; a missing or
        # different hash (foreign / stale file) means regenerate.
        if meta.get("datagenic_config_hash") != config_hash:
            return None
        # single_date tables span the whole date range, so they must be regenerated
        # when end_date changes (incremental extension); date-partitioned tables don't.
        if table is not None and table.get("single_date") and end_date is not None:
            if meta.get("datagenic_end_date") != end_date:
                return None
        return storage.read_parquet_as_polars(path)

    def _make_rng(self, global_seed: int, table_name: str, date: str, batch: int):
        """Return an isolated, reproducible RNG instance for this table/date/batch unit.

        Each unit gets its own ``np.random.default_rng`` seeded deterministically;
        generators and postprocessors draw from it instead of numpy's global state,
        so an extra random draw in one place cannot perturb any other.
        """
        return numpy.random.default_rng(self._derive_seed(global_seed, table_name, date, batch))

    def _with_context(self, fn, udf_args, *, table, date, batch, unit):
        """Run ``fn(**udf_args)``, re-raising any error with table/date/batch context.

        The original exception is preserved as ``__cause__`` (``raise ... from e``),
        so the underlying traceback is intact. Also records an ERROR journal line.
        Only the error path is affected.
        """
        try:
            return fn(**udf_args)
        except Exception as e:
            journal = getattr(self, "_journal", None)
            if journal is not None:
                journal.error(table, date, batch, f"{type(e).__name__}: {e}")
            raise RuntimeError(
                f"{unit} failed for table '{table}', date {date}, batch {batch}: "
                f"{type(e).__name__}: {e}"
            ) from e

    def _build_postprocessor_args(
        self,
        table,
        _result,
        *,
        date,
        previous_date,
        start_date,
        end_date,
        batch,
        number_of_batches,
        post_seed,
    ):
        """Assemble the kwargs passed to a postprocessor.

        `_input_tables` holds exactly the dataframes declared in this postprocessor's
        own `udf_args._input_tables` — no global injection, no more, no less. The
        declared name list is consumed here to build the resolved `_input_tables`
        dict and is NOT forwarded as a list. Framework args use the reserved `_`
        prefix; default `postprocessor_args` (e.g. scale) are passed through.
        """
        pp = table["post_processor"]
        declared = pp["udf_args"].get("_input_tables", [])
        _input_tables = {k: _result[k] for k in declared}

        pp_defaults = {
            k: v
            for k, v in self._data_generator_config_file_obj["default_parameters"][
                "postprocessor_args"
            ].items()
            if k != "_input_tables"
        }
        user_args = {k: v for k, v in pp["udf_args"].items() if k != "_input_tables"}

        return (
            pp_defaults
            | user_args
            | {
                "_input_tables": _input_tables,
                "_table": _materialized(_result[table["table_name"]]),
                "_unicity_keys": table.get("_unicity_keys", []),
            }
            | {
                "_date": date,
                "_previous_date": previous_date,
                "_start_date": start_date,
                "_end_date": end_date,
                "_batch_number": batch,
                "_number_of_batches": number_of_batches,
                "_number_of_rows_per_batch": table["_number_of_rows_per_batch"],
                "_seed": post_seed,
                "_rng": numpy.random.default_rng(post_seed),
            }
        )

    @staticmethod
    def _validate_output_schema(table, df, logger):
        """Validate a to-be-written DataFrame against the table's declared schema.

        Schema source: an explicit ``output_schema`` if present, else the table's
        ``columns``. Missing columns raise ValueError; incompatible dtypes raise
        TypeError; extra columns warn (allowing debug columns). Tables with no
        declared schema (e.g. postprocessor-only with no output_schema) are skipped.
        """
        schema = table.get("output_schema") or table.get("columns") or []
        expected = {c["col_name"]: c["col_type"] for c in schema}
        if not expected:
            return
        actual = _frame_categories(df)
        name = table["table_name"]

        missing = [c for c in expected if c not in actual]
        if missing:
            raise ValueError(f"table '{name}': output is missing expected column(s) {missing}")
        for col, col_type in expected.items():
            exp, act = _expected_category(col_type), actual[col]
            if not _categories_compatible(exp, act):
                raise TypeError(
                    f"table '{name}' column '{col}': expected {exp} dtype "
                    f"(from '{col_type}'), got {act}"
                )
        extra = [c for c in actual if c not in expected]
        if extra:
            print_message(
                f"table '{name}': output has unexpected column(s) {extra} (not in schema)",
                logger,
                type="warning",
            )

    @staticmethod
    def _load_udf_class(udf_module: str, udf_name: str, where: str):
        """Import udf_module and return udf_name, raising a clear error if missing."""
        if not udf_module or not udf_name:
            raise ValueError(f"{where}: udf_module and udf_name are both required")
        try:
            module = importlib.import_module(udf_module)
        except ImportError as e:
            raise ImportError(f"{where}: cannot import udf_module '{udf_module}': {e}") from e
        if not hasattr(module, udf_name):
            raise ImportError(f"{where}: module '{udf_module}' has no attribute '{udf_name}'")
        return getattr(module, udf_name)

    @staticmethod
    def _check_udf_contract(cls, where: str):
        """Enforce the generator/postprocessor contract at validation time.

        Requires a callable ``call``; if ``recall`` is defined it must be callable.
        Catches misspelled/missing call() and a non-callable recall up front rather
        than mid-run on date 2.
        """
        cls_name = getattr(cls, "__name__", cls)
        if not callable(getattr(cls, "call", None)):
            raise TypeError(f"{where}: '{cls_name}' must define a callable call() method")
        recall = getattr(cls, "recall", None)
        if recall is not None and not callable(recall):
            raise TypeError(
                f"{where}: '{cls_name}' defines a non-callable recall; recall must be a method"
            )

    @staticmethod
    def _is_reference_table(table) -> bool:
        """A reference/dimension table is regenerated whole on every shard.

        ``single_batch`` tables (the FK pools) are produced once and broadcast, so a
        postprocessor declaring whole-table scope on one is harmless. Everything else
        is bulk/sharded.
        """
        return bool(table.get("single_batch"))

    def _validate_distributable(self):
        """Distribution-safety gate (static): reject a config a sharded table's
        postprocessor has *declared* unshardable, i.e. ``INPUT_SCOPE="global"``.

        We deliberately do NOT statically reject reading another bulk table: the
        framework injects only the *current batch's* slice into ``_input_tables``
        (``_result[name]``), so such a read is batch-local and shardable. Genuine
        cross-batch breakage (disk reads of other batches, global counters,
        non-injected RNG) is not statically detectable and is caught at runtime by
        ``orchestration.check_distributable`` (determinism probe + the sample
        full-vs-sharded equivalence proof, which is the actual definition of
        shardable).
        """
        for table in self._data_generator_config_file_obj["tables"]:
            if self._is_reference_table(table):
                continue  # reference tables run identically on every worker
            pp = table.get("post_processor") or {}
            if not pp:
                continue
            name = table["table_name"]
            cls = self._load_udf_class(
                pp.get("udf_module"), pp.get("udf_name"), f"table '{name}' post_processor"
            )
            if getattr(cls, "INPUT_SCOPE", "batch") == "global":
                raise ValueError(
                    f"table '{name}' is sharded but its postprocessor declares "
                    f"INPUT_SCOPE='global' — a sharded table cannot depend on whole-table "
                    f"or cross-batch state. Make it batch-independent, or mark the table "
                    f"single_batch (a broadcast reference table)."
                )

    def validate(self, distributed: bool = False):
        """Pre-flight structural validation, run before any data is produced.

        Checks: every udf_module/udf_name is importable; referenced tables exist
        and (for dynamic number_of_rows / global tables) appear earlier; required
        udf_args keys are present (opt-in via a postprocessor's REQUIRED_ARGS);
        and no udf_args key uses the framework-reserved '_' prefix.

        distributed=True additionally runs the distribution-safety gate, which
        refuses (with a clear error) any config that cannot be sharded safely.
        """
        config = self._data_generator_config_file_obj
        tables = config["tables"]
        table_names = {t["table_name"] for t in tables}

        # global postprocessor input tables must reference known tables
        global_tables = (
            config["default_parameters"].get("postprocessor_args", {}).get("_input_tables", [])
        )
        for ref in global_tables:
            if ref not in table_names:
                raise ValueError(
                    f"default_parameters.postprocessor_args._input_tables references unknown "
                    f"table '{ref}'"
                )

        for idx, table in enumerate(tables):
            name = table["table_name"]
            earlier = {t["table_name"] for t in tables[:idx]}

            # column generators are importable and honour the generator contract
            for col in table.get("columns", []):
                cg = col.get("column_generator", {})
                col_where = f"table '{name}' column '{col.get('col_name')}'"
                gen_cls = self._load_udf_class(cg.get("udf_module"), cg.get("udf_name"), col_where)
                self._check_udf_contract(gen_cls, col_where)

            # postprocessor checks
            pp = table.get("post_processor") or {}
            if pp:
                where = f"table '{name}' post_processor"
                cls = self._load_udf_class(pp.get("udf_module"), pp.get("udf_name"), where)
                self._check_udf_contract(cls, where)
                udf_args = pp.get("udf_args", {})

                # `_input_tables` is the one framework-consumed key allowed in udf_args
                # (it declares which tables to inject); any other '_' key is reserved.
                for key in udf_args:
                    if key.startswith("_") and key != "_input_tables":
                        raise ValueError(
                            f"{where}: udf_args key '{key}' starts with '_', which is reserved "
                            f"for framework-injected arguments"
                        )

                for ref in udf_args.get("_input_tables", []):
                    if ref not in table_names:
                        raise ValueError(
                            f"{where}: udf_args._input_tables references unknown table '{ref}'"
                        )

                missing = [k for k in getattr(cls, "REQUIRED_ARGS", []) if k not in udf_args]
                if missing:
                    raise ValueError(f"{where}: missing required udf_args {missing}")

            # dynamic number_of_rows must reference an earlier table
            nr = table.get("number_of_rows")
            if isinstance(nr, dict):
                ft = nr.get("from_table")
                if ft not in table_names:
                    raise ValueError(
                        f"table '{name}': number_of_rows.from_table '{ft}' is not a known table"
                    )
                if ft not in earlier:
                    raise ValueError(
                        f"table '{name}': number_of_rows.from_table '{ft}' must appear earlier "
                        f"in the config than '{name}'"
                    )

        if distributed:
            self._validate_distributable()

    def _print_generation_plan(self):
        """Print the planned generation units without producing any data (dry run)."""
        dp = self._data_generator_config_file_obj["default_parameters"]
        tables = self._data_generator_config_file_obj["tables"]
        n_batches = dp["number_of_batches"]
        n_dates = len(
            pd.bdate_range(dp["start_date"], dp["end_date"], freq="B").strftime("%Y-%m-%d").tolist()
        )
        print(
            f"[dry-run] {len(tables)} tables × {n_dates} dates × {n_batches} batches "
            f"= {len(tables) * n_dates * n_batches} generation units"
        )
        est_per_batch: dict = {}  # table_name -> estimated rows/batch (resolves dynamic chains)
        for i, table in enumerate(tables, 1):
            nr = table.get("number_of_rows")
            name = table["table_name"]
            if isinstance(nr, dict):
                # dynamic: estimate from the referenced table's configured size
                # (defined earlier; chains resolve in order). Ignores dedup, hence ~.
                mult = nr.get("multiplier", 1.0)
                per_batch = round(est_per_batch.get(nr["from_table"], 0) * mult)
                est_per_batch[name] = per_batch
                rows = f"~{per_batch} rows/batch (from {nr['from_table']} ×{mult})"
            elif nr is not None:
                per_batch = nr if table.get("single_batch") else math.ceil(nr / n_batches)
                est_per_batch[name] = per_batch
                rows = f"{per_batch} rows/batch"
            else:
                est_per_batch[name] = 0
                rows = "0 rows/batch"
            flags = [f for f in ("single_date", "single_batch") if table.get(f)]
            deps = table.get("post_processor", {}).get("udf_args", {}).get("_input_tables", [])
            line = f"  {i:>2}. {table['table_name']:<42} — {rows}"
            if flags:
                line += f"  ({', '.join(flags)})"
            if deps:
                line += f"  (depends: {', '.join(deps)})"
            print(line)

    @staticmethod
    def _effective_batch(table, first_batch, batch_number):
        """Batch index to seed with. single_batch tables regenerated by an offset
        worker reuse batch 1's seed so their output matches the canonical run."""
        return 1 if (table["single_batch"] and first_batch > 1) else batch_number

    @staticmethod
    def _resolve_row_count(table, _result, number_of_batches):
        """Set ``table['_number_of_rows_per_batch']`` from its ``number_of_rows``
        spec: a dynamic ``{from_table, multiplier}`` sized off another table's
        current output, a static int split across batches, or absent (0)."""
        nr = table.get("number_of_rows")
        if isinstance(nr, dict):
            # dynamic: size this table from another table's current output
            ref_len = len(_result[nr["from_table"]])
            table["_number_of_rows_per_batch"] = round(ref_len * nr.get("multiplier", 1.0))
        elif nr is not None:
            if table["single_batch"]:
                table["_number_of_rows_per_batch"] = nr
            else:
                table["_number_of_rows_per_batch"] = math.ceil(nr / number_of_batches)
        else:
            table["_number_of_rows_per_batch"] = 0

    @staticmethod
    def _load_postprocessor(table):
        """Import and instantiate a table's configured postprocessor."""
        module = importlib.import_module(table["post_processor"]["udf_module"])
        return getattr(module, table["post_processor"]["udf_name"])()

    def _generate_columns(
        self,
        table,
        *,
        date,
        previous_date,
        start_date,
        end_date,
        effective_batch,
        number_of_batches,
        global_seed,
    ):
        """First-day column generation: build the table's Polars DataFrame from its
        column generators after auto-distributing unicity-key cardinality so the
        keys' product approximates the requested row count. Sets
        ``table['_unicity_keys']`` and returns the deduped, key-sorted frame."""
        for column in table["columns"]:
            column["column_generator"]["args"]["start_date"] = date
            column["column_generator"]["args"]["end_date"] = date

        # Auto-distribute cardinality across the unicity keys so their combined
        # distinct combinations approximate the per-batch row count. Each key with a
        # *fixed* cardinality divides the remaining budget and frees one slot; the
        # leftover budget is then spread evenly (geometric mean) over the free keys.
        remaining_cardinality_budget = table["_number_of_rows_per_batch"]
        free_unicity_key_count = table["number_of_unicity_keys_in_table"]

        unicity_keys = []
        # account for unicity keys whose cardinality is already fixed
        for column in table["columns"]:
            if column["is_unicity_key"]:
                unicity_keys.append(column["col_name"])
                if "datetime" in column["col_type"]:
                    timestamp_cardinality = column["generator_obj"].cardinality(
                        **column["column_generator"]["args"]
                    )
                    remaining_cardinality_budget = max(
                        1, round(remaining_cardinality_budget / timestamp_cardinality)
                    )
                    free_unicity_key_count -= 1
                elif column["col_type"] == "date" and column["col_name"] == "Date":
                    # CurrentAsofDates always produces 1 value per batch;
                    # treat as fixed cardinality=1 so it doesn't consume a free dimension
                    column["column_generator"]["args"]["cardinality"] = 1
                    remaining_cardinality_budget = max(1, round(remaining_cardinality_budget / 1))
                    free_unicity_key_count -= 1
                elif column["column_generator"]["args"].get("list_of_values"):
                    fixed_cardinality = len(column["column_generator"]["args"]["list_of_values"])
                    column["column_generator"]["args"]["cardinality"] = fixed_cardinality
                    remaining_cardinality_budget = max(
                        1, round(remaining_cardinality_budget / fixed_cardinality)
                    )
                    free_unicity_key_count -= 1
                elif column["column_generator"]["args"].get("cardinality") is not None:
                    # an explicitly fixed cardinality consumes a dimension
                    remaining_cardinality_budget = max(
                        1,
                        round(
                            remaining_cardinality_budget
                            / column["column_generator"]["args"]["cardinality"]
                        ),
                    )
                    free_unicity_key_count -= 1

        for column in table["columns"]:
            # a key column whose cardinality is still unset (absent or None)
            # gets it auto-distributed so the keys' product ~= number_of_rows
            if (
                column["is_unicity_key"]
                and "datetime" not in column["col_type"]
                and column["column_generator"]["args"].get("cardinality") is None
            ):
                column["column_generator"]["args"]["cardinality"] = max(
                    1, round(pow(remaining_cardinality_budget, 1 / free_unicity_key_count))
                )

        # one isolated RNG per (table, date, batch), shared across this
        # table's columns so draws are sequential and reproducible
        _unit_rng = self._make_rng(global_seed, table["table_name"], date, effective_batch)
        # generating data for columns -> build a Polars DataFrame from Series
        columns_data = []
        for column in table["columns"]:
            udf_args = {
                "_date": date,
                "_previous_date": previous_date,
                "_start_date": start_date,
                "_end_date": end_date,
                "_batch_number": effective_batch,
                "_number_of_batches": number_of_batches,
                "_number_of_rows_per_batch": table["_number_of_rows_per_batch"],
                "_rng": _unit_rng,
                "_is_unicity_key": column["is_unicity_key"],
                # partial table: columns generated before this one
                "_table": pl.DataFrame(columns_data),
            } | column["column_generator"]["args"]
            values = self._with_context(
                column["generator_obj"].call,
                udf_args,
                table=table["table_name"],
                date=date,
                batch=effective_batch,
                unit=f"column generator '{column['col_name']}'",
            )
            columns_data.append(_as_pl_series(column["col_name"], values))
        df = pl.DataFrame(columns_data)

        # store for use in postprocessors and recall
        table["_unicity_keys"] = unicity_keys

        if unicity_keys:
            dp = self._data_generator_config_file_obj["default_parameters"]
            if dp.get("scale_unique_by_construction", False) and len(unicity_keys) > 1:
                # Composite keys: make tuples unique BY CONSTRUCTION (mixed-radix
                # enumeration) instead of random-draw-then-dedup. Removes the
                # per-unit unique() and the global sort, and makes key values
                # positional so a row-chunk can be produced independently. Single
                # keys are already incremental (unique by construction) -> the
                # branch below handles them as a no-op dedup.
                df = self._enumerate_composite_keys(df, table, unicity_keys)
            else:
                # Drop duplicate key tuples (keep first). This is a per-unit (hash)
                # pass, not a cross-unit one — disjoint per-batch key ranges already
                # keep batches/shards unique by construction (_cardinality_range
                # offset).
                df = df.unique(subset=unicity_keys, keep="first", maintain_order=True)
                # The global key sort is an O(n log n) whole-frame op that does not
                # scale and blocks sub-unit (row-group) chunking — you cannot sort
                # across chunks. Skip it when scale mode is on; row *order* changes
                # but the set of rows and key uniqueness are identical.
                if not dp.get("scale_skip_global_key_sort", False):
                    df = df.sort(unicity_keys)
        return df

    def _enumerate_composite_keys(self, df, table, unicity_keys):
        """Make composite key tuples unique by construction via mixed-radix indexing.

        Each key column's value domain is its distinct *generated* values (already
        offset-disjoint per batch via ``_cardinality_range``), capped to its
        configured cardinality. Row ``r`` is assigned the mixed-radix tuple
        ``(domain_0[r % r0], domain_1[(r // r0) % r1], …)``, so the first
        ``min(height, ∏ radices)`` rows carry distinct key tuples — no per-unit
        ``unique()`` needed. The assignment is positional (row ``r`` → fixed tuple),
        so a row-chunk ``[a:b)`` reproduces the same keys independently. Non-key
        columns keep their generated values (sliced to the same row count).
        """
        cardinality_by_name = {
            col["col_name"]: col["column_generator"]["args"].get("cardinality")
            for col in table["columns"]
            if col["col_name"] in unicity_keys
        }
        domains = []
        for name in unicity_keys:
            observed = df[name].unique(maintain_order=True).to_list()
            cardinality = cardinality_by_name.get(name)
            domains.append(observed[: int(cardinality)] if cardinality else observed)

        radices = [len(domain) for domain in domains]
        n_rows = min(df.height, math.prod(radices)) if all(radices) else df.height

        row_index = numpy.arange(n_rows)
        new_key_columns = []
        divisor = 1
        for name, domain in zip(unicity_keys, domains, strict=True):
            radix = len(domain)
            positions = (row_index // divisor) % radix
            values = [domain[p] for p in positions]
            new_key_columns.append(pl.Series(name, values).cast(df[name].dtype))
            divisor *= radix

        return df.head(n_rows).with_columns(new_key_columns)

    def _recall_columns(
        self,
        table,
        current_df,
        *,
        date,
        previous_date,
        start_date,
        end_date,
        effective_batch,
        number_of_batches,
        global_seed,
    ):
        """Other-dates column regeneration via each generator's ``recall()``.
        Returns the updated frame, or ``current_df`` unchanged if no column
        implements ``recall``."""
        # one isolated RNG per (table, date, batch) before recall
        _unit_rng = self._make_rng(global_seed, table["table_name"], date, effective_batch)
        recalled = []
        for column in table["columns"]:
            # regenerate data
            if hasattr(column["generator_obj"], "recall"):
                udf_args = {
                    "_date": date,
                    "_previous_date": previous_date,
                    "_start_date": start_date,
                    "_end_date": end_date,
                    "_batch_number": effective_batch,
                    "_number_of_batches": number_of_batches,
                    "_number_of_rows_per_batch": table["_number_of_rows_per_batch"],
                    "input_array": current_df[column["col_name"]].to_numpy(),
                    "_rng": _unit_rng,
                    "_is_unicity_key": column["is_unicity_key"],
                    # partial table: columns recalled before this one
                    "_table": pl.DataFrame(recalled),
                } | column["column_generator"]["args"]
                values = self._with_context(
                    column["generator_obj"].recall,
                    udf_args,
                    table=table["table_name"],
                    date=date,
                    batch=effective_batch,
                    unit=f"column generator '{column['col_name']}' (recall)",
                )
                recalled.append(_as_pl_series(column["col_name"], values))
        if recalled:
            return current_df.with_columns(recalled)
        return current_df

    def _dump_table(
        self,
        table,
        df,
        storage,
        parquet_metadata,
        logger,
        *,
        batch_number,
        number_of_batches,
        date,
        date_position,
    ):
        """Validate the output schema and write one ``table/date/batch`` parquet."""
        params = self._data_generator_config_file_obj["default_parameters"]
        tables_prefix = params["tables_prefix"]
        folder_path = f"{params['output_folder_path']}/{tables_prefix}{table['table_name']}/{date}"
        start_dump = time.time()

        print_message(
            f"Table {table['table_name']} (batch {batch_number} ({batch_number}/{number_of_batches}), {date} ({date_position})): dumping data...",
            logger,
            type="info",
        )

        self._validate_output_schema(table, df, logger)
        storage.dump_dataframe(
            df=df,
            file_path=f"{folder_path}/{table['table_name']}-{batch_number}.{params['output_format']}",
            metadata=parquet_metadata,
            # Optional parquet tuning knobs — absent from config => None => current
            # writer defaults (byte-identical output).
            compression=params.get("parquet_compression"),
            use_dictionary=params.get("parquet_use_dictionary"),
            row_group_size=params.get("parquet_row_group_size"),
        )

        print_message(
            f"Table {table['table_name']} (batch {batch_number} ({batch_number}/{number_of_batches}), {date} ({date_position})): data dumped in ",
            logger,
            type="timer",
            where=f"{round((time.time() - start_dump), 2)}s",
        )

    def generate(self, force_regenerate: bool = False, extend: bool = False, dry_run: bool = False):
        """
        Generates fake data based on a config file.

        force_regenerate: ignore existing parquet checkpoints and regenerate all.
        extend: require the output directory to already contain data (guards
            against accidentally treating a fresh run as an incremental extension).
        dry_run: validate the config and print the generation plan without
            producing any data or invoking any generator/postprocessor.
        """
        # Start every run from a fresh, unmutated working config so repeated
        # generate() calls are independent and reproducible.
        self._build_working_config()

        start = time.time()
        tables_prefix = self._data_generator_config_file_obj["default_parameters"]["tables_prefix"]
        _global_seed = self._data_generator_config_file_obj["default_parameters"].get("seed", 0)
        # config hash embedded in every parquet; checkpoints reuse a file only when it matches
        _config_hash = self._config_hash()
        _parquet_metadata = {
            "datagenic_config_hash": _config_hash,
            "datagenic_seed": str(_global_seed),
            "datagenic_end_date": str(
                self._data_generator_config_file_obj["default_parameters"]["end_date"]
            ),
        }

        self.config_file_controls()
        self.validate()

        if dry_run:
            self._print_generation_plan()
            return

        if extend:
            _out_dir = self._data_generator_config_file_obj["default_parameters"][
                "output_folder_path"
            ]
            if not os.path.isdir(_out_dir) or not any(Path(_out_dir).rglob("*.parquet")):
                raise FileNotFoundError(
                    f"extend=True but no existing parquet data found under '{_out_dir}'"
                )

        # load generators obj
        for table_idx in range(len(self._data_generator_config_file_obj["tables"])):
            for col_idx in range(
                len(self._data_generator_config_file_obj["tables"][table_idx]["columns"])
            ):
                # udf generator
                module = importlib.import_module(
                    self._data_generator_config_file_obj["tables"][table_idx]["columns"][col_idx][
                        "column_generator"
                    ]["udf_module"]
                )
                udf = self._data_generator_config_file_obj["tables"][table_idx]["columns"][col_idx][
                    "column_generator"
                ]["udf_name"]
                self._data_generator_config_file_obj["tables"][table_idx]["columns"][col_idx][
                    "generator_obj"
                ] = getattr(module, udf)()

        # dates
        _start_date = self._data_generator_config_file_obj["default_parameters"]["start_date"]
        _end_date = self._data_generator_config_file_obj["default_parameters"]["end_date"]
        list_of_dates = (
            pd.bdate_range(_start_date, _end_date, freq="B").strftime("%Y-%m-%d").tolist()
        )

        # remove 1st date
        list_of_dates.pop(0)

        # number_of_batches
        number_of_batches = self._data_generator_config_file_obj["default_parameters"][
            "number_of_batches"
        ]
        offset_batches = self._data_generator_config_file_obj["default_parameters"][
            "offset_batches"
        ]
        batch_range = self._data_generator_config_file_obj["default_parameters"].get("batch_range")
        if batch_range is not None:
            first_batch = batch_range[0] + 1
            last_batch = batch_range[1]
        else:
            first_batch = offset_batches + 1
            last_batch = offset_batches + number_of_batches

        # init result and storage
        storage = storages.Storages(
            self._data_generator_config_file_obj["default_parameters"]["storage_type"]
        )
        # Each table starts as an _Ungenerated sentinel; referencing it before it
        # is produced raises instead of silently returning an empty DataFrame.
        _result = {}
        for table in self._data_generator_config_file_obj["tables"]:
            _result[table["table_name"]] = _Ungenerated(table["table_name"])

        # structured progress journal (JSON lines, START/DONE/ERROR per unit)
        _params = self._data_generator_config_file_obj["default_parameters"]
        _log_path = _params.get(
            "generation_log_path",
            f"{_params['output_folder_path']}/{tables_prefix}generation.log",
        )
        _summary_path = _params.get(
            "generation_summary_path",
            f"{_params['output_folder_path']}/{tables_prefix}generation_summary.json",
        )
        self._journal = _Journal(_log_path, _summary_path)

        # construct a logger
        if "logger_folder_path" in self._data_generator_config_file_obj["default_parameters"]:
            logger_folder_path = f"{self._data_generator_config_file_obj['default_parameters']['logger_folder_path']}/{self._data_generator_config_file_obj['default_parameters']['tables_prefix']}generator_logs.log"

            logger = logging.getLogger("my_logger")
            logger.setLevel(logging.DEBUG)

            file_handler = logging.FileHandler(logger_folder_path, mode="w")
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            print_message(
                f"Log file available here: {logger_folder_path}",
                logger,
                type="info",
            )
        else:
            logger = False

        print(f"Generating data for {len(list_of_dates) + 1} dates...")

        for _batch_number in range(first_batch, last_batch + 1):
            print_message(
                f"---------Batch {_batch_number}/{number_of_batches} generation begins---------",
                logger,
                type="important",
            )

            # generates data for start_date
            print_message(
                f"Batch {_batch_number} ({_batch_number}/{number_of_batches}) First day generation begins",
                logger,
                type="important",
            )
            for table in self._data_generator_config_file_obj["tables"]:
                self._journal.start(table["table_name"], _start_date, _batch_number)
                # resolve the per-batch row count BEFORE any checkpoint skip, so a
                # restored table still has _number_of_rows_per_batch for later dates.
                self._resolve_row_count(table, _result, number_of_batches)

                _eff_batch = self._effective_batch(table, first_batch, _batch_number)
                _ckpt_df = self._load_checkpoint(
                    self._checkpoint_path(table, _start_date, _eff_batch),
                    _config_hash,
                    force_regenerate,
                    storage,
                    table=table,
                    end_date=_end_date,
                )
                if _ckpt_df is not None:
                    _result[table["table_name"]] = _ckpt_df
                    self._journal.done(
                        table["table_name"], _start_date, _batch_number, _safe_len(_ckpt_df)
                    )
                    continue
                _date = _start_date
                _previous_date = _start_date

                if (_batch_number == first_batch or not table["single_batch"]) and table.get(
                    "columns"
                ):
                    _result[table["table_name"]] = self._generate_columns(
                        table,
                        date=_date,
                        previous_date=_previous_date,
                        start_date=_start_date,
                        end_date=_end_date,
                        effective_batch=_eff_batch,
                        number_of_batches=number_of_batches,
                        global_seed=_global_seed,
                    )

                # post processing
                if table["post_processor"]:
                    if _batch_number == first_batch or not table["single_batch"]:
                        _effective_batch = self._effective_batch(table, first_batch, _batch_number)
                        _post_seed = self._derive_seed(
                            _global_seed, table["table_name"] + ":post", _date, _effective_batch
                        )
                        udf_args = self._build_postprocessor_args(
                            table,
                            _result,
                            date=_date,
                            previous_date=_previous_date,
                            start_date=_start_date,
                            end_date=_end_date,
                            batch=_effective_batch,
                            number_of_batches=number_of_batches,
                            post_seed=_post_seed,
                        )
                        post_processor_obj = self._load_postprocessor(table)
                        start_gen = time.time()

                        print_message(
                            f"Table {table['table_name']} (batch {_batch_number} ({_batch_number}/{number_of_batches}), {_date} (1/{len(list_of_dates) + 1})): generating data...",
                            logger,
                            type="info",
                        )
                        _result[table["table_name"]] = self._with_context(
                            post_processor_obj.call,
                            udf_args,
                            table=table["table_name"],
                            date=_date,
                            batch=_effective_batch,
                            unit="postprocessor",
                        )

                        print_message(
                            f"Table {table['table_name']} (batch {_batch_number} ({_batch_number}/{number_of_batches}), {_date} (1/{len(list_of_dates) + 1}): data generated in ",
                            logger,
                            type="timer",
                            where=f"{round((time.time() - start_gen), 2)}s",
                        )
                # Dumping results
                if table["dump_results"]:
                    if (_batch_number == first_batch or not table["single_batch"]) and not (
                        table["single_batch"] and first_batch > 1
                    ):
                        self._dump_table(
                            table,
                            _result[table["table_name"]],
                            storage,
                            _parquet_metadata,
                            logger,
                            batch_number=_batch_number,
                            number_of_batches=number_of_batches,
                            date=_date,
                            date_position=f"1/{len(list_of_dates) + 1}",
                        )

                self._journal.done(
                    table["table_name"],
                    _start_date,
                    _batch_number,
                    _safe_len(_result[table["table_name"]]),
                )

            # generates data for other dates
            for _date_idx, _date in enumerate(list_of_dates):
                _previous_date = _start_date if _date_idx == 0 else list_of_dates[_date_idx - 1]

                print_message(
                    f"Batch {_batch_number} ({_batch_number}/{number_of_batches}) Day {_date} ({_date_idx + 2}/{len(list_of_dates) + 1}) generation begins",
                    logger,
                    type="important",
                )
                for table in self._data_generator_config_file_obj["tables"]:
                    self._journal.start(table["table_name"], _date, _batch_number)
                    _eff_batch = self._effective_batch(table, first_batch, _batch_number)
                    _ckpt_df = self._load_checkpoint(
                        self._checkpoint_path(table, _date, _eff_batch),
                        _config_hash,
                        force_regenerate,
                        storage,
                        table=table,
                        end_date=_end_date,
                    )
                    if _ckpt_df is not None:
                        _result[table["table_name"]] = _ckpt_df
                        self._journal.done(
                            table["table_name"], _date, _batch_number, _safe_len(_ckpt_df)
                        )
                        continue
                    if "columns" in table and table["columns"]:
                        if _batch_number == first_batch or not table["single_batch"]:
                            _result[table["table_name"]] = self._recall_columns(
                                table,
                                _result[table["table_name"]],
                                date=_date,
                                previous_date=_previous_date,
                                start_date=_start_date,
                                end_date=_end_date,
                                effective_batch=self._effective_batch(
                                    table, first_batch, _batch_number
                                ),
                                number_of_batches=number_of_batches,
                                global_seed=_global_seed,
                            )

                    # post processing
                    if table["post_processor"]:
                        if not table["single_date"] and (
                            _batch_number == first_batch or not table["single_batch"]
                        ):
                            _effective_batch = self._effective_batch(
                                table, first_batch, _batch_number
                            )
                            _post_seed = self._derive_seed(
                                _global_seed, table["table_name"] + ":post", _date, _effective_batch
                            )
                            udf_args = self._build_postprocessor_args(
                                table,
                                _result,
                                date=_date,
                                previous_date=_previous_date,
                                start_date=_start_date,
                                end_date=_end_date,
                                batch=_effective_batch,
                                number_of_batches=number_of_batches,
                                post_seed=_post_seed,
                            )
                            post_processor_obj = self._load_postprocessor(table)
                            if hasattr(post_processor_obj, "recall"):
                                _result[table["table_name"]] = self._with_context(
                                    post_processor_obj.recall,
                                    udf_args,
                                    table=table["table_name"],
                                    date=_date,
                                    batch=_effective_batch,
                                    unit="postprocessor (recall)",
                                )
                            else:
                                start_gen = time.time()

                                print_message(
                                    f"Table {table['table_name']} (batch {_batch_number} ({_batch_number}/{number_of_batches}), {_date} ({_date_idx + 2}/{len(list_of_dates) + 1})): generating data...",
                                    logger,
                                    type="info",
                                )
                                _result[table["table_name"]] = self._with_context(
                                    post_processor_obj.call,
                                    udf_args,
                                    table=table["table_name"],
                                    date=_date,
                                    batch=_effective_batch,
                                    unit="postprocessor",
                                )

                                print_message(
                                    f"Table {table['table_name']} (batch {_batch_number} ({_batch_number}/{number_of_batches}), {_date} ({_date_idx + 2}/{len(list_of_dates) + 1}): data generated in ",
                                    logger,
                                    type="timer",
                                    where=f"{round((time.time() - start_gen), 2)}s",
                                )
                    # Dumping results
                    if table["dump_results"]:
                        if (
                            not table["single_date"]
                            and (_batch_number == first_batch or not table["single_batch"])
                            and not (table["single_batch"] and first_batch > 1)
                        ):
                            self._dump_table(
                                table,
                                _result[table["table_name"]],
                                storage,
                                _parquet_metadata,
                                logger,
                                batch_number=_batch_number,
                                number_of_batches=number_of_batches,
                                date=_date,
                                date_position=f"{_date_idx + 2}/{len(list_of_dates) + 1}",
                            )

                    self._journal.done(
                        table["table_name"],
                        _date,
                        _batch_number,
                        _safe_len(_result[table["table_name"]]),
                    )

        self._journal.close()

        print_message(
            f"Data generation completed in {math.ceil(time.time() - start)}s",
            logger,
            type="final",
        )

        print_message(
            f"Data files available here: {self._data_generator_config_file_obj['default_parameters']['output_folder_path']} ({self._data_generator_config_file_obj['default_parameters']['storage_type']})",
            logger,
            type="final",
        )

        if "logger_folder_path" in self._data_generator_config_file_obj["default_parameters"]:
            print_message(
                f"Log file available here: {logger_folder_path}",
                logger,
                type="final",
            )
