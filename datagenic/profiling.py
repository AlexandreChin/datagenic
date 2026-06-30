"""Scaffold a generation config from a sample of data files — no IR authored by hand.

``build_config(sources)`` reads sample frames, infers an IR (tables, columns, key
candidates, foreign-key candidates) **and** per-column generator arguments from
observed statistics (cardinality, mean/std, value frequencies, null rate), scaffolds
through ``Configs``, and returns a ready generation config. The IR is synthesised
internally (optionally emitted for review) — the user supplies only files. This is a
**scaffold-time** tool: it produces a frozen config, so generation from it stays
byte-identical regardless.

What is fitted per column:
  - dtype                              → IR type (int/float/string/date/datetime)
  - n_unique == row_count (no nulls)   → a unicity-key candidate
  - value-set ⊆ another table's key    → a foreign-key candidate (nullable if it has nulls)
  - low-cardinality categorical        → list_of_values + observed probabilities
  - high-cardinality discrete          → cardinality
  - float                              → mu / sigma (normal fit)
  - null fraction                      → null_spec.rate (non-key) / FK null_rate
"""

import os

import polars as pl

from datagenic.scaffolding import Configs

_MAX_CATEGORICAL = 50
_INT_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_FLOAT_DTYPES = {pl.Float32, pl.Float64}


def _ir_type(dtype) -> str:
    if dtype in _INT_DTYPES:
        return "int"
    if dtype in _FLOAT_DTYPES:
        return "float"
    if dtype == pl.Date:
        return "date"
    if isinstance(dtype, pl.Datetime):
        return "datetime"
    return "string"


def _categorical_args(series) -> dict:
    """list_of_values + a probability vector that sums to exactly 1.0 (numpy-safe)."""
    vc = series.value_counts(sort=True)  # value col first, count col last, desc by count
    values = vc[vc.columns[0]].to_list()
    counts = vc[vc.columns[-1]].to_list()
    total = sum(counts)
    probs = [round(c / total, 6) for c in counts]
    probs[0] = round(probs[0] + (1.0 - sum(probs)), 6)  # absorb rounding residual into the largest
    return {"list_of_values": values, "probabilities": probs}


def _profile_column(series, n, max_categorical):
    """Return (ir_type, is_key, override) for one column. ``override`` is
    ``{"args": {...}, "null_rate": float}`` (either key may be absent)."""
    ir_type = _ir_type(series.dtype)
    null_count = series.null_count()
    # only int/string identifiers are key candidates — a continuous float column can be
    # all-distinct by accident, and floats/dates are not unicity keys in practice.
    is_key = ir_type in ("int", "string") and n > 0 and null_count == 0 and series.n_unique() == n
    if is_key:
        return ir_type, True, {}

    override: dict = {}
    if null_count:
        override["null_rate"] = round(null_count / n, 4)

    non_null = series.drop_nulls()
    distinct = non_null.n_unique()
    args: dict = {}
    if ir_type == "float":
        if len(non_null):
            args["mu"] = round(float(non_null.mean()), 6)
            args["sigma"] = round(float(non_null.std()) if len(non_null) > 1 else 0.0, 6)
    elif ir_type in ("int", "string", "date"):
        if 0 < distinct <= max_categorical:
            args = _categorical_args(non_null)
        elif distinct:
            args["cardinality"] = int(distinct)
    if args:
        override["args"] = args
    return ir_type, False, override


def profile_frames(frames: dict, *, max_categorical: int = _MAX_CATEGORICAL):
    """Profile ``{table_name: polars.DataFrame}`` into ``(ir, overrides_by_table)``.

    ``ir`` is the synthesised internal representation; ``overrides_by_table`` maps each
    table to ``{"rows": int, "cols": {col: {"args": {...}, "null_rate": float}}}`` to
    merge onto the scaffolded config (non-key columns only).
    """
    per: dict = {}  # name -> (df, columns, keys, col_overrides)
    for name, df in frames.items():
        columns, keys, col_ov = [], [], {}
        for col in df.columns:
            ir_type, is_key, override = _profile_column(df[col], df.height, max_categorical)
            columns.append({"name": col, "type": ir_type})
            if is_key:
                keys.append(col)
            elif override:
                col_ov[col] = override
        per[name] = (df, columns, keys, col_ov)

    # single-column key value-sets, for foreign-key candidate detection
    key_sets = {
        (name, keys[0]): set(df[keys[0]].to_list())
        for name, (df, _cols, keys, _ov) in per.items()
        if len(keys) == 1
    }

    ir_tables, overrides_by_table = [], {}
    for name, (df, columns, keys, col_ov) in per.items():
        foreign_keys = []
        for col in [c["name"] for c in columns]:
            if col in keys:
                continue
            values = set(df[col].drop_nulls().to_list())
            if len(values) < 2:
                continue  # too thin to call a relationship
            for (parent, parent_key), parent_values in key_sets.items():
                if parent == name:
                    continue
                if values <= parent_values:
                    fk = {"column": col, "references": f"{parent}.{parent_key}"}
                    moved = col_ov.pop(
                        col, None
                    )  # FK values come from KeyEnforcer, not the generator
                    if moved and moved.get("null_rate"):
                        fk["null_rate"] = moved["null_rate"]  # nullable FK (Feature 4 hint)
                    foreign_keys.append(fk)
                    break

        table = {
            "name": name,
            "type": "fact" if foreign_keys else "referential",
            "keys": keys,
            "columns": columns,
        }
        if foreign_keys:
            table["foreign_keys"] = foreign_keys
        ir_tables.append(table)
        overrides_by_table[name] = {"rows": df.height, "cols": col_ov}

    return {"name": "profiled_model", "tables": ir_tables}, overrides_by_table


def _apply_overrides(config: dict, overrides_by_table: dict) -> None:
    """Merge profiled per-column args / null_spec and per-table row counts onto a
    scaffolded config (in place). Key columns are left untouched."""
    for table in config["tables"]:
        ov = overrides_by_table.get(table["table_name"])
        if not ov:
            continue
        if ov.get("rows") is not None:
            table["number_of_rows"] = ov["rows"]
        col_ov = ov.get("cols", {})
        for col in table["columns"]:
            o = col_ov.get(col["col_name"])
            if not o or col.get("is_unicity_key"):
                continue
            if o.get("args"):
                col["column_generator"]["args"].update(o["args"])
            if o.get("null_rate") is not None:
                col["null_spec"] = {"rate": o["null_rate"]}


def _read_one(path: str, storage_type: str):
    if storage_type == "local":
        return pl.read_parquet(path) if path.endswith(".parquet") else pl.read_csv(path)
    from datagenic.iokit import storages

    return storages.Storages(storage_type).read_parquet_as_polars(path)


def read_sources(sources, storage_type: str = "local") -> dict:
    """Resolve ``sources`` to ``{table_name: polars.DataFrame}``.

    ``sources`` may be a dict (``name -> frame | path``) or a directory path whose
    ``<table>.parquet`` / ``<table>.csv`` files become one table each (stem = name).
    """
    if isinstance(sources, dict):
        return {
            name: (src if hasattr(src, "columns") else _read_one(src, storage_type))
            for name, src in sources.items()
        }
    frames = {}
    for filename in sorted(os.listdir(sources)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in (".parquet", ".csv"):
            frames[stem] = _read_one(os.path.join(sources, filename), storage_type)
    return frames


def build_config(
    sources,
    *,
    storage_type: str = "local",
    semantic_inferrer=None,
    max_categorical: int = _MAX_CATEGORICAL,
    **configs_kwargs,
):
    """Profile ``sources`` into a generation config. Returns ``(config, ir)``.

    The IR is synthesised from the files and never required from the caller. Extra
    ``configs_kwargs`` (output_format, output_folder_path, number_of_batches, seed,
    start_date/end_date, …) pass straight through to ``Configs``.
    """
    frames = read_sources(sources, storage_type)
    ir, overrides_by_table = profile_frames(frames, max_categorical=max_categorical)
    config = Configs(
        input=ir,
        storage_type=storage_type,
        semantic_inferrer=semantic_inferrer,
        samples=frames,  # let a value-based inferrer reuse the same sample data
        **configs_kwargs,
    ).config_template
    _apply_overrides(config, overrides_by_table)
    return config, ir
