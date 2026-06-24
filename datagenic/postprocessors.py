"""Built-in postprocessors that datagenic's scaffolder wires by default.

These ship with ``datagenic`` so a default-scaffolded config is self-contained
(references only ``datagenic.*``), with no dependency on the top-level
``postprocessors_udfs`` package (the dev/sample UDFs in ``postprocessors_udfs/generic_udfs.py``,
distinct from this ``datagenic.postprocessors`` module):

- ``TemplatePostProcessor`` — the no-op hook the scaffolder attaches to every table
  (``--no-postprocessor-templates`` / ``template_postprocessors=False`` to skip).
- ``KeyEnforcer`` — foreign-key / key-alignment enforcement wired onto fact tables
  when ``enforce_fk=True`` (the default).
"""

import polars as pl


def _as_pl(df) -> pl.DataFrame:
    """Accept a pandas or Polars (eager/lazy) frame, return an eager Polars frame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return pl.from_pandas(df)


class TemplatePostProcessor:
    """No-op postprocessor template.

    Returns the generated table unchanged. The scaffolder wires this onto every
    table so each one has a postprocessor hook ready to customize: copy this
    class, rename it, point the table's ``post_processor.udf_name`` at the copy,
    and edit ``call`` (add ``recall`` if later dates need different behaviour).

    Return value: the DataFrame to dump (pandas or Polars). Returning ``_table``
    is the identity / no-op.

    The framework injects the arguments below on every ``call``/``recall``. Name
    the ones you use; ``**kwargs`` swallows the rest (so new framework args never
    break your code). All framework args use a leading ``_``; your own config
    ``udf_args`` arrive as plain kwargs alongside them.

    Framework-injected arguments
    ----------------------------
    _table : DataFrame
        This table's own generated rows. Empty frame if the table has no
        generated columns (postprocessor-only table).
    _input_tables : dict[str, DataFrame]
        The upstream tables this postprocessor declared in
        ``udf_args._input_tables``, resolved to their data — exactly those,
        nothing more. ``{}`` when none are declared.
    _unicity_keys : list[str]
        This table's unicity-key column names.
    _date : str
        The asof / partition date currently being generated.
    _previous_date : str | None
        The preceding date in the range; ``None`` on the first date.
    _start_date, _end_date : str
        Bounds of the configured date range.
    _batch_number : int
        Current batch index (1-based).
    _number_of_batches : int
        Total number of batches configured.
    _number_of_rows_per_batch : int
        Rows generated per batch for this table.
    _seed : int
        Deterministic seed derived from sha256(seed | table | date | batch).
    _rng : numpy.random.Generator
        A ``default_rng(_seed)``. Use THIS for any randomness so runs stay
        reproducible and resumable — do not call the global ``numpy.random``.
    """

    def call(self, _table, _input_tables=None, **kwargs):
        # `kwargs` holds the framework args documented above (e.g. kwargs["_rng"],
        # kwargs["_date"]) plus any of your own config `udf_args`.
        return _table


class KeyEnforcer:
    """
    Unified FK / key-alignment postprocessor. Accepts a list of rules:

      {"col": "dest",           "copy": "src_col"}
          Copy src_col from the same table into dest (deterministic, row-wise).

      {"col": "dest",           "literal": value}
          Set dest to a constant literal.

      {"col": "dest",           "from": "Table.col"}
      {"col": "dest",           "from": ["Table1.col1", "Table2.col2"]}
          Sample dest from the pool of Table.col (intersection when multiple sources).
          On recall(), skipped for unicity-key columns (keeps values stable).

      {"cols": ["c1", "c2"],    "from": "Table"}
          Copy entire key tuples from Table (guarantees valid multi-col combinations).
          On recall(), skipped (tuples already set on first date stay stable).

      {"cols": ["c1", "c2"],    "from": "Table",  "cross": "CrossTable.col"}
          Same as above but cross-joined with every unique value of CrossTable.col,
          producing the full cartesian key space.

    After all rules, duplicate rows are dropped on the unicity keys (if any).
    """

    def _apply(
        self,
        _table,
        _input_tables: dict,
        rules: list[dict],
        _unicity_keys: list[str],
        preserve_tuple_and_sample: bool,
        _seed: int = 42,
    ) -> pl.DataFrame:
        """
        Execute all rules in order against the table.

        preserve_tuple_and_sample=True on recall() to skip tuple and sample
        rules for unicity-key columns, keeping synthetic keys stable across dates.
        """
        df = _as_pl(_table)
        curr = df  # kept for fact-value sampling in tuple rules

        for rule in rules:
            # --- copy rule ---
            if "copy" in rule:
                df = df.with_columns(pl.col(rule["copy"]).alias(rule["col"]))

            # --- literal rule ---
            elif "literal" in rule:
                df = df.with_columns(pl.lit(rule["literal"]).alias(rule["col"]))

            # --- tuple rule (multi-col from a source table) ---
            elif "cols" in rule:
                if preserve_tuple_and_sample:
                    continue
                rename = rule.get("rename", {})
                src = (
                    _as_pl(_input_tables[rule["from"]])
                    .select(rule["cols"])
                    .unique(maintain_order=True)
                )
                if rename:
                    src = src.rename(rename)
                dest_cols = [rename.get(c, c) for c in rule["cols"]]
                # Optionally inner-join with another table to restrict tuples to
                # valid cross-table combinations before expanding/sampling.
                if "join" in rule:
                    join_on_src = {v: k for k, v in rename.items()}.get(
                        rule["join"]["on"], rule["join"]["on"]
                    )
                    join_tables = (
                        rule["join"]["table"]
                        if isinstance(rule["join"]["table"], list)
                        else [rule["join"]["table"]]
                    )
                    for jtable in join_tables:
                        join_src = (
                            _as_pl(_input_tables[jtable])
                            .select([rule["join"]["on"], rule["join"]["pick"]])
                            .unique(maintain_order=True)
                        )
                        src = src.join(
                            join_src,
                            left_on=join_on_src,
                            right_on=rule["join"]["on"],
                            how="inner",
                        )
                    src = src.unique(maintain_order=True)
                    dest_cols = list(dict.fromkeys(dest_cols + [rule["join"]["pick"]]))
                if rule.get("sample"):
                    # Sample with replacement to match the current table row count,
                    # overwriting only the target columns (keeps other columns intact).
                    sampled = src.sample(n=len(df), with_replacement=True, seed=_seed)
                    for col in dest_cols:
                        df = df.with_columns(sampled[col].alias(col))
                else:
                    if "cross" in rule:
                        cross_table, cross_col = rule["cross"].rsplit(".", 1)
                        extra = (
                            _as_pl(_input_tables[cross_table])
                            .select([cross_col])
                            .unique(maintain_order=True)
                        )
                        src = src.join(extra, how="cross")
                    n = len(src)
                    src = src.with_columns(pl.lit(curr["Date"][0]).alias("Date"))
                    for col in curr.columns:
                        if col not in src.columns:
                            src = src.with_columns(
                                curr[col].sample(n=n, with_replacement=True, seed=_seed).alias(col)
                            )
                    df = src.select(curr.columns)

            # --- sample rule (single col from pool) ---
            else:
                col = rule["col"]
                if preserve_tuple_and_sample and col in _unicity_keys:
                    continue
                sources = rule["from"] if isinstance(rule["from"], list) else [rule["from"]]
                sets = []
                _table_cache: dict = {}
                for ref in sources:
                    ref_table, ref_col = ref.rsplit(".", 1)
                    if ref_table not in _table_cache:
                        _table_cache[ref_table] = _as_pl(_input_tables[ref_table])
                    sets.append(set(_table_cache[ref_table][ref_col].drop_nulls().to_list()))
                pool = sorted(sets[0] if len(sets) == 1 else set.intersection(*sets))
                if not pool:
                    raise ValueError(f"KeyEnforcer: empty pool for {col} from {sources}")
                df = df.with_columns(
                    pl.Series(
                        col,
                        pl.Series(pool)
                        .sample(n=len(df), with_replacement=True, seed=_seed)
                        .to_list(),
                    )
                )

        if _unicity_keys:
            df = df.unique(subset=_unicity_keys, maintain_order=True)
        return df

    def call(
        self, _table, _input_tables, rules: list, _unicity_keys=None, _seed: int = 42, **kwargs
    ):
        """First date: apply all rules including tuple and sample rules."""
        return self._apply(_table, _input_tables, rules, _unicity_keys or [], False, _seed)

    def recall(
        self, _table, _input_tables, rules: list, _unicity_keys=None, _seed: int = 42, **kwargs
    ):
        """Subsequent dates: skip tuple/sample rules on unicity keys to preserve key stability."""
        return self._apply(_table, _input_tables, rules, _unicity_keys or [], True, _seed)
