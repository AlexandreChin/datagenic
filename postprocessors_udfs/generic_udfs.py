"""Example and test-support UDFs for the data generator (NOT part of any shipped
package — a dev/sample fixture resolved via pythonpath).

Holds example UDFs (ImportFile, MergeDataframes, DerivedColumn, filters) and the
test-support probes (RecordingPostProcessor, Boom*, etc.). The two postprocessors
datagenic's scaffolder wires by default — ``KeyEnforcer`` and
``TemplatePostProcessor`` — now ship in ``datagenic.postprocessors`` and are
re-exported below so existing ``udf_module: "postprocessors_udfs.generic_udfs"`` references resolve.
"""

import datetime  # noqa: F401
import itertools  # noqa: F401
import math  # noqa: F401

import numpy as np
import pandas as pd
import polars as pl

# KeyEnforcer / TemplatePostProcessor moved into the datagenic package (they are
# wired by datagenic's scaffolder); re-exported here so existing configs/tests that
# reference "postprocessors_udfs.generic_udfs.*" keep working.
from datagenic.postprocessors import (  # noqa: E402, F401
    KeyEnforcer,
    TemplatePostProcessor,
    _as_pl,
)


def ReadFromFile(file_path: str):
    if file_path[-8:] == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path[-4:] == ".csv":
        df = pd.read_csv(file_path)
    elif file_path[-5:] == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        raise Exception("unknown file format")

    return df


def FilterFromDfOnIndex(batch_number: int, number_of_rows_per_batch: int, df: pd.DataFrame):
    # extract a certain number of rows for each batch
    if ((batch_number - 1) * number_of_rows_per_batch) < (df.index[-1] + 1):
        start_idx = (batch_number - 1) * number_of_rows_per_batch
        end_idx = min(start_idx + number_of_rows_per_batch, df.index[-1] + 1)
    else:
        end_idx = df.index[-1] + 1
        start_idx = max(0, end_idx - number_of_rows_per_batch)

    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    return df


def FilterFromDfOnDate(df: pd.DataFrame, date: str):
    # filter on date
    df = df.loc[df["Date"] == date]

    return df


def brownian_bridge(
    start_time: str, periods: int, vol: float, freq: str, start_value: float, end_value: float
) -> pd.array:
    """
    periods: number of prices to generate
    freq: frequence (eg. "1s", "10s" etc.)
    """
    # generate time stamps
    timestamps = pd.date_range(start_time, periods=periods + 1, freq=freq)

    # generate prices without shocks
    drift = np.random.normal(loc=(end_value - start_value) / periods, scale=0, size=len(timestamps))
    drift[0] = 0
    prices = start_value + drift.cumsum()

    # generate shocks
    shocks = np.random.normal(loc=0, scale=vol, size=len(timestamps))
    shocks[0] = 0
    cumulative_shocks = shocks.cumsum()

    # reversion factor for brownian bridge
    reversion_factor = np.random.normal(loc=1 / periods, scale=0, size=len(timestamps))
    reversion_factor[0] = 0
    reversion_factor = np.flip(reversion_factor.cumsum())
    corrected_cumulative_shocks = cumulative_shocks * reversion_factor

    prices_with_shocks = prices + corrected_cumulative_shocks

    return prices_with_shocks


class ImportFile:
    def call(self, file_path: str, **kwargs):
        final_df = ReadFromFile(file_path)
        return final_df


class SomeUserDefinedColumnGenerator:
    def call(
        self,
        prefix: str,
        _batch_number: int,
        _number_of_rows_per_batch: int,
        cardinality: int,
        list_of_values: list = None,
        probabilities: list = None,
        **kwargs,
    ):
        rng = kwargs.get("_rng") or np.random.default_rng()
        if list_of_values:
            return rng.choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        else:
            offset = (
                (_batch_number - 1) * cardinality if kwargs.get("_is_unicity_key", False) else 0
            )
            return np.char.add(
                np.full(_number_of_rows_per_batch, prefix),
                rng.integers(
                    low=offset + 1,
                    high=offset + cardinality + 1,
                    size=_number_of_rows_per_batch,
                ).astype(str),
            )


class DerivedColumn:
    """Correlated column generator: derives values from an earlier column.

    Demonstrates partial table injection — the framework passes ``_table``
    containing the columns generated before this one. Returns ``"<source>_d"``
    when ``source`` is available, else a fallback marker.
    """

    def call(self, _number_of_rows_per_batch, source=None, _table=None, **kwargs):
        cols = getattr(_table, "columns", [])
        if _table is not None and source in cols:
            return np.array([f"{v}_d" for v in _table[source].to_list()])
        return np.full(_number_of_rows_per_batch, "NO_SOURCE")


class MergeDataframes:
    def call(
        self,
        _input_tables: dict,
        left_on: list = None,
        right_on: list = None,
        join_type: str = "left",
        **kwargs,
    ) -> pd.DataFrame:
        # Merge the declared tables in the order they appear in udf_args.tables
        # (that order is preserved by _input_tables). The first two are left/right.
        # Coerce Polars frames (e.g. column-generated or checkpoint-reloaded) to pandas.
        if right_on is None:
            right_on = ["Date"]
        if left_on is None:
            left_on = ["Date"]
        dfs = [d.to_pandas() if hasattr(d, "to_pandas") else d for d in _input_tables.values()]
        left_df, right_df = dfs[0], dfs[1]

        merged_df = left_df.merge(right_df, left_on=left_on, right_on=right_on, how=join_type)

        return merged_df


# Append-only log of framework invocations; tests clear it before each run.
RECORDS = []


def _snapshot(method, _table, _input_tables, _date, _batch_number):
    return {
        "method": method,
        "date": _date,
        "batch": _batch_number,
        "table_len": len(_table) if _table is not None else None,
        "table_cols": list(_table.columns) if hasattr(_table, "columns") else None,
        "table_mod": type(_table).__module__,
        "tables_keys": sorted(_input_tables.keys()) if _input_tables else [],
        "tables_lens": {k: len(v) for k, v in (_input_tables or {}).items()},
        "tables_mods": {k: type(v).__module__ for k, v in (_input_tables or {}).items()},
    }


class RecordingPostProcessor:
    """Records every call/recall and echoes ``_table`` back so it can be dumped."""

    def call(self, _table=None, _input_tables=None, _date=None, _batch_number=None, **kwargs):
        rec = _snapshot("call", _table, _input_tables, _date, _batch_number)
        rec["raw_tables_kwarg"] = "tables" in kwargs
        RECORDS.append(rec)
        return _passthrough(_table)

    def recall(self, _table=None, _input_tables=None, _date=None, _batch_number=None, **kwargs):
        rec = _snapshot("recall", _table, _input_tables, _date, _batch_number)
        rec["raw_tables_kwarg"] = "tables" in kwargs
        RECORDS.append(rec)
        return _passthrough(_table)


def _passthrough(_table):
    if _table is not None and hasattr(_table, "columns") and len(_table.columns):
        return _table
    return pd.DataFrame({"_probe": [0]})


class DictReturning:
    """Returns a plain dict — a non-DataFrame intermediate (for dump_results:false)."""

    def call(self, **kwargs):
        return {"marker": "intermediate", "n": 3}


class RequiresArgs:
    """Declares a required udf_arg via REQUIRED_ARGS (for validate() tests)."""

    REQUIRED_ARGS = ["needed"]

    def call(self, **kwargs):
        return pd.DataFrame({"x": [1]})


class BoomGenerator:
    """Column generator that always raises (for contextual-error-wrapping tests)."""

    def call(self, **kwargs):
        raise ValueError("boom-gen")


class BoomPostProcessor:
    """Postprocessor that always raises (for contextual-error-wrapping tests)."""

    def call(self, **kwargs):
        raise ValueError("boom-pp")


class NonCallableRecall:
    """Postprocessor with a non-callable `recall` attribute (protocol-violation test)."""

    recall = 42  # not a method

    def call(self, **kwargs):
        return pd.DataFrame({"x": [1]})


class NonCallableCallGenerator:
    """Column generator whose `call` is not callable (protocol-violation test)."""

    call = 7  # not a method


class ReturnsFrame:
    """Returns a pandas DataFrame with the given columns (output-schema tests)."""

    def call(self, cols, dtype="string", n=3, **kwargs):
        if dtype == "int":
            data = {c: list(range(n)) for c in cols}
        else:
            data = {c: [f"v{i}" for i in range(n)] for c in cols}
        return pd.DataFrame(data)


class DuplicateKeyPostProcessor:
    """Returns a frame with intentionally duplicated (k1, k2) key tuples.

    Used to prove the framework applies NO uniqueness to postprocessor output:
    these duplicates must survive to disk even with scale uniqueness flags on.
    """

    def call(self, **kwargs):
        return pl.DataFrame({"k1": [1, 1, 2], "k2": [9, 9, 8], "val": [0.1, 0.2, 0.3]})


class GlobalScopePostProcessor:
    """Declares a whole-table dependency — must be rejected for sharded tables."""

    INPUT_SCOPE = "global"

    def call(self, _table=None, _input_tables=None, **kwargs):
        return _passthrough(_table)


class ReferenceReadingPostProcessor:
    """Reads reference tables only — safe to shard (passes the gate)."""

    INPUT_SCOPE = "reference"

    def call(self, _table=None, _input_tables=None, **kwargs):
        return _passthrough(_table)


class NonDeterministicPostProcessor:
    """Uses the GLOBAL numpy RNG instead of the injected _rng — breaks
    reproducibility, so the determinism probe must flag it."""

    def call(self, _number_of_rows_per_batch=3, **kwargs):
        return pl.DataFrame({"r": np.random.random(_number_of_rows_per_batch)})
