"""Frame dtype-category helpers, the ``_Ungenerated`` sentinel, and small
Series/len/timestamp utilities shared by the generation engine. Split out of
``data_generator`` (package split Part 6).
"""

import datetime

import numpy
import pandas as pd
import polars as pl


def _as_pl_series(name: str, values) -> pl.Series:
    """Wrap a generator's output (numpy array, list, or pandas Series/Index) as a pl.Series.

    Generators may return numpy arrays, lists (array columns), or pandas
    datetime structures (e.g. RandomDates). pandas containers are normalised to
    numpy so Polars infers the dtype the same way the old pandas path did.
    """
    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy()
    # numpy object arrays (e.g. datetime.datetime from np.full) don't carry a
    # native dtype Polars can infer; hand it a Python list so Polars infers the
    # element type (datetime -> Datetime) the way the old pandas path coerced it.
    if isinstance(values, numpy.ndarray) and values.dtype == object:
        values = values.tolist()
    return pl.Series(name, values)


class _Ungenerated:
    """Placeholder for a table not yet generated; any real use raises a clear error.

    Replaces the silent empty DataFrame that used to be returned when a
    postprocessor referenced a table appearing later in the config.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str):
        self._name = name

    def _raise(self):
        raise RuntimeError(
            f"Table '{self._name}' was accessed before it was generated "
            f"(it likely appears later in the config than the table referencing it)"
        )

    def __len__(self):
        self._raise()

    def __getitem__(self, key):
        self._raise()

    def __iter__(self):
        self._raise()

    def __getattr__(self, name):
        # _name resolves via __slots__; any other attribute access is premature use
        self._raise()


def _materialized(value):
    """A table's own result for postprocessor input.

    A postprocessor-only table has no generated columns, so its own result is an
    empty frame until the postprocessor builds it. Cross-table references keep the
    _Ungenerated sentinel so premature access to *other* tables still raises.
    """
    return pl.DataFrame() if isinstance(value, _Ungenerated) else value


def _expected_category(col_type: str) -> str:
    """Map a config col_type to a coarse dtype category for schema validation."""
    t = col_type.lower()
    if "array" in t:
        return "array"
    if "int" in t:
        return "int"
    if "float" in t:
        return "float"
    if "datetime" in t or "date" in t:
        return "temporal"
    if "str" in t:
        return "string"
    return "other"


def _frame_categories(df) -> dict:
    """Return {column: coarse dtype category} for a Polars or pandas DataFrame."""
    cats = {}
    if isinstance(df, pl.DataFrame):
        for name, dt in df.schema.items():
            if isinstance(dt, pl.List):
                cats[name] = "array"
            elif dt.is_integer():
                cats[name] = "int"
            elif dt.is_float():
                cats[name] = "float"
            elif dt.is_temporal():
                cats[name] = "temporal"
            elif dt == pl.String:
                cats[name] = "string"
            else:
                cats[name] = "other"
    elif isinstance(df, pd.DataFrame):
        for name, dt in df.dtypes.items():
            if pd.api.types.is_integer_dtype(dt):
                cats[name] = "int"
            elif pd.api.types.is_float_dtype(dt):
                cats[name] = "float"
            elif pd.api.types.is_datetime64_any_dtype(dt):
                cats[name] = "temporal"
            elif pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt):
                cats[name] = "object"  # pandas object is ambiguous (string or list)
            else:
                cats[name] = "other"
    return cats


def _categories_compatible(expected: str, actual: str) -> bool:
    """Allowed coercions: same category, or pandas 'object' for string/array."""
    if expected == actual:
        return True
    return actual == "object" and expected in ("string", "array")


def _safe_len(obj):
    """len(obj) or None — never raises (e.g. on an _Ungenerated sentinel)."""
    try:
        return len(obj)
    except Exception:
        return None


def _ts_diff(a: str, b: str) -> float:
    """Seconds between two ISO timestamps, 0.0 if unparseable."""
    try:
        return (
            datetime.datetime.fromisoformat(b) - datetime.datetime.fromisoformat(a)
        ).total_seconds()
    except Exception:
        return 0.0
