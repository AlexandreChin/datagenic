# default generators functions
import datetime

import numpy as np
import pandas as pd


def _rng_from(kwargs) -> np.random.Generator:
    """Return the framework-injected RNG instance, or a fresh one for direct calls.

    The data generator injects a seeded ``_rng`` (np.random.default_rng) per
    table/date/batch; generators draw from it so no global numpy state is used.
    """
    return kwargs.get("_rng") or np.random.default_rng()


def _cardinality_range(cardinality, _batch_number, kwargs, base=0):
    """Half-open [low, high) integer range of size ``cardinality`` to draw values from.

    `cardinality` is the column's distinct-value count. Non-key columns draw from
    the SAME pool every batch (total distinct == cardinality). Unicity-key columns
    get a disjoint slice per batch so keys stay unique when batches are concatenated.
    """
    offset = (_batch_number - 1) * cardinality if kwargs.get("_is_unicity_key", False) else 0
    return base + offset, base + offset + cardinality


def _draw_floats(rng, n, mu, sigma, distribution):
    """Draw ``n`` floats from the chosen distribution.

    - ``"normal"`` (default) → ``N(mu, sigma)`` — the historical behaviour, byte-identical.
    - ``"lognormal"`` → a positive, right-skewed draw whose **median is ``mu`` (real
      units)** and whose log-space spread is ``sigma`` (a shape parameter, typically
      ~0.3-1.5). Suited to quantities like notionals/prices that span orders of magnitude.
    """
    if distribution == "lognormal":
        if mu <= 0:
            raise ValueError(f"lognormal distribution needs mu (median) > 0, got {mu}")
        return rng.lognormal(mean=float(np.log(mu)), sigma=sigma, size=n)
    if distribution != "normal":
        raise ValueError(f"unknown distribution {distribution!r}; expected 'normal' or 'lognormal'")
    return rng.normal(loc=mu, scale=sigma, size=n)


def _draw_float_array(rng, rows, size, mu, sigma, distribution, path):
    """Draw ``rows`` float arrays of length ``size`` from the chosen distribution.

    Routes through ``_draw_floats`` on a flat ``rows*size`` draw then reshapes —
    byte-identical to ``rng.normal(size=(rows, size))`` for the default
    ``distribution="normal"``, so existing array output is unchanged.

    - ``path="iid"`` (default) → each element an independent draw.
    - ``path="walk"`` → a per-row cumulative series (``cumsum`` along the array
      axis): the elements form a time path of per-step increments, suited to a
      PnL/returns series. (For a signed/``normal`` family this is a random walk;
      for ``lognormal`` the positive increments make a monotone series.)
    """
    flat = _draw_floats(rng, rows * size, mu, sigma, distribution)
    arr = np.asarray(flat, dtype=float).reshape(rows, size)
    if path == "walk":
        return np.cumsum(arr, axis=1)
    if path != "iid":
        raise ValueError(f"unknown path {path!r}; expected 'iid' or 'walk'")
    return arr


def _table_column(table, col):
    """Read column ``col`` from the partial ``_table`` (polars or pandas frame),
    or ``None`` if the frame/column is absent or empty. Frame-agnostic so a
    conditional generator can read a parent column without importing polars."""
    if table is None:
        return None
    columns = getattr(table, "columns", None)
    if columns is None or col not in columns:
        return None
    series = table[col]
    if len(series) == 0:
        return None
    to_numpy = getattr(series, "to_numpy", None)
    return to_numpy() if to_numpy is not None else np.asarray(series)


class SingleValues:
    def __init__(self) -> None:
        self._generator_description = "generates a unique value for the whole column"

    def call(self, _number_of_rows_per_batch: int, value, **kwargs):
        return np.full(_number_of_rows_per_batch, value)


class ValuesFromFile:
    def __init__(self) -> None:
        self._generator_description = "get values from a column in a csv or parquet file"

    def call(
        self,
        _batch_number: int,
        _number_of_rows_per_batch: int,
        file_path: str,
        column_name: str,
        **kwargs,
    ):
        if ".parquet" in file_path or ".csv" in file_path:
            df_temp = pd.read_csv(file_path)
        else:
            raise Exception("unknown file format")

        start_idx = min(
            (_batch_number - 1) * _number_of_rows_per_batch,
            df_temp.index[-1] - _number_of_rows_per_batch + 1,
        )
        end_idx = min((_batch_number) * _number_of_rows_per_batch, df_temp.index[-1] + 1)
        df_temp = df_temp.iloc[start_idx:end_idx]

        return df_temp[column_name]


class ListOfValues:
    def __init__(self) -> None:
        self._generator_description = "generates a random values from defined list"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _rng_from(kwargs).choice(
            list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
        )


class UniqueIncrementalIntegers:
    def __init__(self) -> None:
        self._generator_description = "generates unique incremental integers"

    def call(self, _batch_number: int, _number_of_rows_per_batch: int, **kwargs):
        return np.arange(
            (_batch_number - 1) * _number_of_rows_per_batch + 1,
            _batch_number * _number_of_rows_per_batch + 1,
        )


class UniqueIncrementalStrings:
    def __init__(self) -> None:
        self._generator_description = "generates unique incremental strings"

    def call(self, prefix: str, _batch_number: int, _number_of_rows_per_batch: int, **kwargs):
        return np.char.add(
            np.full(_number_of_rows_per_batch, prefix),
            np.arange(
                (_batch_number - 1) * _number_of_rows_per_batch + 1,
                _batch_number * _number_of_rows_per_batch + 1,
            ).astype(str),
        )


class UniqueIncrementalDates:
    def __init__(self) -> None:
        self._generator_description = "generates unique incremental dates"

    def call(self, _batch_number: int, _start_date: str, _number_of_rows_per_batch: int, **kwargs):
        return np.arange(
            np.datetime64(_start_date)
            + np.timedelta64((_batch_number - 1) * _number_of_rows_per_batch, "D"),
            np.datetime64(_start_date)
            + np.timedelta64(_batch_number * _number_of_rows_per_batch, "D"),
        )


class RandomStrings:
    def __init__(self) -> None:
        self._generator_description = "generates random strings. You can also specify a list_of_values=['bla',...,'blabla'] with its associated probabilities=[0.1,...,0.3]"

    def call(
        self,
        prefix: str,
        _batch_number: int,
        _number_of_rows_per_batch: int,
        cardinality: int | None = None,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        else:
            low, high = _cardinality_range(cardinality, _batch_number, kwargs, base=1)
            return np.char.add(
                np.full(_number_of_rows_per_batch, prefix),
                _rng_from(kwargs)
                .integers(low=low, high=high, size=_number_of_rows_per_batch)
                .astype(str),
            )


class RandomIntegers:
    def __init__(self) -> None:
        self._generator_description = "generates random integers. You can also specify a list_of_values=[1,...,17] with its associated probabilities=[0.1,...,0.3]"

    def call(
        self,
        _batch_number: int,
        _number_of_rows_per_batch: int,
        cardinality: int | None = None,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        else:
            low, high = _cardinality_range(cardinality, _batch_number, kwargs)
            return _rng_from(kwargs).integers(low=low, high=high, size=_number_of_rows_per_batch)


class RandomFloats:
    def __init__(self) -> None:
        self._generator_description = "generates random floats — distribution='normal' (N(mu, sigma), default) or 'lognormal' (positive, right-skewed; mu is the median in real units, sigma the log-space spread). You can also specify a list_of_values=[2.7182,...,3.1415] with its associated probabilities=[0.1,...,0.3]"

    def call(
        self,
        _number_of_rows_per_batch: int,
        mu: float = 1000,
        sigma: float = 100,
        distribution: str = "normal",
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        return _draw_floats(_rng_from(kwargs), _number_of_rows_per_batch, mu, sigma, distribution)


class BrownianFloats:
    def __init__(self) -> None:
        self._generator_description = "generates random floats (distribution='normal' default, or 'lognormal' — mu is the median in real units), and propagates the values across time by applying small shocks"

    def call(
        self,
        _number_of_rows_per_batch: int,
        mu: float = 1000,
        sigma: float = 100,
        distribution: str = "normal",
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        return _draw_floats(_rng_from(kwargs), _number_of_rows_per_batch, mu, sigma, distribution)

    def recall(self, input_array: np.ndarray, drift: float = 0, noise: float = 10, **kwargs):
        shocks = self.call(
            mu=drift,
            sigma=noise,
            _number_of_rows_per_batch=len(input_array),
            _rng=kwargs.get("_rng"),
        )
        return input_array + shocks


class ConditionalFloats:
    def __init__(self) -> None:
        self._generator_description = (
            "generates floats whose distribution depends on another column's value: "
            "reads by_column from the partial table and, per row, draws from the "
            "matching entry in cases={value: {distribution, mu, sigma}} (falling back "
            "to default). E.g. a notional whose scale depends on asset_class. Falls "
            "back to a flat draw from default when by_column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        by_column: str = "asset_class",
        cases: dict | None = None,
        default: dict | None = None,
        **kwargs,
    ):
        rng = _rng_from(kwargs)
        default = default or {"distribution": "normal", "mu": 1000, "sigma": 100}
        parent = _table_column(kwargs.get("_table"), by_column)
        if parent is None:  # parent column absent -> a single flat draw from default
            return _draw_floats(
                rng,
                _number_of_rows_per_batch,
                default["mu"],
                default["sigma"],
                default.get("distribution", "normal"),
            )
        cases = cases or {}
        out = np.empty(_number_of_rows_per_batch, dtype=float)
        for value in np.unique(parent):  # vectorised per distinct parent (sorted -> deterministic)
            mask = parent == value
            spec = cases.get(str(value), default)
            out[mask] = _draw_floats(
                rng,
                int(mask.sum()),
                spec["mu"],
                spec["sigma"],
                spec.get("distribution", "normal"),
            )
        return out


class CurrentAsofDates:
    def __init__(self) -> None:
        self._generator_description = "generates an asof date"

    def call(self, _date: str, _number_of_rows_per_batch: int, **kwargs):

        date = datetime.datetime.strptime(_date, "%Y-%m-%d")
        return np.full(_number_of_rows_per_batch, date)

    def recall(self, input_array, _date: str, **kwargs):
        date = datetime.datetime.strptime(_date, "%Y-%m-%d")
        return np.full(len(input_array), date)


class RandomDates:
    def __init__(self) -> None:
        self._generator_description = "generates random dates. You can also specify a list_of_values=['2024-04-01',...,'2024-04-22'] with its associated probabilities=[0.1,...,0.3]"

    def call(
        self,
        _batch_number: int,
        _start_date: str,
        _number_of_rows_per_batch: int,
        cardinality: int | None = None,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        else:
            # unicity keys get a disjoint date window per batch (cross-batch uniqueness);
            # other columns draw from the same `cardinality`-day window every batch.
            assert cardinality is not None  # required when list_of_values is not given
            batch_offset = (_batch_number - 1) if kwargs.get("_is_unicity_key", False) else 0
            start_dt = datetime.datetime.strptime(_start_date, "%Y-%m-%d") + datetime.timedelta(
                days=cardinality * batch_offset
            )
            end_dt = start_dt + datetime.timedelta(days=cardinality - 1)
            date_range = (end_dt - start_dt).days + 1
            random_days = _rng_from(kwargs).integers(date_range, size=_number_of_rows_per_batch)
            return pd.to_datetime(start_dt) + pd.to_timedelta(random_days, unit="d")


class RandomTimeStamps:
    def __init__(self) -> None:
        self._generator_description = "generates random timestamp"

    def call(
        self,
        _batch_number: int,
        _start_date: str,
        _end_date: str,
        _number_of_rows_per_batch: int,
        start_time: str = "09:00:00",
        stop_time: str = "18:00:00",
        ticks_in_seconds: int = 60,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return _rng_from(kwargs).choice(
                list_of_values, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )
        else:
            list_of_timestamps = np.arange(
                np.datetime64(f"{_start_date} {start_time}"),
                np.datetime64(f"{_end_date} {stop_time}"),
                np.timedelta64(ticks_in_seconds, "s"),
            ).tolist()
            return _rng_from(kwargs).choice(
                list_of_timestamps, size=_number_of_rows_per_batch, replace=True, p=probabilities
            )

    def recall(
        self,
        input_array: np.ndarray,
        _batch_number: int,
        _start_date: str,
        _end_date: str,
        start_time: str = "09:00:00",
        stop_time: str = "18:00:00",
        ticks_in_seconds: int = 60,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return self.call(
            _batch_number=_batch_number,
            _start_date=_start_date,
            _end_date=_end_date,
            _number_of_rows_per_batch=len(input_array),
            start_time=start_time,
            stop_time=stop_time,
            ticks_in_seconds=ticks_in_seconds,
            list_of_values=list_of_values,
            probabilities=probabilities,
            _rng=kwargs.get("_rng"),
        )

    def cardinality(
        self,
        _start_date: str,
        _end_date: str,
        start_time: str = "09:00:00",
        stop_time: str = "18:00:00",
        ticks_in_seconds: int = 60,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        if list_of_values:
            return len(list_of_values)
        else:
            return len(
                np.arange(
                    np.datetime64(f"{_start_date} {start_time}"),
                    np.datetime64(f"{_end_date} {stop_time}"),
                    np.timedelta64(ticks_in_seconds, "s"),
                ).tolist()
            )


class RandomArrayFloats:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a fixed-size array of floats per row — distribution='normal' "
            "(default) or 'lognormal' (mu=median, sigma=log-σ); path='iid' (independent "
            "elements, default) or 'walk' (a cumulative per-row series, e.g. a PnL path)"
        )

    def call(
        self,
        size_of_arrays: int,
        _number_of_rows_per_batch: int,
        mu: float = 1000,
        sigma: float = 100,
        distribution: str = "normal",
        path: str = "iid",
        **kwargs,
    ):
        return _draw_float_array(
            _rng_from(kwargs),
            _number_of_rows_per_batch,
            size_of_arrays,
            mu,
            sigma,
            distribution,
            path,
        ).tolist()


class BrownianArrayFloats:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a fixed-size array of floats per row (distribution='normal' default "
            "or 'lognormal'; path='iid' default or 'walk'), and propagates the values across "
            "dates by applying small shocks"
        )

    def call(
        self,
        size_of_arrays: int,
        _number_of_rows_per_batch: int,
        mu: float = 1000,
        sigma: float = 100,
        distribution: str = "normal",
        path: str = "iid",
        **kwargs,
    ):
        return _draw_float_array(
            _rng_from(kwargs),
            _number_of_rows_per_batch,
            size_of_arrays,
            mu,
            sigma,
            distribution,
            path,
        ).tolist()

    def recall(
        self,
        input_array: np.ndarray,
        size_of_arrays: int,
        mu: float,
        sigma: float,
        _number_of_rows_per_batch: int,
        **kwargs,
    ):
        shocks = self.call(
            size_of_arrays=size_of_arrays,
            mu=mu / 10,
            sigma=sigma / 10,
            _number_of_rows_per_batch=len(input_array),
            _rng=kwargs.get("_rng"),
        )
        return np.sum([input_array.tolist(), shocks], axis=0).tolist()


class ConditionalArrayFloats:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a fixed-size float array per row whose distribution depends on "
            "another column's value: reads by_column from the partial table and, per row, "
            "draws an array from the matching entry in cases={value: {distribution, mu, "
            "sigma}} (falling back to default). size_of_arrays and path are shared across "
            "cases. E.g. a PnL series whose volatility depends on asset_class. Falls back "
            "to a flat draw from default when by_column is absent."
        )

    def call(
        self,
        size_of_arrays: int,
        _number_of_rows_per_batch: int,
        by_column: str = "asset_class",
        cases: dict | None = None,
        default: dict | None = None,
        path: str = "iid",
        **kwargs,
    ):
        rng = _rng_from(kwargs)
        default = default or {"distribution": "normal", "mu": 0, "sigma": 1}
        parent = _table_column(kwargs.get("_table"), by_column)
        if parent is None:  # parent column absent -> a single flat draw from default
            return _draw_float_array(
                rng,
                _number_of_rows_per_batch,
                size_of_arrays,
                default["mu"],
                default["sigma"],
                default.get("distribution", "normal"),
                path,
            ).tolist()
        cases = cases or {}
        out = np.empty((_number_of_rows_per_batch, size_of_arrays), dtype=float)
        for value in np.unique(parent):  # sorted -> deterministic
            mask = parent == value
            spec = cases.get(str(value), default)
            out[mask] = _draw_float_array(
                rng,
                int(mask.sum()),
                size_of_arrays,
                spec["mu"],
                spec["sigma"],
                spec.get("distribution", "normal"),
                path,
            )
        return out.tolist()


class RandomMultivariateArrayFloats:
    def __init__(self) -> None:
        self._generator_description = "generates random array of floats from a normal distribution with random mean and variance for each row"

    def call(
        self,
        size_of_arrays: int,
        mean_of_means: float,
        mean_of_stdevs: float,
        stdev_of_means: float,
        stdev_of_stdevs: float,
        _number_of_rows_per_batch: int,
        **kwargs,
    ):
        # array fact normal dist with random mean and variance for each row
        means = np.absolute(
            _rng_from(kwargs).normal(mean_of_means, stdev_of_means, _number_of_rows_per_batch)
        )
        stdevs = np.absolute(
            _rng_from(kwargs).normal(mean_of_stdevs, stdev_of_stdevs, _number_of_rows_per_batch)
        )
        cov = np.diag(stdevs)

        # array fact normal dist
        return np.transpose(
            _rng_from(kwargs).multivariate_normal(means, cov, size_of_arrays)
        ).tolist()

    def recall(
        self,
        input_array: np.ndarray,
        size_of_arrays: int,
        mean_of_means: float,
        mean_of_stdevs: float,
        stdev_of_means: float,
        stdev_of_stdevs: float,
        **kwargs,
    ):
        shocks = self.call(
            size_of_arrays,
            mean_of_means / 10,
            mean_of_stdevs / 10,
            stdev_of_means / 10,
            stdev_of_stdevs / 10,
            _number_of_rows_per_batch=len(input_array),
            _rng=kwargs.get("_rng"),
        )
        return np.sum([input_array.tolist(), shocks], axis=0).tolist()


class RandomArrayStrings:
    def __init__(self) -> None:
        self._generator_description = "generates random array of strings"

    def call(
        self,
        prefix: str,
        size_of_arrays: int,
        cardinality: float,
        _number_of_rows_per_batch: int,
        _batch_number: int,
        **kwargs,
    ):
        # array columns are never unicity keys: draw from the same pool of
        # `cardinality` values every batch (total distinct == cardinality)
        cardinality = int(cardinality)
        list_of_values = np.char.add(
            np.full(cardinality, prefix),
            np.arange(1, cardinality + 1).astype(str),
        )
        return (
            _rng_from(kwargs)
            .choice(list_of_values, size=(_number_of_rows_per_batch, size_of_arrays))
            .tolist()
        )


class RandomArrayUniqueStrings:
    def __init__(self) -> None:
        self._generator_description = "generates random array of unique, sorted strings per row"

    def call(
        self,
        prefix: str,
        size_of_arrays: int,
        cardinality: float,
        _number_of_rows_per_batch: int,
        _batch_number: int,
        **kwargs,
    ):
        cardinality = int(cardinality)

        if size_of_arrays > cardinality:
            raise ValueError(
                f"size_of_arrays ({size_of_arrays}) cannot be greater than "
                f"cardinality ({cardinality}) when generating unique elements."
            )

        # array columns are never unicity keys: same pool every batch
        list_of_values = np.char.add(
            np.full(cardinality, prefix), np.arange(1, cardinality + 1).astype(str)
        )

        # 1. Generate a 2D grid of random numbers
        random_noise = _rng_from(kwargs).random((_number_of_rows_per_batch, cardinality))

        # 2. Get unique random indices per row
        unique_indices = np.argsort(random_noise, axis=1)[:, :size_of_arrays]

        # 3. ADDED: Sort the indices row-by-row so the final strings come out sorted
        unique_indices = np.sort(unique_indices, axis=1)

        # 4. Index the pool of values
        return list_of_values[unique_indices].tolist()


class UniqueArrayIncrementalStrings:
    def __init__(self) -> None:
        self._generator_description = (
            "generates incremental arrays of unique strings where each row is identical"
        )

    def call(
        self,
        prefix: str,
        size_of_arrays: int,
        cardinality: float,
        _number_of_rows_per_batch: int,
        _batch_number: int,
        **kwargs,
    ):
        cardinality = int(cardinality)

        if size_of_arrays > cardinality:
            raise ValueError(
                f"size_of_arrays ({size_of_arrays}) cannot be greater than "
                f"cardinality ({cardinality})."
            )

        # array columns are never unicity keys: same `cardinality` pool every batch
        list_of_values = np.char.add(
            np.full(cardinality, prefix), np.arange(1, cardinality + 1).astype(str)
        )

        # 2. Slice the first 'size_of_arrays' elements (guarantees sorted + unique)
        single_incremental_row = list_of_values[:size_of_arrays]

        # 3. Duplicate this single row vertically for '_number_of_rows_per_batch'
        result_matrix = np.tile(single_incremental_row, (_number_of_rows_per_batch, 1))

        return result_matrix.tolist()


class IncrementalArrayDates:
    def __init__(self) -> None:
        self._generator_description = "array of incremetal dates"

    def call(self, _start_date: str, size_of_arrays: int, _number_of_rows_per_batch: int, **kwargs):
        list_of_dates = (
            np.arange(
                np.datetime64(_start_date),
                np.datetime64(_start_date) + np.timedelta64(size_of_arrays, "D"),
            )
            .astype(str)
            .tolist()
        )
        return np.full((_number_of_rows_per_batch, size_of_arrays), list_of_dates).tolist()


class InternalMappings:
    def __init__(self) -> None:
        self._generator_description = "generates values for a column based on the values of other columns within the same table"

    def call(self, table: pd.DataFrame, mapping: dict, default_value=None, **kwargs):
        """
        example of mapping: {"(table['book']=='b2') & (table['ctp']=='c3')":"EQD", "table['book']=='b2'": "FI"}
        important: always use the keyword table for defining the table when defining the mapping
        """
        condlist = list(mapping.keys())
        choicelist = list(mapping.values())

        for idx in range(len(condlist)):
            condlist[idx] = eval(condlist[idx])

        return np.select(condlist=condlist, choicelist=choicelist, default=default_value)

    def recall(self, table: pd.DataFrame, mapping: dict, default_value=None, **kwargs):
        return self.call(table=table, mapping=mapping, default_value=default_value)


class Evals:
    def __init__(self) -> None:
        self._generator_description = "generates values based on a user defined function"

    def call(self, udf: str, **kwargs):
        """
        example of udf: "np.random.normal(loc=100, scale=2, size=1000)"
        """
        return eval(udf)

    def recall(self, udf: str, **kwargs):
        return self.call(udf=udf)


class DecrementalArrayDates:
    """
    Generates an array(date) where the first element equals _date and subsequent
    elements go backwards in time (one day per step).
    On recall(), the array shifts so the first element equals the new _date,
    keeping the array anchored to the current asof date across all generated dates.
    """

    def __init__(self) -> None:
        self._generator_description = (
            "generates an array of dates starting at _date and going backwards; "
            "the first element always equals the current asof date"
        )

    def call(self, _date: str, size_of_arrays: int, _number_of_rows_per_batch: int, **kwargs):
        anchor = np.datetime64(_date)
        list_of_dates = (anchor - np.arange(size_of_arrays)).astype(str).tolist()
        return np.full((_number_of_rows_per_batch, size_of_arrays), list_of_dates).tolist()

    def recall(self, input_array, _date: str, size_of_arrays: int, **kwargs):
        anchor = np.datetime64(_date)
        list_of_dates = (anchor - np.arange(size_of_arrays)).astype(str).tolist()
        return np.full((len(input_array), size_of_arrays), list_of_dates).tolist()


class PreviousAsofDate:
    """
    Generates a date column that always contains the previous asof date (_previous_date).
    On the first date, _previous_date == _start_date (same as the current date).
    On subsequent dates, _previous_date is the previous business day.
    This ensures Date_previous is consistent with the actual previous date in the dataset.
    """

    def __init__(self) -> None:
        self._generator_description = (
            "generates the previous asof date (_previous_date) for all rows"
        )

    def call(self, _previous_date: str, _number_of_rows_per_batch: int, **kwargs):
        date = datetime.datetime.strptime(_previous_date, "%Y-%m-%d")
        return np.full(_number_of_rows_per_batch, date)

    def recall(self, input_array: np.ndarray, _previous_date: str, **kwargs):
        date = datetime.datetime.strptime(_previous_date, "%Y-%m-%d")
        return np.full(len(input_array), date)


class PreviousFloats:
    """
    Generates a float column that tracks the previous date's values.
    On call() (first date): generates fresh random values from a normal distribution.
    On recall() (subsequent dates): returns the current date's corresponding column values
    (i.e., the values that were 'current' on the previous date are now 'previous').
    """

    def __init__(self) -> None:
        self._generator_description = (
            "generates a float column representing the previous date's values. "
            "On the first date, generates fresh random floats. "
            "On subsequent dates, returns the snapshot of the current-date column."
        )

    def call(self, _number_of_rows_per_batch: int, mu: float = 1000, sigma: float = 100, **kwargs):
        return _rng_from(kwargs).normal(loc=mu, scale=sigma, size=_number_of_rows_per_batch)

    def recall(self, input_array: np.ndarray, **kwargs):
        # On date N, the 'previous' values are what was 'current' on date N-1.
        # The generator framework passes input_array = the current state of this column
        # (which was set to the date N-1 'current' values by the previous recall).
        # We simply keep those values unchanged so they reflect the previous date snapshot.
        return np.array(input_array)


class PreviousArrayFloats:
    """
    Generates an array(float64) column that tracks the previous date's values.
    On call() (first date): generates fresh random float arrays from a normal distribution.
    On recall() (subsequent dates): returns the snapshot of the current-date array column.
    """

    def __init__(self) -> None:
        self._generator_description = (
            "generates an array(float64) column representing the previous date's values. "
            "On the first date, generates fresh random float arrays. "
            "On subsequent dates, returns the snapshot of the current-date array column."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        size_of_arrays: int = 100,
        mu: float = 1000,
        sigma: float = 500,
        **kwargs,
    ):
        return (
            _rng_from(kwargs)
            .normal(mu, sigma, size=(_number_of_rows_per_batch, size_of_arrays))
            .tolist()
        )

    def recall(self, input_array, **kwargs):
        # On date N, 'previous' arrays are what were 'current' on date N-1.
        # Return unchanged so they reflect the previous date snapshot.
        if isinstance(input_array, pd.Series):
            return input_array.tolist()
        return list(input_array)
