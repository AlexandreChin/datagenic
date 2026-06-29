"""Financial-markets domain column generators.

Same contract as ``datagenic.generators``: each class has ``call(self,
_number_of_rows_per_batch, ..., **kwargs)``, draws randomness from the injected
``_rng`` (never global numpy state), and returns a numpy array / list. All
generators here are **stateless** (no ``recall``) — a reference attribute like
asset class or ISIN does not change day to day, so the engine keeps the
first-date values across the whole date range, which is exactly what we want.

Two families:

* **Coherent hierarchy** — ``FinancialInstrumentType`` and
  ``FinancialRiskFactorGroup`` read the partial ``_table`` (the columns already
  generated this unit) and sample a child valid for *each row's* asset class, so
  ``(asset_class, instrument_type)`` pairs are always taxonomy-valid. They fall
  back to a flat draw if the parent column is absent.
* **Independent dimensions, identifiers, org** — currency/tenor/rating/sector,
  ISIN/CUSIP/SEDOL with valid check digits, and counterparty/book/desk names.

Determinism: every draw goes through the per-unit ``_rng``; key-column
identifiers use the same disjoint-per-batch offset trick as
``datagenic.generators`` so codes stay globally unique across batches/shards.
"""

import numpy as np

from datagenic import _finance_data as fd


def _rng_from(kwargs) -> np.random.Generator:
    """Return the framework-injected RNG, or a fresh one for direct calls."""
    return kwargs.get("_rng") or np.random.default_rng()


def _column_values(table, col):
    """Read column ``col`` from the partial ``_table`` (a polars or pandas frame),
    or ``None`` if the frame/column is absent or empty. Frame-agnostic so this
    module needs no polars import."""
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


def _first_present(table, candidates):
    """First of ``candidates`` present (non-empty) in ``_table``, else ``None``."""
    for name in candidates:
        values = _column_values(table, name)
        if values is not None:
            return values
    return None


def _effective_pool(vocab, kwargs, list_of_values=None):
    """The value pool to sample from, capped to ``cardinality`` distinct values
    when the engine (or the user) assigned one. A fixed-domain categorical cannot
    exceed its vocabulary, so the cap is ``min(cardinality, len(pool))`` — we slice
    the head of the pool deterministically rather than fabricate new values."""
    pool = list(list_of_values) if list_of_values else list(vocab)
    cardinality = kwargs.get("cardinality")
    if cardinality is not None:
        pool = pool[: max(1, min(int(cardinality), len(pool)))]
    return pool


def _declared_cardinality(vocab, list_of_values, kwargs):
    """The number of distinct values this generator will actually produce — its
    vocabulary size, capped by an assigned ``cardinality``. The engine consults
    this (via a generator's ``cardinality()`` method) to budget the *other* unicity
    keys correctly, instead of over-allocating distinct values this column can
    never deliver."""
    n = len(list_of_values) if list_of_values else len(vocab)
    cardinality = kwargs.get("cardinality")
    return max(1, min(int(cardinality), n)) if cardinality is not None else n


def _sample(vocab, n, kwargs, list_of_values=None, probabilities=None):
    """Sample ``n`` values from ``list_of_values`` (if given) else ``vocab``,
    honouring an assigned ``cardinality`` as a cap on distinct values."""
    pool = _effective_pool(vocab, kwargs, list_of_values)
    # probabilities apply only to a full, untruncated user-supplied list
    p = probabilities if (probabilities is not None and len(probabilities) == len(pool)) else None
    return _rng_from(kwargs).choice(pool, size=n, replace=True, p=p)


def _sample_by_parent(parent_values, children_by_parent, all_children, n, rng):
    """For each row, sample a child valid for that row's parent value.

    Vectorised per distinct parent (cost is O(distinct parents), not O(rows)).
    Rows whose parent has no entry in the map draw from ``all_children``.
    """
    result = np.empty(n, dtype=object)
    for parent in np.unique(parent_values):
        mask = parent_values == parent
        children = children_by_parent.get(str(parent), all_children)
        result[mask] = rng.choice(children, size=int(mask.sum()))
    return result


# ── Coherent hierarchy ───────────────────────────────────────────────────────


class FinancialAssetClass:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a financial asset class (Rates, Credit, Equity, FX, "
            "Commodity, Inflation). Optionally restrict/weight via "
            "list_of_values=[...] and probabilities=[...]."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.ASSET_CLASSES,
            _number_of_rows_per_batch,
            kwargs,
            list_of_values,
            probabilities,
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.ASSET_CLASSES, list_of_values, kwargs)


class FinancialInstrumentType:
    def __init__(self) -> None:
        self._generator_description = (
            "generates an instrument type coherent with each row's asset class "
            "(reads the asset_class_column from the partial table). Falls back to "
            "a flat draw over all instrument types if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        asset_class_column: str = "asset_class",
        **kwargs,
    ):
        parent = _column_values(kwargs.get("_table"), asset_class_column)
        if parent is None:
            return _sample(fd.ALL_INSTRUMENT_TYPES, _number_of_rows_per_batch, kwargs)
        return _sample_by_parent(
            parent,
            fd.INSTRUMENT_TYPES_BY_ASSET_CLASS,
            fd.ALL_INSTRUMENT_TYPES,
            _number_of_rows_per_batch,
            _rng_from(kwargs),
        )


class FinancialRiskFactorGroup:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a risk factor group coherent with each row's asset class "
            "(reads the asset_class_column from the partial table). Falls back to "
            "a flat draw over all risk factor groups if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        asset_class_column: str = "asset_class",
        **kwargs,
    ):
        parent = _column_values(kwargs.get("_table"), asset_class_column)
        if parent is None:
            return _sample(fd.ALL_RISK_FACTOR_GROUPS, _number_of_rows_per_batch, kwargs)
        return _sample_by_parent(
            parent,
            fd.RISK_FACTOR_GROUPS_BY_ASSET_CLASS,
            fd.ALL_RISK_FACTOR_GROUPS,
            _number_of_rows_per_batch,
            _rng_from(kwargs),
        )


# ── Independent market dimensions ────────────────────────────────────────────


class Currency:
    def __init__(self) -> None:
        self._generator_description = "generates an ISO 4217 currency code"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.CURRENCIES, _number_of_rows_per_batch, kwargs, list_of_values, probabilities
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.CURRENCIES, list_of_values, kwargs)


class CurrencyPair:
    def __init__(self) -> None:
        self._generator_description = "generates an FX currency pair (e.g. EUR/USD)"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.CURRENCY_PAIRS, _number_of_rows_per_batch, kwargs, list_of_values, probabilities
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.CURRENCY_PAIRS, list_of_values, kwargs)


class Tenor:
    def __init__(self) -> None:
        self._generator_description = "generates a tenor / maturity bucket (e.g. 5Y)"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(fd.TENORS, _number_of_rows_per_batch, kwargs, list_of_values, probabilities)

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.TENORS, list_of_values, kwargs)


class CreditRating:
    def __init__(self) -> None:
        self._generator_description = "generates a credit rating (scale='SP' or 'Moody')"

    def call(
        self,
        _number_of_rows_per_batch: int,
        scale: str = "SP",
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        vocab = fd.MOODYS_RATINGS if str(scale).lower().startswith("mood") else fd.SP_RATINGS
        return _sample(vocab, _number_of_rows_per_batch, kwargs, list_of_values, probabilities)

    def cardinality(self, scale: str = "SP", list_of_values=None, **kwargs):
        vocab = fd.MOODYS_RATINGS if str(scale).lower().startswith("mood") else fd.SP_RATINGS
        return _declared_cardinality(vocab, list_of_values, kwargs)


class GICSSector:
    def __init__(self) -> None:
        self._generator_description = "generates a GICS sector"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.GICS_SECTORS, _number_of_rows_per_batch, kwargs, list_of_values, probabilities
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.GICS_SECTORS, list_of_values, kwargs)


# ── Identifiers (valid check digits) ─────────────────────────────────────────


def _char_value(ch: str) -> int:
    """Digit -> its value; letter -> A=10 .. Z=35 (ISIN/CUSIP/SEDOL convention)."""
    return int(ch) if ch.isdigit() else ord(ch.upper()) - 55


def _isin_check_digit(body11: str) -> str:
    """ISIN check digit: expand chars to digits, then Luhn (double every second
    digit starting from the rightmost)."""
    digits = "".join(str(_char_value(ch)) for ch in body11)
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return str((10 - total % 10) % 10)


def _cusip_check_digit(body8: str) -> str:
    """CUSIP check digit (mod 10, doubling every second position)."""
    total = 0
    for i, ch in enumerate(body8):
        v = _char_value(ch)
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def _sedol_check_digit(body6: str) -> str:
    """SEDOL check digit (weighted mod 10, weights 1,3,1,7,3,9)."""
    weights = (1, 3, 1, 7, 3, 9)
    total = sum(_char_value(ch) * w for ch, w in zip(body6, weights, strict=True))
    return str((10 - total % 10) % 10)


def _serials(n: int, _batch_number: int, is_key: bool, width: int, rng) -> list[str]:
    """Zero-padded numeric bodies. Key columns get a disjoint, incremental range
    per batch (globally unique); non-key columns draw at random (may repeat)."""
    if is_key:
        start = (_batch_number - 1) * n + 1
        nums = np.arange(start, start + n)
    else:
        nums = rng.integers(0, 10**width, size=n)
    modulo = 10**width
    return [str(int(x) % modulo).zfill(width) for x in nums]


class ISIN:
    def __init__(self) -> None:
        self._generator_description = (
            "generates synthetic ISINs (2-letter country + 9-digit body + Luhn "
            "check digit). Unicity-key columns are globally unique per batch."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        _batch_number: int = 1,
        country: str | None = None,
        **kwargs,
    ):
        rng = _rng_from(kwargs)
        is_key = kwargs.get("_is_unicity_key", False)
        bodies = _serials(_number_of_rows_per_batch, _batch_number, is_key, 9, rng)
        if country is not None:
            countries = [country] * _number_of_rows_per_batch
        else:
            countries = rng.choice(fd.ISIN_COUNTRIES, size=_number_of_rows_per_batch)
        out = []
        for ctry, body in zip(countries, bodies, strict=True):
            base = f"{ctry}{body}"
            out.append(base + _isin_check_digit(base))
        return np.array(out, dtype=object)


class CUSIP:
    def __init__(self) -> None:
        self._generator_description = (
            "generates synthetic CUSIPs (8-char body + mod-10 check digit). "
            "Unicity-key columns are globally unique per batch."
        )

    def call(self, _number_of_rows_per_batch: int, _batch_number: int = 1, **kwargs):
        rng = _rng_from(kwargs)
        is_key = kwargs.get("_is_unicity_key", False)
        bodies = _serials(_number_of_rows_per_batch, _batch_number, is_key, 8, rng)
        return np.array([b + _cusip_check_digit(b) for b in bodies], dtype=object)


class SEDOL:
    def __init__(self) -> None:
        self._generator_description = (
            "generates synthetic SEDOLs (6-digit body + weighted mod-10 check "
            "digit). Unicity-key columns are globally unique per batch."
        )

    def call(self, _number_of_rows_per_batch: int, _batch_number: int = 1, **kwargs):
        rng = _rng_from(kwargs)
        is_key = kwargs.get("_is_unicity_key", False)
        bodies = _serials(_number_of_rows_per_batch, _batch_number, is_key, 6, rng)
        return np.array([b + _sedol_check_digit(b) for b in bodies], dtype=object)


# ── Composed human-readable names ────────────────────────────────────────────


class InstrumentName:
    def __init__(self) -> None:
        self._generator_description = (
            "composes a human-readable instrument name from the row's currency, "
            "instrument type and tenor columns (e.g. 'EUR IR Swap 5Y')."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        currency_column: str = "ccy",
        instrument_type_column: str = "instrument_type",
        tenor_column: str = "tenor",
        **kwargs,
    ):
        table = kwargs.get("_table")
        ccy = _first_present(table, [currency_column, "ccy", "currency", "cur"])
        instr = _first_present(table, [instrument_type_column, "instrument_type", "product_type"])
        tenor = _first_present(table, [tenor_column, "tenor", "maturity"])
        out = []
        for i in range(_number_of_rows_per_batch):
            parts = [str(values[i]) for values in (ccy, instr, tenor) if values is not None]
            out.append(" ".join(parts) if parts else f"INSTR_{i + 1}")
        return np.array(out, dtype=object)


class RiskFactorName:
    def __init__(self) -> None:
        self._generator_description = (
            "composes a human-readable risk factor name from the row's currency, "
            "risk factor group and tenor columns (e.g. 'EUR IR Curve 10Y')."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        currency_column: str = "ccy",
        risk_factor_group_column: str = "risk_factor_group",
        tenor_column: str = "tenor",
        **kwargs,
    ):
        table = kwargs.get("_table")
        ccy = _first_present(table, [currency_column, "ccy", "currency", "cur"])
        rfg = _first_present(table, [risk_factor_group_column, "risk_factor_group"])
        tenor = _first_present(table, [tenor_column, "tenor", "maturity"])
        out = []
        for i in range(_number_of_rows_per_batch):
            parts = [str(values[i]) for values in (ccy, rfg, tenor) if values is not None]
            out.append(" ".join(parts) if parts else f"RF_{i + 1}")
        return np.array(out, dtype=object)


# ── Organisation / counterparty ──────────────────────────────────────────────


class Counterparty:
    def __init__(self) -> None:
        self._generator_description = "generates a (fictional) counterparty / legal entity name"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.COUNTERPARTIES, _number_of_rows_per_batch, kwargs, list_of_values, probabilities
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.COUNTERPARTIES, list_of_values, kwargs)


class TradingBook:
    def __init__(self) -> None:
        self._generator_description = "generates a trading book label (e.g. 'Rates Flow EMEA')"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(
            fd.TRADING_BOOKS, _number_of_rows_per_batch, kwargs, list_of_values, probabilities
        )

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.TRADING_BOOKS, list_of_values, kwargs)


class Desk:
    def __init__(self) -> None:
        self._generator_description = "generates a trading desk label (e.g. 'FX Options Desk')"

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(fd.DESKS, _number_of_rows_per_batch, kwargs, list_of_values, probabilities)

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(fd.DESKS, list_of_values, kwargs)
