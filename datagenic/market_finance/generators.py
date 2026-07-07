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

import hashlib
import re

import numpy as np

from datagenic.market_finance import _data as fd


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
        self._generator_description = (
            "generates a counterparty / legal entity name (real large-cap issuers from the "
            "major indices by default; sector & country available via COUNTERPARTY_METADATA "
            "and the CounterpartySector / CounterpartyCountry generators)"
        )

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


def _counterparty_attribute(n, counterparty_column, attr, vocab, kwargs):
    """Per row, map the row's counterparty (read from the partial ``_table``) to its
    ``attr`` in ``COUNTERPARTY_METADATA`` — so ``(counterparty, sector/country)`` is
    always coherent. Vectorised per distinct counterparty. Unknown names (or a missing
    parent column) draw from ``vocab`` on the ``_rng`` (deterministic, rare)."""
    rng = _rng_from(kwargs)
    parent = _column_values(kwargs.get("_table"), counterparty_column)
    if parent is None:
        return _sample(vocab, n, kwargs)
    meta = fd.COUNTERPARTY_METADATA
    out = np.empty(n, dtype=object)
    for value in np.unique(parent):  # sorted -> deterministic
        mask = parent == value
        info = meta.get(str(value))
        mapped = info.get(attr) if info else None
        out[mask] = mapped if mapped is not None else rng.choice(vocab, size=int(mask.sum()))
    return out


class CounterpartySector:
    def __init__(self) -> None:
        self._generator_description = (
            "generates the GICS sector coherent with each row's counterparty (reads the "
            "counterparty_column from the partial table and maps via COUNTERPARTY_METADATA). "
            "Falls back to a flat GICS-sector draw if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        counterparty_column: str = "counterparty",
        **kwargs,
    ):
        return _counterparty_attribute(
            _number_of_rows_per_batch, counterparty_column, "sector", fd.GICS_SECTORS, kwargs
        )


class CounterpartyCountry:
    def __init__(self) -> None:
        self._generator_description = (
            "generates the domicile country (ISO-2) coherent with each row's counterparty "
            "(reads the counterparty_column from the partial table and maps via "
            "COUNTERPARTY_METADATA). Falls back to a flat draw if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        counterparty_column: str = "counterparty",
        **kwargs,
    ):
        vocab = fd.COUNTERPARTY_COUNTRIES or fd.ISIN_COUNTRIES
        return _counterparty_attribute(
            _number_of_rows_per_batch, counterparty_column, "country", vocab, kwargs
        )


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


# ── Trade attributes (fixed-vocabulary categoricals) ─────────────────────────


def _fixed_vocab_generator(name, vocab, description):
    """Build a simple fixed-vocabulary categorical generator class (draws from
    ``vocab`` on the injected rng, honours list_of_values/probabilities/cardinality
    and declares ``cardinality()`` — same contract as Currency/Tenor/…)."""

    def call(
        self,
        _number_of_rows_per_batch: int,
        list_of_values: list | None = None,
        probabilities: list | None = None,
        **kwargs,
    ):
        return _sample(vocab, _number_of_rows_per_batch, kwargs, list_of_values, probabilities)

    def cardinality(self, list_of_values=None, **kwargs):
        return _declared_cardinality(vocab, list_of_values, kwargs)

    return type(
        name,
        (),
        {
            "__init__": lambda self: setattr(self, "_generator_description", description),
            "call": call,
            "cardinality": cardinality,
        },
    )


Side = _fixed_vocab_generator("Side", fd.SIDES, "generates a trade side (Buy / Sell)")
OptionType = _fixed_vocab_generator(
    "OptionType", fd.OPTION_TYPES, "generates an option type (Call / Put)"
)
ExerciseStyle = _fixed_vocab_generator(
    "ExerciseStyle",
    fd.EXERCISE_STYLES,
    "generates an option exercise style (European / American / Bermudan)",
)
PaymentFrequency = _fixed_vocab_generator(
    "PaymentFrequency", fd.PAYMENT_FREQUENCIES, "generates a payment / coupon frequency"
)
DayCountConvention = _fixed_vocab_generator(
    "DayCountConvention",
    fd.DAY_COUNT_CONVENTIONS,
    "generates a day-count convention (30/360, ACT/360, …)",
)
TradeStatus = _fixed_vocab_generator(
    "TradeStatus",
    fd.TRADE_STATUSES,
    "generates a trade lifecycle status (New / Amended / Live / …)",
)
Exchange = _fixed_vocab_generator(
    "Exchange", fd.EXCHANGES, "generates a trading-venue MIC (ISO 10383, e.g. XNYS, XLON)"
)


class ClearingHouse:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a clearing house / CCP coherent with each row's asset class "
            "(reads the asset_class_column from the partial table). Falls back to a flat "
            "draw over all CCPs if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        asset_class_column: str = "asset_class",
        **kwargs,
    ):
        parent = _column_values(kwargs.get("_table"), asset_class_column)
        if parent is None:
            return _sample(fd.ALL_CLEARING_HOUSES, _number_of_rows_per_batch, kwargs)
        return _sample_by_parent(
            parent,
            fd.CLEARING_HOUSES_BY_ASSET_CLASS,
            fd.ALL_CLEARING_HOUSES,
            _number_of_rows_per_batch,
            _rng_from(kwargs),
        )


# ── Coherent dates (trade / maturity / settlement) ───────────────────────────

_TENOR_UNIT_DAYS = {"D": 1, "W": 7, "M": 30, "Y": 365}


def _tenor_to_days(tenor) -> int:
    """Approximate calendar-day offset for a tenor string (ON, 1W, 3M, 5Y, …).
    Defaults to 1Y when unparseable."""
    t = str(tenor).strip().upper()
    if t in ("ON", "O/N", "T/N", "TN", "SN"):
        return 1
    m = re.fullmatch(r"(\d+)\s*([DWMY])", t)
    return int(m.group(1)) * _TENOR_UNIT_DAYS[m.group(2)] if m else 365


def _base_dates(table, trade_date_column, n, _date):
    """Row base dates: the trade_date column if present (as datetime64[D]), else the
    current as-of ``_date`` for every row."""
    trade = _column_values(table, trade_date_column)
    if trade is not None:
        return np.asarray(trade, dtype="datetime64[D]")
    return np.full(n, np.datetime64(_date, "D"))


class TradeDate:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a trade / booking date: a random business-day offset before the current "
            "as-of date. When the row's tenor is available (from `tenor_column` in the partial "
            "table) the offset is bounded by that tenor (capped at `lookback_days`) so that every "
            "trade is still live at the as-of date — maturity = trade_date + tenor lands on or "
            "after it, i.e. maturity >= as-of >= trade_date holds row-wise. Falls back to a flat "
            "offset within `lookback_days` when no tenor column is present."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        _date: str,
        lookback_days: int = 365,
        tenor_column: str = "tenor",
        **kwargs,
    ):
        base = np.datetime64(_date, "D")
        rng = _rng_from(kwargs)
        lookback = max(1, int(lookback_days))
        tenor = _column_values(kwargs.get("_table"), tenor_column)
        if tenor is not None:
            # bound each row's age by its own tenor (capped at lookback) so that
            # maturity = trade_date + tenor lands on or after the as-of date — every
            # trade stays live — while trades remain no older than `lookback_days`.
            tenor_days = np.array([_tenor_to_days(t) for t in tenor], dtype="int64")
            bound = np.minimum(tenor_days, lookback)
            offsets = rng.integers(0, bound + 1)  # per-row integer in [0, bound]
            return base - offsets.astype("timedelta64[D]")
        offsets = rng.integers(0, lookback, size=_number_of_rows_per_batch)
        return base - offsets.astype("timedelta64[D]")


class MaturityDate:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a maturity / expiry date coherent with the row's trade_date and tenor "
            "(maturity = trade_date + tenor). Reads trade_date_column and tenor_column from the "
            "partial table; falls back to the as-of date + 1Y when they're absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        _date: str,
        trade_date_column: str = "trade_date",
        tenor_column: str = "tenor",
        **kwargs,
    ):
        table = kwargs.get("_table")
        base = _base_dates(table, trade_date_column, _number_of_rows_per_batch, _date)
        tenor = _column_values(table, tenor_column)
        if tenor is None:
            days = np.full(_number_of_rows_per_batch, 365, dtype="int64")
        else:
            days = np.array([_tenor_to_days(t) for t in tenor], dtype="int64")
        return base + days.astype("timedelta64[D]")


class SettlementDate:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a settlement / value date = trade_date + settlement_lag_days (default T+2). "
            "Reads trade_date_column from the partial table; falls back to the as-of date."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        _date: str,
        trade_date_column: str = "trade_date",
        settlement_lag_days: int = 2,
        **kwargs,
    ):
        base = _base_dates(
            kwargs.get("_table"), trade_date_column, _number_of_rows_per_batch, _date
        )
        return base + np.timedelta64(int(settlement_lag_days), "D")


# ── Rolling book: a constant-size book whose trades roll at maturity ───────────
#
# Each row is a "slot" holding exactly one *live* trade at the current as-of date.
# A slot runs a deterministic renewal process (contiguous: a new trade is booked
# the day the previous matures), seeded date-independently from the global seed,
# batch and slot index. At any as-of date the slot emits the trade whose
# [trade_date, maturity) interval contains it — so every emitted trade is live
# (trade_date <= as-of < maturity), the book stays exactly N rows, short-tenor
# slots turn over (a new tenor + a new trade_id on every roll) while long-tenor
# trades persist. The schedule is a pure function of (global_seed, batch, slot),
# so it is identical on every date and in every shard/worker — byte-reproducible.
#
# These generators key off row index == slot, so the rolling scaffold skips the
# global key sort (``scale_skip_global_key_sort``) to keep that mapping stable
# across dates (``recall`` regenerates the frame row by row).

_ROLL_ID_DAY_RADIX = 100_000  # trade_id packing: room for the day-offset component
_ROLL_ID_EPOCH = np.datetime64("1970-01-01", "D")


def _rolling_rng(global_seed, batch, salt: str) -> np.random.Generator:
    """A date-independent RNG for one draw in a slot's renewal stream: keyed on the
    global seed, batch and salt only (never the as-of date), so a slot's schedule is
    identical on every date and in every shard."""
    digest = hashlib.sha256(f"{global_seed}|{batch}|{salt}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _rolling_schedule(n, elapsed, tenor_days_choices, global_seed, batch, probabilities=None):
    """For each of ``n`` slots return the trade active ``elapsed`` days after the start
    date, as (book_days, tenor_index): ``book_days`` is the active trade's booking offset
    (days since start; may be negative), ``tenor_index`` its tenor. Contiguous renewal
    gives ``book_days <= elapsed < book_days + tenor_days[tenor_index]`` — live by
    construction. Vectorised lockstep across slots; each slot's k-th trade uses draw k of
    a date-independent stream, so the sequence is fixed regardless of the as-of date."""
    m = len(tenor_days_choices)

    def draw_idx(k):
        rng = _rolling_rng(global_seed, batch, f"tenor:{k}")
        if probabilities is not None and len(probabilities) == m:
            return rng.choice(m, size=n, p=probabilities)
        return rng.integers(0, m, size=n)

    idx = draw_idx(0)
    tdays = tenor_days_choices[idx]
    # trade 0's age in [0, tenor_0) so it is already live at the start date
    age = np.floor(_rolling_rng(global_seed, batch, "age").random(n) * tdays).astype("int64")
    book = -age
    mat = book + tdays
    k = 0
    while np.any(mat <= elapsed):  # current trade matured on/before the as-of -> roll
        k += 1
        roll = mat <= elapsed
        nidx = draw_idx(k)
        ntd = tenor_days_choices[nidx]
        book = np.where(roll, mat, book)
        idx = np.where(roll, nidx, idx)
        tdays = np.where(roll, ntd, tdays)
        mat = book + tdays
    return book, idx


def _rolling_positions(n, repeat):
    """Number of distinct positions (slots) behind ``n`` rows when each position is
    repeated ``repeat`` times (a scenario dimension). ``repeat=1`` -> row == slot."""
    repeat = max(1, int(repeat))
    return -(-n // repeat), repeat  # ceil division


def _tile_positions(values, repeat, n):
    """Tile per-position ``values`` across the scenario dimension (block layout: each
    position's ``repeat`` rows are contiguous), then clip to ``n`` rows."""
    return np.repeat(values, repeat)[:n] if repeat > 1 else values


def _rolling_active(kwargs, tenors=None, probabilities=None, repeat=1):
    """Resolve the active trade for this unit from the injected framework args
    (date-independent seed). Returns (start_date, book_days, tenor_labels).

    ``repeat`` > 1 makes each position span ``repeat`` consecutive rows (a scenario
    dimension): the schedule is computed for ``ceil(n/repeat)`` positions and each is
    tiled ``repeat`` times, so a trade recurs across its scenario rows."""
    tenors = list(tenors) if tenors else list(fd.TENORS)
    n = int(kwargs["_number_of_rows_per_batch"])
    positions, repeat = _rolling_positions(n, repeat)
    start = np.datetime64(kwargs["_start_date"], "D")
    asof = np.datetime64(kwargs["_date"], "D")
    elapsed = int((asof - start).astype("timedelta64[D]").astype("int64"))
    global_seed = kwargs.get("_global_seed", 0)
    batch = int(kwargs.get("_batch_number", 1))
    choices = np.array([_tenor_to_days(t) for t in tenors], dtype="int64")
    book, idx = _rolling_schedule(positions, elapsed, choices, global_seed, batch, probabilities)
    book = _tile_positions(book, repeat, n)
    idx = _tile_positions(idx, repeat, n)
    return start, book, np.asarray(tenors, dtype=object)[idx]


class RollingTradeDate:
    def __init__(self) -> None:
        self._generator_description = (
            "rolling-book trade date: the booking date of the trade live at the as-of date "
            "in each slot's renewal process. A slot rolls at maturity so it always holds a "
            "live trade (trade_date <= as-of < maturity); the date-independent seed keeps "
            "the schedule stable across dates and shards. Bias short maturities via "
            "tenors=[...] / probabilities=[...]."
        )

    def _compute(self, kwargs, tenors, probabilities, repeat):
        start, book, _ = _rolling_active(kwargs, tenors, probabilities, repeat)
        return start + book.astype("timedelta64[D]")

    def call(self, _number_of_rows_per_batch: int, tenors=None, probabilities=None, repeat=1, **kwargs):
        kwargs["_number_of_rows_per_batch"] = _number_of_rows_per_batch
        return self._compute(kwargs, tenors, probabilities, repeat)

    def recall(self, input_array=None, tenors=None, probabilities=None, repeat=1, **kwargs):
        return self._compute(kwargs, tenors, probabilities, repeat)


class RollingTenor:
    def __init__(self) -> None:
        self._generator_description = (
            "rolling-book tenor: the tenor of the trade live at the as-of date in each slot's "
            "renewal process (a fresh tenor is drawn on every roll). Coherent with "
            "RollingTradeDate / RollingMaturityDate via the shared per-slot schedule."
        )

    def _compute(self, kwargs, tenors, probabilities, repeat):
        _, _, labels = _rolling_active(kwargs, tenors, probabilities, repeat)
        return labels

    def call(self, _number_of_rows_per_batch: int, tenors=None, probabilities=None, repeat=1, **kwargs):
        kwargs["_number_of_rows_per_batch"] = _number_of_rows_per_batch
        return self._compute(kwargs, tenors, probabilities, repeat)

    def recall(self, input_array=None, tenors=None, probabilities=None, repeat=1, **kwargs):
        return self._compute(kwargs, tenors, probabilities, repeat)


class RollingMaturityDate:
    def __init__(self) -> None:
        self._generator_description = (
            "rolling-book maturity date = trade_date + tenor, read from the (rolling) "
            "trade_date and tenor columns of the partial table — so maturity is always exactly "
            "consistent with the active trade and strictly after the as-of date."
        )

    def _compute(self, kwargs, trade_date_column, tenor_column):
        table = kwargs.get("_table")
        n = int(kwargs["_number_of_rows_per_batch"])
        base = _base_dates(table, trade_date_column, n, kwargs["_date"])
        tenor = _column_values(table, tenor_column)
        days = (
            np.array([_tenor_to_days(t) for t in tenor], dtype="int64")
            if tenor is not None
            else np.full(n, 365, dtype="int64")
        )
        return base + days.astype("timedelta64[D]")

    def call(
        self,
        _number_of_rows_per_batch: int,
        trade_date_column: str = "trade_date",
        tenor_column: str = "tenor",
        **kwargs,
    ):
        kwargs["_number_of_rows_per_batch"] = _number_of_rows_per_batch
        return self._compute(kwargs, trade_date_column, tenor_column)

    def recall(
        self,
        input_array=None,
        trade_date_column: str = "trade_date",
        tenor_column: str = "tenor",
        **kwargs,
    ):
        return self._compute(kwargs, trade_date_column, tenor_column)


class RollingTradeId:
    def __init__(self) -> None:
        self._generator_description = (
            "rolling-book trade id: a stable-while-live, new-on-roll integer reference derived "
            "from (batch, slot, booking date). A persisting trade keeps its id across dates; a "
            "rolled slot gets a brand-new id. Reads the (rolling) trade_date column."
        )

    def _compute(self, kwargs, trade_date_column, repeat):
        n = int(kwargs["_number_of_rows_per_batch"])
        positions, repeat = _rolling_positions(n, repeat)
        batch = int(kwargs.get("_batch_number", 1))
        book = _column_values(kwargs.get("_table"), trade_date_column)
        if book is None:  # trade_date absent (mis-ordered) -> fall back to the as-of date
            book = np.full(n, np.datetime64(kwargs["_date"], "D"))
        day_offset = (np.asarray(book, dtype="datetime64[D]") - _ROLL_ID_EPOCH).astype("int64")
        # position index recurs within each scenario block (repeat>1) so a trade keeps
        # one id across its scenario rows; unique per (batch, position) otherwise.
        pos = _tile_positions(np.arange(positions, dtype="int64"), repeat, n)
        slot_global = batch * positions + pos
        return slot_global * _ROLL_ID_DAY_RADIX + day_offset

    def call(self, _number_of_rows_per_batch: int, trade_date_column: str = "trade_date", repeat=1, **kwargs):
        kwargs["_number_of_rows_per_batch"] = _number_of_rows_per_batch
        return self._compute(kwargs, trade_date_column, repeat)

    def recall(self, input_array=None, trade_date_column: str = "trade_date", repeat=1, **kwargs):
        return self._compute(kwargs, trade_date_column, repeat)


class RollingScenario:
    def __init__(self) -> None:
        self._generator_description = (
            "rolling-book scenario dimension: cycles a fixed set of scenario labels within each "
            "position block (block layout, aligned with the rolling schedule's repeat), so a "
            "composite key (rolling_trade_id, scenario) is unique while the trade id recurs "
            "across scenarios. Set list_of_values=[...] to name the scenarios, or count=N."
        )

    def _compute(self, kwargs, list_of_values, count):
        n = int(kwargs["_number_of_rows_per_batch"])
        labels = list(list_of_values) if list_of_values else [f"scenario_{i}" for i in range(max(1, int(count)))]
        m = len(labels)
        # row r's scenario == r % m; the schedule generators tile each position m times
        # (block layout), so each position spans exactly the m distinct scenario labels.
        return np.asarray(labels, dtype=object)[np.arange(n) % m]

    def call(self, _number_of_rows_per_batch: int, list_of_values=None, count=1, **kwargs):
        kwargs["_number_of_rows_per_batch"] = _number_of_rows_per_batch
        return self._compute(kwargs, list_of_values, count)

    def recall(self, input_array=None, list_of_values=None, count=1, **kwargs):
        return self._compute(kwargs, list_of_values, count)


# ── Counterparty identifiers (coherent with the counterparty name) ────────────


def _ticker_for(name: str) -> str:
    """A deterministic synthetic ticker: the first 4 alphanumerics of the name,
    uppercased (not a real exchange ticker; stable per name)."""
    letters = "".join(ch for ch in str(name).upper() if ch.isalnum())
    return letters[:4] or "XXXX"


def _lei_check_digits(body18: str) -> str:
    """ISO 7064 MOD 97-10 check digits for a 20-char LEI's 18-char body."""
    numeric = "".join(str(int(ch, 36)) for ch in (body18 + "00"))
    return f"{98 - int(numeric) % 97:02d}"


def _lei_for(name: str) -> str:
    """A deterministic synthetic LEI: 18 hex chars derived from the name + 2 valid
    ISO 7064 check digits (passes an LEI checksum; not a registered LEI)."""
    body = hashlib.sha256(str(name).encode()).hexdigest().upper()[:18]
    return body + _lei_check_digits(body)


class CounterpartyTicker:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a (synthetic, deterministic) ticker coherent with the row's counterparty "
            "(reads counterparty_column). Random uppercase codes if that column is absent."
        )

    def call(
        self, _number_of_rows_per_batch: int, counterparty_column: str = "counterparty", **kwargs
    ):
        parent = _column_values(kwargs.get("_table"), counterparty_column)
        if parent is None:
            letters = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            idx = _rng_from(kwargs).integers(0, 26, size=(_number_of_rows_per_batch, 4))
            return np.array(["".join(letters[row]) for row in idx], dtype=object)
        cache: dict[str, str] = {}
        return np.array([cache.setdefault(str(c), _ticker_for(c)) for c in parent], dtype=object)


class CounterpartyLEI:
    def __init__(self) -> None:
        self._generator_description = (
            "generates a (synthetic, deterministic) 20-char LEI with a valid ISO 7064 check "
            "digit, coherent with the row's counterparty (reads counterparty_column)."
        )

    def call(
        self, _number_of_rows_per_batch: int, counterparty_column: str = "counterparty", **kwargs
    ):
        parent = _column_values(kwargs.get("_table"), counterparty_column)
        if parent is None:
            rng = _rng_from(kwargs)
            alnum = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            idx = rng.integers(0, len(alnum), size=(_number_of_rows_per_batch, 18))
            bodies = ["".join(alnum[row]) for row in idx]
            return np.array([b + _lei_check_digits(b) for b in bodies], dtype=object)
        cache: dict[str, str] = {}
        return np.array([cache.setdefault(str(c), _lei_for(c)) for c in parent], dtype=object)


# ── Underlying / reference entity (coherent with asset class) ─────────────────


class Underlying:
    def __init__(self) -> None:
        self._generator_description = (
            "generates the underlying / reference entity coherent with the row's asset class: "
            "a real issuer name for Equity / Credit rows, 'N/A' otherwise (reads "
            "asset_class_column). Flat draw over issuer names if that column is absent."
        )

    def call(
        self,
        _number_of_rows_per_batch: int,
        asset_class_column: str = "asset_class",
        **kwargs,
    ):
        rng = _rng_from(kwargs)
        parent = _column_values(kwargs.get("_table"), asset_class_column)
        if parent is None:
            return _sample(fd.COUNTERPARTIES, _number_of_rows_per_batch, kwargs)
        single_name = {"Equity", "Credit"}
        out = np.empty(_number_of_rows_per_batch, dtype=object)
        for value in np.unique(parent):
            mask = parent == value
            if str(value) in single_name:
                out[mask] = rng.choice(fd.COUNTERPARTIES, size=int(mask.sum()))
            else:
                out[mask] = "N/A"
        return out
