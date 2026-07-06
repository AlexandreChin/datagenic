"""Semantic inference for scaffolding: guess each column's domain meaning so the
scaffolder can wire a realistic generator automatically, without the user
hand-tagging the IR.

This runs at scaffold time only (``Configs``), so it is *not* subject to the
byte-identical generation invariant — the inference produces a config that the
user then reviews and may hand-edit; generation from the frozen config stays
deterministic regardless.

Inferrer contract::

    fn(table_name: str, columns: list[dict], samples=None) -> dict[col_name, record]

where ``columns`` is a list of ``{"name", "type"}`` and each record is::

    {"semantic": str, "confidence": float, "parent": str | None,
     "alternatives": list[str], "by": str}

``resolve_inferrer`` maps a CLI/string spec to a callable. Built-ins: ``heuristic``
(name-based, dependency-free), ``values`` (sample-value based, with a name-heuristic
fallback) and ``agent`` (opt-in LLM via ``datagenic.semantics_agent``; needs the
anthropic SDK, lazy-imported only when invoked).
"""

import importlib
import re

from datagenic.market_finance._data import (  # pure data, no third-party imports
    ARRAY_QUANTITY_PRIORS,
    CONDITIONAL_ARRAY_QUANTITY_PRIORS,
    CONDITIONAL_QUANTITY_PRIORS,
    QUANTITY_PRIORS,
)

# semantic -> (udf_module, udf_class, base_args, parent_arg_name | None)
SEMANTIC_GENERATORS: dict[str, tuple[str, str, dict, str | None]] = {
    "asset_class": ("datagenic.market_finance.generators", "FinancialAssetClass", {}, None),
    "instrument_type": (
        "datagenic.market_finance.generators",
        "FinancialInstrumentType",
        {},
        "asset_class_column",
    ),
    "risk_factor_group": (
        "datagenic.market_finance.generators",
        "FinancialRiskFactorGroup",
        {},
        "asset_class_column",
    ),
    "currency": ("datagenic.market_finance.generators", "Currency", {}, None),
    "currency_pair": ("datagenic.market_finance.generators", "CurrencyPair", {}, None),
    "tenor": ("datagenic.market_finance.generators", "Tenor", {}, None),
    "credit_rating": ("datagenic.market_finance.generators", "CreditRating", {}, None),
    "gics_sector": ("datagenic.market_finance.generators", "GICSSector", {}, None),
    "isin": ("datagenic.market_finance.generators", "ISIN", {}, None),
    "cusip": ("datagenic.market_finance.generators", "CUSIP", {}, None),
    "sedol": ("datagenic.market_finance.generators", "SEDOL", {}, None),
    "instrument_name": ("datagenic.market_finance.generators", "InstrumentName", {}, None),
    "risk_factor_name": ("datagenic.market_finance.generators", "RiskFactorName", {}, None),
    "counterparty": ("datagenic.market_finance.generators", "Counterparty", {}, None),
    "counterparty_sector": (
        "datagenic.market_finance.generators",
        "CounterpartySector",
        {},
        "counterparty_column",
    ),
    "counterparty_country": (
        "datagenic.market_finance.generators",
        "CounterpartyCountry",
        {},
        "counterparty_column",
    ),
    "counterparty_ticker": (
        "datagenic.market_finance.generators",
        "CounterpartyTicker",
        {},
        "counterparty_column",
    ),
    "counterparty_lei": (
        "datagenic.market_finance.generators",
        "CounterpartyLEI",
        {},
        "counterparty_column",
    ),
    "trading_book": ("datagenic.market_finance.generators", "TradingBook", {}, None),
    "desk": ("datagenic.market_finance.generators", "Desk", {}, None),
    # trade attributes (fixed-vocabulary categoricals)
    "side": ("datagenic.market_finance.generators", "Side", {}, None),
    "option_type": ("datagenic.market_finance.generators", "OptionType", {}, None),
    "exercise_style": ("datagenic.market_finance.generators", "ExerciseStyle", {}, None),
    "payment_frequency": ("datagenic.market_finance.generators", "PaymentFrequency", {}, None),
    "day_count_convention": ("datagenic.market_finance.generators", "DayCountConvention", {}, None),
    "trade_status": ("datagenic.market_finance.generators", "TradeStatus", {}, None),
    "exchange": ("datagenic.market_finance.generators", "Exchange", {}, None),
    "clearing_house": (
        "datagenic.market_finance.generators",
        "ClearingHouse",
        {},
        "asset_class_column",
    ),
    "underlying": ("datagenic.market_finance.generators", "Underlying", {}, "asset_class_column"),
    # coherent dates
    "trade_date": ("datagenic.market_finance.generators", "TradeDate", {}, None),
    "maturity_date": (
        "datagenic.market_finance.generators",
        "MaturityDate",
        {},
        "trade_date_column",
    ),
    "settlement_date": (
        "datagenic.market_finance.generators",
        "SettlementDate",
        {},
        "trade_date_column",
    ),
}

# Numeric quantity semantics: a distribution-aware float generator wired with a
# book-agnostic prior (real-unit median/spread + family) so magnitudes are
# plausible out of the box. Flat (no parent) in this phase.
for _semantic, _prior in QUANTITY_PRIORS.items():
    SEMANTIC_GENERATORS[_semantic] = (
        "datagenic.generators",
        _prior["generator"],
        {"mu": _prior["mu"], "sigma": _prior["sigma"], "distribution": _prior["distribution"]},
        None,
    )

# Conditional numeric semantics: a value whose distribution depends on a parent
# category (e.g. notional scaled by asset_class). Maps to ConditionalFloats with
# the parent column wired into `by_column`; the case/default priors are the base
# args. Falls back to `default` at runtime when the parent column is absent.
for _semantic, _prior in CONDITIONAL_QUANTITY_PRIORS.items():
    SEMANTIC_GENERATORS[_semantic] = (
        "datagenic.generators",
        "ConditionalFloats",
        {"cases": _prior["cases"], "default": _prior["default"]},
        "by_column",
    )

# Array-typed quantity semantics: a distribution-aware array generator wired with
# a book-agnostic prior (family + real-unit params + array length + path). Flat
# (no parent) in this phase. Applied only to array(float) columns (see the gate).
for _semantic, _prior in ARRAY_QUANTITY_PRIORS.items():
    SEMANTIC_GENERATORS[_semantic] = (
        "datagenic.generators",
        _prior["generator"],
        {k: v for k, v in _prior.items() if k != "generator"},
        None,
    )

# Conditional array-quantity semantics: a per-row array whose distribution depends
# on a parent category (e.g. a PnL series whose vol depends on asset_class). Maps to
# ConditionalArrayFloats with the parent wired into `by_column`; size_of_arrays/path
# and the case/default priors are the base args.
for _semantic, _prior in CONDITIONAL_ARRAY_QUANTITY_PRIORS.items():
    SEMANTIC_GENERATORS[_semantic] = (
        "datagenic.generators",
        "ConditionalArrayFloats",
        {k: v for k, v in _prior.items() if k not in ("parent_semantic", "by_column")},
        "by_column",
    )

# Categorical semantics made coherent by reading the asset_class column.
_HIERARCHY_CHILDREN = ("instrument_type", "risk_factor_group", "clearing_house", "underlying")

# Counterparty attributes that must stay coherent with the row's counterparty column.
_COUNTERPARTY_CHILDREN = (
    "counterparty_sector",
    "counterparty_country",
    "counterparty_ticker",
    "counterparty_lei",
)

# Dates derived from the row's trade_date (maturity = trade_date + tenor; settlement = +lag).
_TRADE_DATE_CHILDREN = ("maturity_date", "settlement_date")

# Conditional (scalar or array) numeric semantics that condition on the asset_class
# column, linked to it the same way the categorical hierarchy children are.
_ASSET_CLASS_CONDITIONAL = tuple(
    sem
    for sem, prior in {**CONDITIONAL_QUANTITY_PRIORS, **CONDITIONAL_ARRAY_QUANTITY_PRIORS}.items()
    if prior.get("parent_semantic") == "asset_class"
)

# Ordered (specific first) name patterns -> (semantic, confidence). The first
# matching pattern wins, so e.g. currency_pair is tested before currency.
_PATTERNS: list[tuple[str, str, float]] = [
    (r"isin", "isin", 0.97),
    (r"cusip", "cusip", 0.97),
    (r"sedol", "sedol", 0.97),
    (r"(currency|ccy|fx)[_ ]?pair", "currency_pair", 0.9),
    (r"risk[_ ]?factor[_ ]?(group|class)|rf[_ ]?group|riskfactorgroup", "risk_factor_group", 0.9),
    (r"risk[_ ]?factor[_ ]?name|rf[_ ]?name", "risk_factor_name", 0.85),
    (r"instrument[_ ]?type|instr[_ ]?type|product[_ ]?type", "instrument_type", 0.9),
    (r"instrument[_ ]?name|security[_ ]?name|instr[_ ]?name", "instrument_name", 0.85),
    (r"asset[_ ]?class|assetclass|asset[_ ]?cls", "asset_class", 0.9),
    (r"credit[_ ]?rating|^rating$|\brating\b", "credit_rating", 0.8),
    # counterparty attributes — tested before both the generic `sector` pattern and the
    # plain `counterparty` pattern (which would otherwise swallow "counterparty_sector").
    (r"(counterparty|cpty|ctp)[_ ]?sector", "counterparty_sector", 0.9),
    (r"(counterparty|cpty|ctp)[_ ]?(country|domicile|nationality)", "counterparty_country", 0.9),
    (r"(counterparty|cpty|ctp)[_ ]?ticker|^ticker$|\bticker\b|symbol", "counterparty_ticker", 0.85),
    (r"\blei\b|legal[_ ]?entity[_ ]?id", "counterparty_lei", 0.9),
    (r"gics|^sector$|\bsector\b|industry", "gics_sector", 0.75),
    (r"counterparty|^ctp$|^cpty$|legal[_ ]?entity", "counterparty", 0.8),
    (r"trading[_ ]?book|^book$|portfolio", "trading_book", 0.7),
    (r"^desk$|\bdesk\b", "desk", 0.8),
    # trade attributes
    (r"^side$|buy[_ ]?sell|\bdirection\b|long[_ ]?short", "side", 0.85),
    (r"option[_ ]?type|call[_ ]?put|put[_ ]?call", "option_type", 0.85),
    (r"exercise[_ ]?(style|type)", "exercise_style", 0.85),
    (r"payment[_ ]?freq|coupon[_ ]?freq|pay[_ ]?freq|\bfrequency\b", "payment_frequency", 0.8),
    (r"day[_ ]?count|daycount|\bdcc\b", "day_count_convention", 0.85),
    (r"trade[_ ]?status|^status$|lifecycle", "trade_status", 0.8),
    (r"clearing[_ ]?house|\bccp\b|clearer|clearing[_ ]?broker", "clearing_house", 0.85),
    (r"exchange|\bvenue\b|\bmic\b|listing", "exchange", 0.75),
    (r"underlying|underlier|reference[_ ]?(entity|name)", "underlying", 0.85),
    (r"tenor|maturity|^term$", "tenor", 0.8),
    (r"^ccy$|^cur$|currency|ccy[_ ]?code", "currency", 0.85),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), sem, conf) for p, sem, conf in _PATTERNS]

# Numeric quantity name patterns (specific first) -> (semantic, confidence).
# Gated to float-typed columns, so ids/counts (ints) and categoricals (strings)
# are never matched. fx_rate is tested before rate; pv before price.
_NUMERIC_PATTERNS: list[tuple[str, str, float]] = [
    (r"notional|principal|face[_ ]?(value|amount|amt)?", "notional", 0.8),
    (r"fx[_ ]?rate|spot[_ ]?rate|exchange[_ ]?rate", "fx_rate", 0.85),
    (r"spread", "spread", 0.8),
    (
        r"\bpv\b|\bnpv\b|\bmtm\b|present[_ ]?value|mark[_ ]?to[_ ]?market|market[_ ]?value",
        "pv",
        0.75,
    ),
    (r"price|^px$|\bpx\b", "price", 0.8),
    (r"quantity|^qty$|\bsize\b|volume|shares|\bunits\b", "quantity", 0.7),
    (r"\brate\b|yield|coupon", "rate", 0.7),
    (r"\bpnl\b|profit[_ ]?and[_ ]?loss|p_and_l|p_n_l", "pnl", 0.75),
    # option value drivers + risk measures + Greeks
    (r"volatility|implied[_ ]?vol|\bvol\b|\biv\b", "volatility", 0.8),
    (r"\bstrike\b|strike[_ ]?price", "strike", 0.85),
    (r"\bspot\b|spot[_ ]?price", "spot", 0.8),
    (r"expected[_ ]?shortfall|\bcvar\b|\bes\b", "expected_shortfall", 0.85),
    (r"\bvar\b|value[_ ]?at[_ ]?risk", "var", 0.8),
    (r"\bdelta\b", "delta", 0.9),
    (r"\bgamma\b", "gamma", 0.9),
    (r"\bvega\b", "vega", 0.9),
    (r"\btheta\b", "theta", 0.9),
    (r"\brho\b", "rho", 0.9),
]
_NUMERIC_COMPILED = [
    (re.compile(p, re.IGNORECASE), sem, conf) for p, sem, conf in _NUMERIC_PATTERNS
]

# Array-quantity name patterns (specific first) -> (semantic, confidence). Gated
# to array(float) columns. The same name (e.g. "pnl") maps to a *different*
# semantic here than in the scalar table — the column's type decides which set is
# consulted, so scalar pnl (signed float) and array pnl_series never collide.
_ARRAY_NUMERIC_PATTERNS: list[tuple[str, str, float]] = [
    (r"\bpnl\b|pnl[_ ]?(series|vector|path|ts)|profit[_ ]?and[_ ]?loss", "pnl_series", 0.8),
    (r"return|\brtns?\b", "returns", 0.7),
    (r"curve|term[_ ]?structure|yield[_ ]?curve", "curve", 0.75),
    (r"sensitivit|\bdeltas\b|bucket|risk[_ ]?weights?|\bladder\b", "sensitivities", 0.8),
]
_ARRAY_NUMERIC_COMPILED = [
    (re.compile(p, re.IGNORECASE), sem, conf) for p, sem, conf in _ARRAY_NUMERIC_PATTERNS
]

# Date name patterns (specific first) -> (semantic, confidence). Gated to date/datetime
# columns. maturity/settlement derive from trade_date (+ tenor / + lag).
_DATE_PATTERNS: list[tuple[str, str, float]] = [
    (r"trade[_ ]?date|booking[_ ]?date|execution[_ ]?date|deal[_ ]?date", "trade_date", 0.85),
    (
        r"maturity[_ ]?date|expiry[_ ]?date|expiration[_ ]?date|^maturity$|^expiry$|end[_ ]?date",
        "maturity_date",
        0.85,
    ),
    (r"settlement[_ ]?date|value[_ ]?date|settle[_ ]?date|^settlement$", "settlement_date", 0.85),
]
_DATE_COMPILED = [(re.compile(p, re.IGNORECASE), sem, conf) for p, sem, conf in _DATE_PATTERNS]


def _is_string_type(col_type: str | None) -> bool:
    """Finance categorical/identifier semantics apply to string-typed columns only."""
    return bool(col_type) and "string" in col_type.lower()


def _is_float_array_type(col_type: str | None) -> bool:
    """Array-quantity semantics apply to array(float) columns."""
    return bool(col_type) and "array(float)" in col_type.lower()


def _is_float_type(col_type: str | None) -> bool:
    """Scalar numeric semantics apply to *scalar* float columns only — not ints
    (keys/counts) and not array(float) (which get the array-quantity semantics)."""
    t = col_type.lower() if col_type else ""
    return "float" in t and "array" not in t


def _is_date_type(col_type: str | None) -> bool:
    """Date semantics apply to date / datetime columns."""
    return bool(col_type) and "date" in col_type.lower()


def _match_from(compiled, name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in compiled:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def _match_name(name: str) -> tuple[str | None, float]:
    return _match_from(_COMPILED, name)


def _match_numeric_name(name: str) -> tuple[str | None, float]:
    return _match_from(_NUMERIC_COMPILED, name)


def _match_array_numeric_name(name: str) -> tuple[str | None, float]:
    return _match_from(_ARRAY_NUMERIC_COMPILED, name)


def _match_by_type(name: str, col_type: str | None) -> tuple[str | None, float]:
    """Name-based match dispatched on column type: string -> categorical/id,
    array(float) -> array quantity, scalar float -> scalar quantity, date/datetime ->
    date semantics. The type (never the name) decides which pattern set is consulted,
    so a name like "pnl" resolves to the scalar or the array semantic by whether the
    column is a float or an array(float). Other types are never matched."""
    if _is_string_type(col_type):
        return _match_from(_COMPILED, name)
    if _is_float_array_type(col_type):
        return _match_from(_ARRAY_NUMERIC_COMPILED, name)
    if _is_float_type(col_type):
        return _match_from(_NUMERIC_COMPILED, name)
    if _is_date_type(col_type):
        return _match_from(_DATE_COMPILED, name)
    return None, 0.0


def heuristic_inferrer(table_name, columns, samples=None):
    """Infer column semantics from column names (and types) alone.

    Dependency-free and deterministic. ``samples`` is accepted for interface
    parity with richer inferrers but unused here.
    """
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        semantic, confidence = _match_by_type(str(name), col.get("type"))
        if semantic is None:
            continue
        result[name] = {
            "semantic": semantic,
            "confidence": confidence,
            "parent": None,
            "alternatives": [],
            "by": "heuristic",
        }

    return _link_hierarchy_parents(result)


def _link_hierarchy_parents(result: dict) -> dict:
    """Point each coherent child record at the column it must read (in place), so the
    generator stays consistent with that parent. Covers the asset_class hierarchy
    (categorical children instrument_type / risk_factor_group + conditional numeric
    children like notional) and counterparty attributes (sector / country). A parent the
    inferrer already chose (e.g. the agent conditioning notional on instrument_type) is
    kept."""
    links = {
        "asset_class": _HIERARCHY_CHILDREN + _ASSET_CLASS_CONDITIONAL,
        "counterparty": _COUNTERPARTY_CHILDREN,
        "trade_date": _TRADE_DATE_CHILDREN,
    }
    for parent_semantic, children in links.items():
        parent_col = next(
            (n for n, rec in result.items() if rec["semantic"] == parent_semantic), None
        )
        if parent_col is None:
            continue
        for rec in result.values():
            if rec["semantic"] in children and not rec.get("parent"):
                rec["parent"] = parent_col
    return result


# ── Value-based inference (Phase 2) ──────────────────────────────────────────

# Minimum fraction of sample values that must match for a semantic to be assigned.
_VALUE_THRESHOLD = 0.9

# Lazily built so importing this module stays dependency-free (the vocab lives in
# market_finance._data, the checksum validators in market_finance.generators which import numpy).
_VALUE_CLASSIFIERS = None


def _build_value_classifiers():
    """Ordered (semantic, predicate) pairs, most specific first. Each predicate
    takes a single string value and returns True if it looks like that semantic."""
    import re as _re

    from datagenic.market_finance import _data as fd
    from datagenic.market_finance import generators as gf

    isin_re = _re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
    cusip_re = _re.compile(r"^[A-Z0-9]{8}[0-9]$")
    sedol_re = _re.compile(r"^[A-Z0-9]{6}[0-9]$")
    pair_re = _re.compile(r"^([A-Z]{3})/([A-Z]{3})$")
    currencies = set(fd.CURRENCIES)

    def _valid(value, regex, check_fn):
        return bool(regex.match(value)) and check_fn(value[:-1]) == value[-1]

    def _is_pair(value):
        m = pair_re.match(value)
        return bool(m) and m.group(1) in currencies and m.group(2) in currencies

    ratings = set(fd.SP_RATINGS) | set(fd.MOODYS_RATINGS)

    return [
        ("isin", lambda v: _valid(v, isin_re, gf._isin_check_digit)),
        ("cusip", lambda v: _valid(v, cusip_re, gf._cusip_check_digit)),
        ("sedol", lambda v: _valid(v, sedol_re, gf._sedol_check_digit)),
        ("currency_pair", _is_pair),
        ("currency", lambda v: v in currencies),
        ("tenor", lambda v: v in set(fd.TENORS)),
        ("asset_class", lambda v: v in set(fd.ASSET_CLASSES)),
        ("credit_rating", lambda v: v in ratings),
        ("gics_sector", lambda v: v in set(fd.GICS_SECTORS)),
        ("instrument_type", lambda v: v in set(fd.ALL_INSTRUMENT_TYPES)),
        ("risk_factor_group", lambda v: v in set(fd.ALL_RISK_FACTOR_GROUPS)),
        ("counterparty", lambda v: v in set(fd.COUNTERPARTIES)),
    ]


def _classify_values(values) -> tuple[str | None, float]:
    """Classify a column from its sample values. Returns the most specific
    semantic whose match fraction clears the threshold, with that fraction as the
    confidence; ``(None, 0.0)`` if nothing matches."""
    global _VALUE_CLASSIFIERS
    vals = [str(v) for v in values if v is not None and str(v) != ""]
    if not vals:
        return None, 0.0
    if _VALUE_CLASSIFIERS is None:
        _VALUE_CLASSIFIERS = _build_value_classifiers()

    best_semantic, best_fraction = None, 0.0
    for semantic, predicate in _VALUE_CLASSIFIERS:  # specific first; strict > keeps it on ties
        fraction = sum(predicate(v) for v in vals) / len(vals)
        if fraction >= _VALUE_THRESHOLD and fraction > best_fraction:
            best_semantic, best_fraction = semantic, fraction
    if best_semantic is None:
        return None, 0.0
    return best_semantic, round(min(0.99, best_fraction), 2)


# ── Numeric distribution fitting (Phase 4) ────────────────────────────────────

# Minimum per-group sample count before a conditional case is fitted (thinner
# groups fall back to the pooled default).
_MIN_GROUP_SAMPLES = 30
# mean/median ratio above which an all-positive column is treated as lognormal.
# For a lognormal this equals exp(sigma**2/2) (≈1.13 at sigma=0.5, ≈1.65 at 1.0);
# a symmetric distribution sits at ≈1.0.
_LOGNORMAL_SKEW = 1.2


def fit_distribution(values) -> dict | None:
    """Fit a ``{distribution, mu, sigma}`` spec to numeric sample values.

    Picks **lognormal** (``mu`` = median in real units, ``sigma`` = log-space
    spread) when the values are all-positive and right-skewed (mean pulled above
    the median), else **normal** (``mu`` = mean, ``sigma`` = std). The
    parameterisation matches ``generators._draw_floats`` exactly, so
    generate→profile→regenerate round-trips. ``None`` when nothing to fit.
    """
    import numpy as np

    x = np.asarray([v for v in values if v is not None], dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    median = float(np.median(x))
    right_skewed = median > 0 and float(x.mean()) / median >= _LOGNORMAL_SKEW
    if bool((x > 0).all()) and x.size >= 2 and right_skewed:
        logs = np.log(x)
        return {
            "distribution": "lognormal",
            "mu": round(float(np.exp(logs.mean())), 6),
            "sigma": round(float(logs.std()), 6),
        }
    return {
        "distribution": "normal",
        "mu": round(float(x.mean()), 6),
        "sigma": round(float(x.std()) if x.size > 1 else 0.0, 6),
    }


def _fit_conditional(child_values, parent_values) -> dict | None:
    """Fit ``{cases, default}`` for ``ConditionalFloats`` from row-aligned child /
    parent samples: a per-category distribution for each parent value with enough
    samples, plus a pooled ``default`` covering thin/unseen categories."""
    n = min(len(child_values), len(parent_values))
    groups: dict[str, list] = {}
    for child, parent in zip(child_values[:n], parent_values[:n], strict=True):
        if child is None or parent is None:
            continue
        groups.setdefault(str(parent), []).append(child)

    default = fit_distribution(child_values)
    if default is None:
        return None
    cases = {}
    for value, vals in groups.items():
        if len(vals) >= _MIN_GROUP_SAMPLES:
            spec = fit_distribution(vals)
            if spec is not None:
                cases[value] = spec
    return {"cases": cases, "default": default}


def _flatten_arrays(arrays) -> list:
    """All elements of a list-of-arrays sample, skipping null rows/elements."""
    return [v for arr in arrays if arr is not None for v in arr if v is not None]


def _array_length(arrays) -> int | None:
    """Representative (median) array length across the sample rows."""
    import statistics

    lengths = [len(arr) for arr in arrays if arr is not None]
    return round(statistics.median(lengths)) if lengths else None


def fit_array_distribution(arrays) -> dict | None:
    """Fit ``{distribution, mu, sigma, size_of_arrays}`` for an array-quantity
    column: the element family/params (flattening all rows) plus the array length.
    ``None`` when there is nothing to fit."""
    spec = fit_distribution(_flatten_arrays(arrays))
    if spec is None:
        return None
    size = _array_length(arrays)
    if size:
        spec = {**spec, "size_of_arrays": size}
    return spec


def _fit_conditional_array(child_arrays, parent_values) -> dict | None:
    """Fit ``{cases, default, size_of_arrays}`` for ``ConditionalArrayFloats`` from
    row-aligned array/parent samples: per-category element params (enough elements),
    pooled ``default``, and a shared array length."""
    n = min(len(child_arrays), len(parent_values))
    groups: dict[str, list] = {}
    for arr, parent in zip(child_arrays[:n], parent_values[:n], strict=True):
        if arr is None or parent is None:
            continue
        groups.setdefault(str(parent), []).append(arr)

    default = fit_distribution(_flatten_arrays(child_arrays))
    if default is None:
        return None
    cases = {}
    for value, arrs in groups.items():
        flat = _flatten_arrays(arrs)
        if len(flat) >= _MIN_GROUP_SAMPLES:
            spec = fit_distribution(flat)
            if spec is not None:
                cases[value] = spec
    fitted = {"cases": cases, "default": default}
    size = _array_length(child_arrays)
    if size:
        fitted["size_of_arrays"] = size
    return fitted


def _fit_numeric(result: dict, samples: dict) -> dict:
    """Attach fitted distribution args (in place) to numeric-semantic records
    using the sample values — conditional (per parent category) for conditional
    semantics, flat otherwise. Overrides the curated priors at ``build_generator``."""
    for col, rec in result.items():
        semantic = rec["semantic"]
        values = samples.get(col)
        if not values:
            continue
        if semantic in CONDITIONAL_QUANTITY_PRIORS:
            parent_values = samples.get(rec.get("parent"))
            fitted = _fit_conditional(values, parent_values) if parent_values else None
            if fitted is None:  # no usable parent -> still fit a data-driven default
                spec = fit_distribution(values)
                fitted = {"default": spec} if spec else None
            if fitted:
                rec["args"] = fitted
        elif semantic in CONDITIONAL_ARRAY_QUANTITY_PRIORS:
            parent_values = samples.get(rec.get("parent"))
            fitted = _fit_conditional_array(values, parent_values) if parent_values else None
            if fitted is None:  # no usable parent -> pooled element fit as default
                spec = fit_distribution(_flatten_arrays(values))
                fitted = {"default": spec} if spec else None
            if fitted:
                rec["args"] = fitted
        elif semantic in QUANTITY_PRIORS:
            spec = fit_distribution(values)
            if spec:
                rec["args"] = spec
        elif semantic in ARRAY_QUANTITY_PRIORS:
            fitted = fit_array_distribution(values)
            if fitted:
                rec["args"] = fitted
    return result


def values_inferrer(table_name, columns, samples=None):
    """Infer semantics from real sample values, falling back to the name
    heuristic for columns with no sample evidence (so one pass covers everything).
    Numeric quantities additionally have their distribution **fitted** from the
    samples (family + params, per parent category when conditional).

    ``samples`` is ``{col_name: [values]}`` (as normalised by ``Configs``).
    """
    samples = samples or {}
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        semantic, confidence, by = None, 0.0, "values"

        if name in samples:
            semantic, confidence = _classify_values(samples[name])
        if semantic is None:
            semantic, confidence = _match_by_type(str(name), col.get("type"))
            by = "heuristic"
        if semantic is None:
            continue

        result[name] = {
            "semantic": semantic,
            "confidence": confidence,
            "parent": None,
            "alternatives": [],
            "by": by,
        }

    _link_hierarchy_parents(result)  # sets parents so conditional fits can group
    return _fit_numeric(result, samples)


def _agent_inferrer(table_name, columns, samples=None):
    """Lazy delegator to the LLM-backed inferrer. Imported on first call only, so
    neither this module nor ``scaffolding`` pulls in the anthropic SDK at import."""
    from datagenic.semantics_agent import agent_inferrer

    return agent_inferrer(table_name, columns, samples=samples)


_BUILTINS = {
    "heuristic": heuristic_inferrer,
    "values": values_inferrer,
    "agent": _agent_inferrer,
}


def resolve_inferrer(spec):
    """Resolve an inferrer spec to a callable.

    ``spec`` may be a callable (returned as-is), ``None`` (returns ``None`` = no
    inference), a built-in name (``heuristic`` / ``values`` / ``agent``), or a
    ``module:function`` path (resolved like the scaffolder's ``--adaptor``).
    """
    if spec is None or callable(spec):
        return spec
    if spec in _BUILTINS:
        return _BUILTINS[spec]
    if ":" in spec:
        module_name, _, func_name = spec.partition(":")
        return getattr(importlib.import_module(module_name), func_name)
    raise ValueError(
        f"unknown semantic inferrer {spec!r}; expected a built-in "
        f"({', '.join(_BUILTINS)}) or a 'module:function' path"
    )


def build_generator(
    semantic: str, parent: str | None = None, overrides: dict | None = None
) -> dict:
    """Build the ``column_generator`` dict for a recognised semantic.

    Introspects the target generator's ``call`` defaults (same as ``Columns``
    does for the built-in generators) and wires the parent column into the
    coherence arg when applicable. ``overrides`` (e.g. distribution params fitted
    from a sample) take precedence over the curated priors. Raises ``KeyError``
    for an unknown semantic.
    """
    module_name, class_name, base_args, parent_arg = SEMANTIC_GENERATORS[semantic]
    from datagenic.iokit import func

    gen_cls = getattr(importlib.import_module(module_name), class_name)
    args = func.get_default_args(gen_cls().call) | dict(base_args)
    if parent_arg and parent:
        args[parent_arg] = parent
    if overrides:
        args.update(overrides)
    return {"udf_module": module_name, "udf_name": class_name, "args": args}
