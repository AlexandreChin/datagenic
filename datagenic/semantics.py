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

from datagenic._finance_data import (  # pure data, no third-party imports
    ARRAY_QUANTITY_PRIORS,
    CONDITIONAL_ARRAY_QUANTITY_PRIORS,
    CONDITIONAL_QUANTITY_PRIORS,
    QUANTITY_PRIORS,
)

# semantic -> (udf_module, udf_class, base_args, parent_arg_name | None)
SEMANTIC_GENERATORS: dict[str, tuple[str, str, dict, str | None]] = {
    "asset_class": ("datagenic.generators_finance", "FinancialAssetClass", {}, None),
    "instrument_type": (
        "datagenic.generators_finance",
        "FinancialInstrumentType",
        {},
        "asset_class_column",
    ),
    "risk_factor_group": (
        "datagenic.generators_finance",
        "FinancialRiskFactorGroup",
        {},
        "asset_class_column",
    ),
    "currency": ("datagenic.generators_finance", "Currency", {}, None),
    "currency_pair": ("datagenic.generators_finance", "CurrencyPair", {}, None),
    "tenor": ("datagenic.generators_finance", "Tenor", {}, None),
    "credit_rating": ("datagenic.generators_finance", "CreditRating", {}, None),
    "gics_sector": ("datagenic.generators_finance", "GICSSector", {}, None),
    "isin": ("datagenic.generators_finance", "ISIN", {}, None),
    "cusip": ("datagenic.generators_finance", "CUSIP", {}, None),
    "sedol": ("datagenic.generators_finance", "SEDOL", {}, None),
    "instrument_name": ("datagenic.generators_finance", "InstrumentName", {}, None),
    "risk_factor_name": ("datagenic.generators_finance", "RiskFactorName", {}, None),
    "counterparty": ("datagenic.generators_finance", "Counterparty", {}, None),
    "trading_book": ("datagenic.generators_finance", "TradingBook", {}, None),
    "desk": ("datagenic.generators_finance", "Desk", {}, None),
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

# Semantics whose generator is made coherent by reading a parent column.
_HIERARCHY_CHILDREN = ("instrument_type", "risk_factor_group")

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
    (r"gics|^sector$|\bsector\b|industry", "gics_sector", 0.75),
    (r"counterparty|^ctp$|^cpty$|legal[_ ]?entity", "counterparty", 0.8),
    (r"trading[_ ]?book|^book$|portfolio", "trading_book", 0.7),
    (r"^desk$|\bdesk\b", "desk", 0.8),
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
]
_ARRAY_NUMERIC_COMPILED = [
    (re.compile(p, re.IGNORECASE), sem, conf) for p, sem, conf in _ARRAY_NUMERIC_PATTERNS
]


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


def _match_name(name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in _COMPILED:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def _match_numeric_name(name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in _NUMERIC_COMPILED:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def _match_array_numeric_name(name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in _ARRAY_NUMERIC_COMPILED:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def _match_by_type(name: str, col_type: str | None) -> tuple[str | None, float]:
    """Name-based match dispatched on column type: string -> categorical/id,
    array(float) -> array quantity, scalar float -> scalar quantity. The type
    (never the name) decides which pattern set is consulted, so a name like "pnl"
    resolves to the scalar or the array semantic by whether the column is a float
    or an array(float). Other types are never matched."""
    if _is_string_type(col_type):
        return _match_name(name)
    if _is_float_array_type(col_type):
        return _match_array_numeric_name(name)
    if _is_float_type(col_type):
        return _match_numeric_name(name)
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
    """Point asset-class-dependent records at the table's asset_class column (in
    place), so the generators read it for coherence. Covers both the categorical
    hierarchy children (instrument_type / risk_factor_group) and conditional
    numeric children (e.g. notional)."""
    asset_col = next((n for n, rec in result.items() if rec["semantic"] == "asset_class"), None)
    if asset_col is not None:
        for rec in result.values():
            # don't overwrite a parent the inferrer already chose (e.g. the agent
            # conditioning a notional on instrument_type instead of asset_class).
            if rec["semantic"] in _HIERARCHY_CHILDREN + _ASSET_CLASS_CONDITIONAL and not rec.get(
                "parent"
            ):
                rec["parent"] = asset_col
    return result


# ── Value-based inference (Phase 2) ──────────────────────────────────────────

# Minimum fraction of sample values that must match for a semantic to be assigned.
_VALUE_THRESHOLD = 0.9

# Lazily built so importing this module stays dependency-free (the vocab lives in
# _finance_data, the checksum validators in generators_finance which imports numpy).
_VALUE_CLASSIFIERS = None


def _build_value_classifiers():
    """Ordered (semantic, predicate) pairs, most specific first. Each predicate
    takes a single string value and returns True if it looks like that semantic."""
    import re as _re

    from datagenic import _finance_data as fd
    from datagenic import generators_finance as gf

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
