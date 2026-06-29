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

# Semantics whose generator is made coherent by reading a parent column.
_HIERARCHY_CHILDREN = ("instrument_type", "risk_factor_group")

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


def _is_string_type(col_type: str | None) -> bool:
    """Finance semantics apply to string-typed columns only."""
    return bool(col_type) and "string" in col_type.lower()


def _match_name(name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in _COMPILED:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def heuristic_inferrer(table_name, columns, samples=None):
    """Infer column semantics from column names (and types) alone.

    Dependency-free and deterministic. ``samples`` is accepted for interface
    parity with richer inferrers but unused here.
    """
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        col_type = col.get("type")
        if not _is_string_type(col_type):
            continue
        semantic, confidence = _match_name(str(name))
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
    """Point instrument_type / risk_factor_group records at the table's
    asset_class column (in place), so the generators read it for coherence."""
    asset_col = next((n for n, rec in result.items() if rec["semantic"] == "asset_class"), None)
    if asset_col is not None:
        for rec in result.values():
            if rec["semantic"] in _HIERARCHY_CHILDREN:
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


def values_inferrer(table_name, columns, samples=None):
    """Infer semantics from real sample values, falling back to the name
    heuristic for columns with no sample evidence (so one pass covers everything).

    ``samples`` is ``{col_name: [values]}`` (as normalised by ``Configs``).
    """
    samples = samples or {}
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        semantic, confidence, by = None, 0.0, "values"

        if name in samples:
            semantic, confidence = _classify_values(samples[name])
        if semantic is None and _is_string_type(col.get("type")):
            semantic, confidence = _match_name(str(name))
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

    return _link_hierarchy_parents(result)


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


def build_generator(semantic: str, parent: str | None = None) -> dict:
    """Build the ``column_generator`` dict for a recognised semantic.

    Introspects the target generator's ``call`` defaults (same as ``Columns``
    does for the built-in generators) and wires the parent column into the
    coherence arg when applicable. Raises ``KeyError`` for an unknown semantic.
    """
    module_name, class_name, base_args, parent_arg = SEMANTIC_GENERATORS[semantic]
    from datagenic.iokit import func

    gen_cls = getattr(importlib.import_module(module_name), class_name)
    args = func.get_default_args(gen_cls().call) | dict(base_args)
    if parent_arg and parent:
        args[parent_arg] = parent
    return {"udf_module": module_name, "udf_name": class_name, "args": args}
