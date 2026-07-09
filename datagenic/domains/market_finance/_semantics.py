"""Registration of the ``market_finance`` domain pack into the semantic registry.

Everything financial-markets-specific that scaffold-time inference needs lives here
(moved out of the domain-agnostic ``datagenic.semantics``): the semantic → generator
map, the type-gated name patterns, the parent-coherence links, and the value-based
classifiers. ``register(registry)`` is the single hook the core discovery calls.

Module-level content is only literals + function defs, so importing this stays light;
the pure-data priors/vocab (``_data``) are imported inside ``register`` / the classifier
factory, and the numpy-backed generators only when a generator is actually built.
"""

_GEN = "datagenic.domains.market_finance.generators"

# Categorical / identifier / date / special generators: semantic -> (class, parent_arg).
_CATEGORICAL_GENERATORS: dict[str, tuple[str, str | None]] = {
    "asset_class": ("FinancialAssetClass", None),
    "instrument_type": ("FinancialInstrumentType", "asset_class_column"),
    "risk_factor_group": ("FinancialRiskFactorGroup", "asset_class_column"),
    "currency": ("Currency", None),
    "currency_pair": ("CurrencyPair", None),
    "tenor": ("Tenor", None),
    "credit_rating": ("CreditRating", None),
    "gics_sector": ("GICSSector", None),
    "isin": ("ISIN", None),
    "cusip": ("CUSIP", None),
    "sedol": ("SEDOL", None),
    "eligible_collateral": ("EligibleCollateral", None),
    "instrument_name": ("InstrumentName", None),
    "risk_factor_name": ("RiskFactorName", None),
    # tenor-bucketed sensitivities: a ladder array scaled by asset_class, shaped by
    # instrument_type, zeroed beyond the remaining maturity (reads maturity_date).
    "sensi_type": ("SensiType", None),
    "tenor_grid": ("TenorGrid", None),
    "tenor_sensitivities": ("TenorSensitivities", "maturity_date_column"),
    "counterparty": ("Counterparty", None),
    "counterparty_sector": ("CounterpartySector", "counterparty_column"),
    "counterparty_country": ("CounterpartyCountry", "counterparty_column"),
    "counterparty_ticker": ("CounterpartyTicker", "counterparty_column"),
    "counterparty_lei": ("CounterpartyLEI", "counterparty_column"),
    "trading_book": ("TradingBook", None),
    "desk": ("Desk", None),
    # trade attributes (fixed-vocabulary categoricals)
    "side": ("Side", None),
    "option_type": ("OptionType", None),
    "exercise_style": ("ExerciseStyle", None),
    "payment_frequency": ("PaymentFrequency", None),
    "day_count_convention": ("DayCountConvention", None),
    "trade_status": ("TradeStatus", None),
    "exchange": ("Exchange", None),
    "clearing_house": ("ClearingHouse", "asset_class_column"),
    "underlying": ("Underlying", "asset_class_column"),
    # coherent dates
    "trade_date": ("TradeDate", "tenor_column"),
    "maturity_date": ("MaturityDate", "trade_date_column"),
    "settlement_date": ("SettlementDate", "trade_date_column"),
    # rolling book: a constant-size book whose trades roll at maturity (opt-in via
    # Configs(rolling_book=True), which remaps trade_date/tenor/maturity_date/the key).
    "rolling_trade_date": ("RollingTradeDate", None),
    "rolling_tenor": ("RollingTenor", None),
    "rolling_maturity_date": ("RollingMaturityDate", None),
    "rolling_trade_id": ("RollingTradeId", "trade_date_column"),
    "rolling_scenario": ("RollingScenario", None),
}

# Ordered (specific first) string name patterns -> (semantic, confidence). First match
# wins, so e.g. currency_pair is tested before currency. The generic product-id pattern
# is registered by the core (semantics_registry), not here.
_STRING_PATTERNS: list[tuple[str, str, float]] = [
    (r"isin", "isin", 0.97),
    (r"cusip", "cusip", 0.97),
    (r"sedol", "sedol", 0.97),
    # eligible collateral (CSA field); matches the correct spelling and the common
    # "elligible" typo — `l+` covers one or two l's, prefix optional so bare
    # "collateral" also matches. String-gated, so a numeric collateral value is skipped.
    (r"(el+igible[_ ]?)?collateral", "eligible_collateral", 0.85),
    (r"(currency|ccy|fx)[_ ]?pair", "currency_pair", 0.9),
    (r"risk[_ ]?factor[_ ]?(group|class)|rf[_ ]?group|riskfactorgroup", "risk_factor_group", 0.9),
    (r"risk[_ ]?factor[_ ]?name|rf[_ ]?name", "risk_factor_name", 0.85),
    (r"sensi[_ ]?type|greek[_ ]?type|risk[_ ]?measure", "sensi_type", 0.9),
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

# Scalar-float name patterns (specific first). fx_rate before rate; pv before price.
_FLOAT_PATTERNS: list[tuple[str, str, float]] = [
    (r"notional|principal|face[_ ]?(value|amount|amt)?", "notional", 0.8),
    (r"fx[_ ]?rate|spot[_ ]?rate|exchange[_ ]?rate", "fx_rate", 0.85),
    (r"spread", "spread", 0.8),
    (
        r"\bpv\b|\bnpv\b|\bmtm\b|\bmtf\b|present[_ ]?value|"
        r"mark[_ ]?to[_ ]?(market|future)|market[_ ]?value",
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

# Array-float name patterns (specific first). Same name may map to a different semantic
# than the scalar table — the column's type decides which set the core consults.
_ARRAY_FLOAT_PATTERNS: list[tuple[str, str, float]] = [
    (r"\bpnl\b|pnl[_ ]?(series|vector|path|ts)|profit[_ ]?and[_ ]?loss", "pnl_series", 0.8),
    # Mark-to-Future (or an array-typed Mark-to-Market) is a valuation path — the array
    # twin of scalar pv.
    (r"\bmtf\b|\bmtm\b|mark[_ ]?to[_ ]?(future|market)", "mtf", 0.8),
    (r"return|\brtns?\b", "returns", 0.7),
    (r"curve|term[_ ]?structure|yield[_ ]?curve", "curve", 0.75),
    (r"\bsensis?\b|tenor[_ ]?sensi|sensi[_ ]?ladder|delta[_ ]?ladder", "tenor_sensitivities", 0.85),
    (r"sensitivit|\bdeltas\b|bucket|risk[_ ]?weights?|\bladder\b", "sensitivities", 0.8),
]

# Date name patterns (specific first). maturity/settlement derive from trade_date.
_DATE_PATTERNS: list[tuple[str, str, float]] = [
    (r"trade[_ ]?date|booking[_ ]?date|execution[_ ]?date|deal[_ ]?date", "trade_date", 0.85),
    (
        r"maturity[_ ]?date|expiry[_ ]?date|expiration[_ ]?date|^maturity$|^expiry$|end[_ ]?date",
        "maturity_date",
        0.85,
    ),
    (r"settlement[_ ]?date|value[_ ]?date|settle[_ ]?date|^settlement$", "settlement_date", 0.85),
]

# Parent-semantic -> child-semantics coherence links (children read the parent column).
_LINKS: dict[str, tuple[str, ...]] = {
    # asset_class hierarchy (categorical children); conditional numeric children are
    # appended in register() from the priors' parent_semantic.
    "asset_class": ("instrument_type", "risk_factor_group", "clearing_house", "underlying"),
    "counterparty": (
        "counterparty_sector",
        "counterparty_country",
        "counterparty_ticker",
        "counterparty_lei",
    ),
    "tenor": ("trade_date",),
    "trade_date": ("maturity_date", "settlement_date"),
    "maturity_date": ("tenor_sensitivities",),
}


def register(registry):
    """Add every market_finance semantic, pattern, link and classifier to ``registry``."""
    from datagenic.domains.market_finance._data import (
        ARRAY_QUANTITY_PRIORS,
        CONDITIONAL_ARRAY_QUANTITY_PRIORS,
        CONDITIONAL_QUANTITY_PRIORS,
        QUANTITY_PRIORS,
    )

    # categorical / date / identifier / special generators
    for semantic, (cls, parent_arg) in _CATEGORICAL_GENERATORS.items():
        registry.add_generator(semantic, _GEN, cls, {}, parent_arg)

    # numeric quantity generators, parameterised by book-agnostic priors (core generators)
    for semantic, prior in QUANTITY_PRIORS.items():
        registry.add_generator(
            semantic,
            "datagenic.generators",
            prior["generator"],
            {"mu": prior["mu"], "sigma": prior["sigma"], "distribution": prior["distribution"]},
            None,
            numeric_kind="scalar",
        )
    for semantic, prior in CONDITIONAL_QUANTITY_PRIORS.items():
        registry.add_generator(
            semantic,
            "datagenic.generators",
            "ConditionalFloats",
            {"cases": prior["cases"], "default": prior["default"]},
            "by_column",
            numeric_kind="scalar_conditional",
        )
    for semantic, prior in ARRAY_QUANTITY_PRIORS.items():
        registry.add_generator(
            semantic,
            "datagenic.generators",
            prior["generator"],
            {k: v for k, v in prior.items() if k != "generator"},
            None,
            numeric_kind="array",
        )
    for semantic, prior in CONDITIONAL_ARRAY_QUANTITY_PRIORS.items():
        registry.add_generator(
            semantic,
            "datagenic.generators",
            "ConditionalArrayFloats",
            {k: v for k, v in prior.items() if k not in ("parent_semantic", "by_column")},
            "by_column",
            numeric_kind="array_conditional",
        )

    # name patterns, per type-kind
    for regex, semantic, conf in _STRING_PATTERNS:
        registry.add_pattern("string", regex, semantic, conf)
    for regex, semantic, conf in _FLOAT_PATTERNS:
        registry.add_pattern("float", regex, semantic, conf)
    for regex, semantic, conf in _ARRAY_FLOAT_PATTERNS:
        registry.add_pattern("array_float", regex, semantic, conf)
    for regex, semantic, conf in _DATE_PATTERNS:
        registry.add_pattern("date", regex, semantic, conf)

    # coherence links: static hierarchy + the asset_class-conditional numeric children
    for parent, children in _LINKS.items():
        registry.add_link(parent, children)
    asset_class_conditional = [
        semantic
        for semantic, prior in {
            **CONDITIONAL_QUANTITY_PRIORS,
            **CONDITIONAL_ARRAY_QUANTITY_PRIORS,
        }.items()
        if prior.get("parent_semantic") == "asset_class"
    ]
    registry.add_link("asset_class", asset_class_conditional)

    # value-based classifiers (lazy: numpy validators imported only when invoked)
    registry.add_value_classifiers(_build_value_classifiers)

    # scaffold transforms — opt-in via Configs(domains={"market_finance":
    # {"transforms": {"rolling_book": {...}}}}). Pure dict manipulation, numpy-free.
    from datagenic.domains.market_finance._transforms import apply_rolling_book

    registry.add_scaffold_transform("rolling_book", apply_rolling_book)


def _build_value_classifiers():
    """Ordered (semantic, predicate) pairs, most specific first. Each predicate takes a
    single string value and returns True if it looks like that semantic. Imports the
    numpy-backed checksum validators, so it is called lazily (value inference only)."""
    import re as _re

    from datagenic.domains.market_finance import _data as fd
    from datagenic.domains.market_finance import generators as gf

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
