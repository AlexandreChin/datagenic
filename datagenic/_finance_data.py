"""Curated financial-markets vocabularies for ``generators_finance``.

Pure data, no third-party imports — keeps the module cheap and lets the finance
generators draw coherent, realistic reference values. The taxonomy is aligned to
the standard FRTB-style risk classes (Rates/GIRR, Credit/CSR, Equity, FX,
Commodity, Inflation) so it reflects how a market-risk platform organises data.

The maps are the single source of truth for cross-column coherence: an
``instrument_type`` and a ``risk_factor_group`` are always valid for their row's
``asset_class`` because the generators sample children from these maps keyed by
the parent value.
"""

# ── Core taxonomy ────────────────────────────────────────────────────────────

ASSET_CLASSES = ["Rates", "Credit", "Equity", "FX", "Commodity", "Inflation"]

INSTRUMENT_TYPES_BY_ASSET_CLASS = {
    "Rates": [
        "IR Swap",
        "Swaption",
        "Cap/Floor",
        "FRA",
        "OIS",
        "Cross-Currency Swap",
        "Government Bond",
        "Bond Future",
    ],
    "Credit": [
        "CDS",
        "CDS Index",
        "Corporate Bond",
        "Credit-Linked Note",
        "Total Return Swap",
    ],
    "Equity": [
        "Equity Spot",
        "Equity Option",
        "Equity Future",
        "Equity Swap",
        "Variance Swap",
        "Dividend Future",
        "ETF",
    ],
    "FX": [
        "FX Spot",
        "FX Forward",
        "FX Swap",
        "FX Option",
        "Non-Deliverable Forward",
    ],
    "Commodity": [
        "Commodity Future",
        "Commodity Option",
        "Commodity Swap",
        "Commodity Forward",
    ],
    "Inflation": [
        "Inflation Swap",
        "Inflation-Linked Bond",
        "Inflation Cap/Floor",
    ],
}

RISK_FACTOR_GROUPS_BY_ASSET_CLASS = {
    "Rates": ["IR Curve", "IR Volatility", "Basis Spread"],
    "Credit": ["Credit Spread", "Credit Volatility", "Recovery Rate"],
    "Equity": ["Equity Spot", "Equity Volatility", "Dividend Yield", "Equity Repo"],
    "FX": ["FX Spot", "FX Volatility"],
    "Commodity": ["Commodity Price", "Commodity Volatility"],
    "Inflation": ["Inflation Curve", "Inflation Volatility"],
}

# Flattened fallbacks used when a parent asset_class column is absent (the
# hierarchy generators then degrade to an independent, possibly-incoherent draw).
ALL_INSTRUMENT_TYPES = [it for its in INSTRUMENT_TYPES_BY_ASSET_CLASS.values() for it in its]
ALL_RISK_FACTOR_GROUPS = [
    rfg for rfgs in RISK_FACTOR_GROUPS_BY_ASSET_CLASS.values() for rfg in rfgs
]

# ── Market dimensions ────────────────────────────────────────────────────────

# ISO 4217, ordered roughly by liquidity (majors first).
CURRENCIES = [
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "HKD",
    "SGD",
    "CNY",
    "ZAR",
    "MXN",
    "BRL",
]

# G10-style major pairs (market convention base/quote).
CURRENCY_PAIRS = [
    "EUR/USD",
    "USD/JPY",
    "GBP/USD",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "EUR/GBP",
    "EUR/JPY",
    "GBP/JPY",
    "EUR/CHF",
    "USD/SGD",
]

TENORS = [
    "ON",
    "1W",
    "2W",
    "1M",
    "3M",
    "6M",
    "9M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "15Y",
    "20Y",
    "30Y",
]

# S&P long-term scale (investment grade through default).
SP_RATINGS = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "D",
]

# Moody's long-term scale.
MOODYS_RATINGS = [
    "Aaa",
    "Aa1",
    "Aa2",
    "Aa3",
    "A1",
    "A2",
    "A3",
    "Baa1",
    "Baa2",
    "Baa3",
    "Ba1",
    "Ba2",
    "Ba3",
    "B1",
    "B2",
    "B3",
    "Caa1",
    "Caa2",
    "Caa3",
    "Ca",
    "C",
]

# GICS sectors.
GICS_SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]

# ── Organisation / counterparty (clearly fictional) ──────────────────────────

COUNTERPARTIES = [
    "Northbridge Capital",
    "Helvetia Partners",
    "Meridian Securities",
    "Aurora Investment Bank",
    "Sterling & Roe",
    "Castlepoint Asset Mgmt",
    "Brightwater Markets",
    "Pinnacle Global",
    "Cumberland Trading",
    "Vantage Financial",
    "Greycliff Holdings",
    "Halcyon Capital",
    "Westgate Bank",
    "Solstice Markets",
    "Ironwood Securities",
    "Lakeshore Partners",
]

# Composed labels built from asset class + style + region (precomputed pools).
TRADING_BOOKS = [
    f"{ac} {style} {region}"
    for ac in ["Rates", "Credit", "Equity", "FX", "Commodity"]
    for style in ("Flow", "Trading")
    for region in ("EMEA", "AMER", "APAC")
]

DESKS = [
    f"{ac} {style}"
    for ac in ["Rates", "Credit", "Equity Derivatives", "FX Options", "Commodities"]
    for style in ("Desk", "Trading")
]

# Country prefixes for synthetic ISINs.
ISIN_COUNTRIES = ["US", "GB", "DE", "FR", "JP", "CH", "NL", "IT", "ES", "CA", "AU"]


# ── Numeric quantity priors ──────────────────────────────────────────────────
# Book-agnostic *defaults* for quantitative fact columns, so a scaffolded config
# yields realistic magnitudes out of the box (a notional in the millions, a
# right-skewed lognormal — not the generic N(1000, 100)). Each prior picks:
#   generator     — "RandomFloats" (stateless: the value is fixed across dates,
#                   e.g. a trade notional) or "BrownianFloats" (evolves over
#                   dates via recall, e.g. a price / PV that walks).
#   distribution  — "lognormal" (positive, right-skewed; ``mu`` is the median in
#                   real units, ``sigma`` the log-space spread) or "normal".
#   mu, sigma     — distribution parameters in the column's real units.
# These are guesses (see the "accuracy ladder"): plausible, editable, and
# annotated under ``_inferred``. For fidelity, fit them from a sample (``values``
# inference / ``profile``). Kept pure-data here; ``semantics`` builds the
# generator specs from them.
QUANTITY_PRIORS: dict[str, dict] = {
    "quantity": {
        "generator": "RandomFloats",
        "distribution": "lognormal",
        "mu": 10_000.0,
        "sigma": 1.2,
    },
    "price": {
        "generator": "BrownianFloats",
        "distribution": "lognormal",
        "mu": 100.0,
        "sigma": 0.4,
    },
    "pv": {
        "generator": "BrownianFloats",
        "distribution": "lognormal",
        "mu": 50_000.0,
        "sigma": 1.0,
    },
    "rate": {"generator": "RandomFloats", "distribution": "normal", "mu": 0.03, "sigma": 0.01},
    "spread": {"generator": "RandomFloats", "distribution": "normal", "mu": 0.005, "sigma": 0.002},
    "fx_rate": {
        "generator": "BrownianFloats",
        "distribution": "lognormal",
        "mu": 1.1,
        "sigma": 0.1,
    },
    # a scalar P&L / mark is signed (can be negative) -> normal, mean ~0. Distinct
    # from `pv` (a positive lognormal exposure) and from the array `pnl_series`.
    "pnl": {"generator": "RandomFloats", "distribution": "normal", "mu": 0.0, "sigma": 50_000.0},
}


# ── Array-typed quantity priors ──────────────────────────────────────────────
# Book-agnostic defaults for array/vector fact columns (a per-row time series or
# curve). Shape mirrors the array generators' args: generator +
# {distribution, mu, sigma, size_of_arrays, path}. ``path="walk"`` makes the array
# a cumulative per-row series (a PnL path); ``"iid"`` independent elements. Guesses
# — fit from a sample (``values`` / ``profile``) for fidelity.
ARRAY_QUANTITY_PRIORS: dict[str, dict] = {
    # a daily-returns series: signed, independent daily draws.
    "returns": {
        "generator": "RandomArrayFloats",
        "distribution": "normal",
        "mu": 0.0,
        "sigma": 0.02,
        "size_of_arrays": 250,
        "path": "iid",
    },
    # a rate/term-structure curve: small positive levels across a handful of tenors.
    "curve": {
        "generator": "RandomArrayFloats",
        "distribution": "normal",
        "mu": 0.03,
        "sigma": 0.01,
        "size_of_arrays": 15,
        "path": "iid",
    },
}


# ── Conditional numeric priors ───────────────────────────────────────────────
# For quantities whose scale depends on a categorical column: the value is drawn
# from a per-category distribution (an Equity trade notional ≈ $100k, a Rates
# IR-swap notional ≈ tens of millions — orders of magnitude apart). The numeric
# twin of the asset_class -> instrument_type taxonomy. Shape mirrors the
# ``ConditionalFloats`` args (``by_column`` / ``cases`` / ``default``); ``cases``
# medians are in real currency units. ``parent_semantic`` names the categorical
# semantic to condition on, so inference links the right column. Book-agnostic
# defaults — fit from a sample (``values`` / ``profile``) for fidelity.
CONDITIONAL_QUANTITY_PRIORS: dict[str, dict] = {
    "notional": {
        "parent_semantic": "asset_class",
        "by_column": "asset_class",
        "cases": {
            "Rates": {"distribution": "lognormal", "mu": 2.5e7, "sigma": 1.0},
            "Credit": {"distribution": "lognormal", "mu": 5.0e6, "sigma": 1.0},
            "Equity": {"distribution": "lognormal", "mu": 1.0e5, "sigma": 0.8},
            "FX": {"distribution": "lognormal", "mu": 1.0e7, "sigma": 0.9},
            "Commodity": {"distribution": "lognormal", "mu": 1.0e6, "sigma": 1.0},
            "Inflation": {"distribution": "lognormal", "mu": 2.0e7, "sigma": 0.9},
        },
        "default": {"distribution": "lognormal", "mu": 1.0e6, "sigma": 1.0},
    },
}


# ── Conditional array-quantity priors ────────────────────────────────────────
# Array/vector quantities whose scale depends on a categorical column — e.g. a PnL
# time series whose volatility depends on asset_class (a Rates book's daily PnL
# swings wider than an Equity book's). The array twin of CONDITIONAL_QUANTITY_PRIORS.
# Shape mirrors the ``ConditionalArrayFloats`` args: ``by_column`` / ``cases`` /
# ``default`` (per-category {distribution, mu, sigma}) plus shared ``size_of_arrays``
# / ``path``. Signed (normal), so a PnL path can go negative. Book-agnostic guesses.
CONDITIONAL_ARRAY_QUANTITY_PRIORS: dict[str, dict] = {
    "pnl_series": {
        "parent_semantic": "asset_class",
        "by_column": "asset_class",
        "size_of_arrays": 250,
        "path": "walk",
        "cases": {
            "Rates": {"distribution": "normal", "mu": 0.0, "sigma": 2.0e5},
            "Credit": {"distribution": "normal", "mu": 0.0, "sigma": 8.0e4},
            "Equity": {"distribution": "normal", "mu": 0.0, "sigma": 5.0e4},
            "FX": {"distribution": "normal", "mu": 0.0, "sigma": 1.0e5},
            "Commodity": {"distribution": "normal", "mu": 0.0, "sigma": 6.0e4},
            "Inflation": {"distribution": "normal", "mu": 0.0, "sigma": 1.5e5},
        },
        "default": {"distribution": "normal", "mu": 0.0, "sigma": 5.0e4},
    },
}
