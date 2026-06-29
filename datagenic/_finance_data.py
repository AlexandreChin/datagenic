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
