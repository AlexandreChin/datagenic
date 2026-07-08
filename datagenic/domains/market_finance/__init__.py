"""Financial-markets domain pack for datagenic (``market_finance``).

Coherent, realistic reference data + generators for a market-risk / trading
dataset: the ``asset_class → instrument_type → risk_factor_group`` taxonomy,
market dimensions (currency, tenor, rating, sector), synthetic identifiers
(ISIN/CUSIP/SEDOL/LEI with valid check digits), real large-cap counterparties +
metadata, trade attributes (side, option type, clearing house, dates…), and
numeric priors (notional, PnL series, Greeks…).

Two submodules:

- ``datagenic.domains.market_finance.generators`` — the column generators (same contract
  as ``datagenic.generators``; draw from the injected ``_rng``).
- ``datagenic.domains.market_finance._data`` — pure-data vocabularies + numeric priors,
  plus the bundled counterparty CSVs under ``data/counterparties/``.

These are wired automatically by ``datagenic.semantics`` when scaffolding with
``--infer-semantics``. Kept as a subpackage so nothing here is imported by a bare
``import datagenic`` (no numpy pulled). See ``README.md`` for the full field list.

``register(registry)`` is the discovery hook the semantic registry calls to add this
pack's semantics/patterns/links/classifiers; importing it stays numpy-free (only
literals + the pure-data priors), so discovery does not pull the generators.
"""

from datagenic.domains.market_finance._semantics import register

__all__ = ["register"]
