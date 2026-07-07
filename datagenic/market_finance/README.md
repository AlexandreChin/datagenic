# `datagenic.market_finance`

A **financial-markets domain pack** for datagenic: coherent, realistic reference data and
column generators for a market-risk / trading dataset, plus the scaffold-time inference that
wires them automatically. Import a bare `datagenic` and none of this loads (no numpy); it's
pulled in only when you scaffold/generate with the finance semantics.

## What it gives you

Instead of `prefix_N` placeholders, a scaffolded trade/position table comes out **coherent
and plausible**:

| asset class | instrument | isin | ccy | tenor | trade\_date | maturity\_date | notional |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Rates | IR Swap | DE000A1B2C39 | EUR | 5Y | 2024-01-10 | 2029-01-08 | 24,300,000 |
| Equity | Equity Option | GB00BH4HKS39 | GBP | 1Y | 2024-03-02 | 2025-03-02 | 88,400 |

Coherence is the point: `(asset_class, instrument_type)` pairs are always taxonomy-valid, a
`notional` is scaled by asset class, `maturity_date = trade_date + tenor`, and a
`counterparty`'s sector/country/ticker/LEI all agree.

## Layout

```
market_finance/
  generators.py            column generators (same contract as datagenic.generators)
  _data.py                 pure-data vocabularies + numeric priors + the counterparty loader
  data/counterparties/*.csv real large-cap names + sector/country, per index
  __init__.py
```

## Identifiable fields (~50)

Detected by column **name** (`heuristic`), **sample values** (`values`), or an **LLM**
(`agent`), and **type-gated** (string / scalar float / `array(float)` / date).

**Categorical & identifier** (string): `asset_class`, `instrument_type`\*, `risk_factor_group`\*,
`currency`, `currency_pair`, `tenor`, `credit_rating`, `gics_sector`, `isin`/`cusip`/`sedol`
(valid check digits), `instrument_name`, `risk_factor_name`, `counterparty`,
`counterparty_sector`\*, `counterparty_country`\*, `counterparty_ticker`\*, `counterparty_lei`\*
(valid ISO 7064 check), `trading_book`, `desk`, `side`, `option_type`, `exercise_style`,
`payment_frequency`, `day_count_convention`, `trade_status`, `exchange`, `clearing_house`\*,
`underlying`\*.

**Scalar numeric** (float): `notional`\* (scaled by asset class), `price`, `pv`, `fx_rate`,
`quantity`, `rate`, `spread`, `pnl` (signed), `volatility`, `strike`, `spot`, `var`,
`expected_shortfall`, and the Greeks `delta`/`gamma`/`vega`/`theta`/`rho`.

**Array numeric** (`array(float)`): `pnl_series`\* (scaled by asset class), `returns`, `curve`,
`sensitivities`, and `sensis`\* — a **tenor-bucketed sensitivity ladder** (see below).

**Dates**: `trade_date`, `maturity_date`\* (`= trade_date + tenor`), `settlement_date`\*
(`= trade_date + lag`).

\* **coherent** — reads a parent column from the partial row (asset\_class, counterparty, or
trade\_date) and stays consistent with it; the scaffolder orders the parent first.

## Tenor-bucketed sensitivities

`TenorSensitivities` (semantic `tenor_sensitivities`, e.g. a `sensis` column) emits a **risk
ladder** — an `array(float)` aligned to a tenor grid (`DEFAULT_TENOR_GRID`, or a `tenor_grid`
arg / reference table). For each row it is:

- **scaled by `asset_class`** — a Rates rate-delta dwarfs an Equity one (`SENSI_MAGNITUDE_BY_ASSET_CLASS`);
- **shaped by `instrument_type`** — a *bullet* (one non-zero bucket at maturity) for
  zero-coupon-like instruments (`BULLET_INSTRUMENTS`: FX forwards, futures, ETFs, FRAs…), a
  *ladder* (all live buckets) for swaps, options, coupon bonds, CDS…;
- **zeroed beyond maturity** — buckets past the instrument's remaining maturity
  (`maturity_date − as-of`, snapped to the nearest grid bucket) are `0`, so a trade maturing
  in ~1M has non-zero sensis only up to the `1M` bucket. It `recall`s per date, so the live
  region shrinks as the trade ages (while `trade_date`/`maturity_date` stay fixed).

Companion generators: `SensiType` (`delta`/`gamma`/`vega`/`theta`/`rho`) and `TenorGrid` (the
axis). Works whether `sensis` is a column on the trades table or a row in a separate
sensitivities table — it just reads `asset_class`/`instrument_type`/`maturity_date` from the
partial row.

## Using it

```bash
datagenic generate-data-config trades_ir.json -o config.json --infer-semantics heuristic
datagenic profile ./samples -o config.json --infer-semantics values   # fit from real data
datagenic review-config config.json                                    # inspect the picks
```

```python
from datagenic import Configs
Configs(input=ir, semantic_inferrer="heuristic").dump(file_path="config.json")
```

Every inferred column is annotated under `_inferred` and is fully hand-editable. You can also
wire a generator by hand: `{"udf_module": "datagenic.market_finance.generators",
"udf_name": "ClearingHouse", "args": {...}}`.

**Accuracy ladder** (numeric fields): `heuristic` = a curated prior (plausible guess);
`values`/`profile` = fitted from your sample (family + params, per-category when conditional);
`agent` = proposed for the long tail. Categoricals/identifiers are drawn from curated vocabularies.

## Rolling book (live positions across dates)

By default trade attributes are frozen at inception, so a multi-date run is the *same*
trades observed over time — a short-dated trade booked on the first date has matured by
later dates. For a **live book** — a constant-size set of positions where matured trades
drop out and are replaced by freshly-booked ones — scaffold with `rolling_book=True`:

```python
Configs(input=ir, semantic_inferrer="heuristic", rolling_book=True).dump(file_path="config.json")
```

Each row is a **slot** running a deterministic renewal process: a trade is booked, matures,
and the slot immediately re-books a new trade (new tenor, new `trade_id`). At any as-of date
the slot emits the trade whose `[trade_date, maturity)` interval contains it, so:

- **every emitted trade is live** — `trade_date ≤ as-of < maturity_date`, at *every* date;
- **`maturity_date = trade_date + tenor`** stays exact (same active trade);
- the book is **constant size** — short-tenor slots turn over (new id each roll), long-tenor
  trades persist; the pool never shrinks;
- it is **byte-reproducible** across dates, shards and workers — the schedule is a pure
  function of `(global_seed, batch, slot)`, never the as-of date.

`rolling_book=True` remaps the inferred `trade_date` / `tenor` / `maturity_date` and the single
unicity key to `RollingTradeDate` / `RollingTenor` / `RollingMaturityDate` / `RollingTradeId`
(so it needs an inferrer). Bias toward short maturities via those generators' `tenors=[...]` /
`probabilities=[...]` args. `trade_id` is emitted as a packed int; the key sort is skipped
(`scale_skip_global_key_sort`) so row-index → slot stays stable across dates.

## Counterparties (real index constituents)

`data/counterparties/*.csv` hold real large-cap names + `sector` (GICS) + `country`, one file
per index: CAC 40, DAX 40, EURO STOXX 50, SMI, FTSE 100, S&P 500, Nikkei 225, TSX 60, ASX,
HSI, Nifty 50, KOSPI, Ibovespa, Taiwan, and European extras (Nordics/Iberia/Benelux/Italy) —
a **point-in-time snapshot** of major constituents (not complete or current index membership).
**~1,040 unique names**, unioned and deduped by `_data._load_counterparty_metadata()`, exposed
as `COUNTERPARTIES` + `COUNTERPARTY_METADATA`. **Grow/refresh** by editing a CSV or dropping in
a new `<index>.csv` (header `name,sector,country`) — the pool and the `cardinality()` ceiling
rise automatically. If the package data is missing, a small fictional fallback is used.

## Extending it

- **A vocabulary / prior**: add to `_data.py` (categoricals) or the `*_PRIORS` maps (numeric),
  and a `SEMANTIC_GENERATORS` entry + name pattern in `datagenic/semantics.py`.
- **A coherent field**: give the generator a parent-column arg (read via `_column_values`)
  and add its semantic to the right `_*_CHILDREN` tuple so inference links + orders it.
- Determinism holds automatically: draw only from the injected `_rng`, and generation stays
  byte-identical across shards.
