# datagenic — Implementation Plan: 5 Features

Implementation plans for five proposed features, grounded in the current codebase:

1. Null / missing-value injection
2. Distribution fitting ("profile from a sample")
3. Post-generation profiling / quality report
4. Cardinality-aware relationship enforcement
5. PII-shaped generators

Each plan states the goal, the design, the files touched, the determinism/sharding
implications, the test strategy, and an effort/risk estimate. Read the
cross-cutting constraints first — they bind every feature.

---

## Cross-cutting constraints (every feature must respect these)

1. **Determinism is the prime invariant.** Any randomness must draw from the
   framework-injected `_rng` (`numpy.random.Generator`), never `numpy.random.*`,
   `random`, `uuid`, or `time`. Column generators receive the per-unit
   `_unit_rng` shared across a table's columns (`engine.py:_make_rng`);
   postprocessors receive a `_rng` seeded from
   `sha256(seed | table:post | date | batch)`.

2. **One postprocessor slot per table.** `table["post_processor"]` is a *single
   dict*, not a list (`engine.py:1260`, `engine.py:1367`). Fact tables already
   spend that slot on `KeyEnforcer`. So **null injection and cardinality
   enforcement must NOT be new standalone postprocessors** — they would collide
   with FK enforcement. This plan routes them through (a) a central column-level
   mask in the engine and (b) an extension of `KeyEnforcer`'s rule vocabulary,
   respectively. An optional postprocessor-*chaining* refactor that would unlock
   cleaner designs is noted at the end.

3. **Golden baseline.** `tests/test_2_safety_net.py::test_golden_manifest_matches`
   hashes output against frozen fixtures (`tests/golden/golden_fixtures.py`).
   Adding new generators/rules is safe; the golden configs are explicit, so they
   never pick up new scaffolder defaults. Any change to *existing* generator
   output or default selection requires `python tests/golden/capture_baseline.py`
   and a deliberate review — a failure you did not intend is a determinism
   regression.

4. **Package boundary.** `import datagenic` must stay cheap — no heavy deps
   (e.g. `faker`, polars eager paths, cloud SDKs) imported eagerly. New public
   API goes through the lazy `_LAZY` map in `datagenic/__init__.py`.

5. **Two pre-existing bugs to fix opportunistically** (separate small PR, not in
   scope here): `engine.py:config_file_controls` lists `avro` as a supported
   output format but it is unimplemented in the writer, and
   `supported_storage_type = ["local", "azure_blob"]` omits `s3`/`sftp` despite
   those backends existing under `iokit/cloud_providers/`.

---

## Feature 1 — Null / missing-value injection

**Goal.** Declarative per-column null rates (and simple correlated-null
patterns), fully deterministic and shardable.

### Design — central column mask (recommended)

Rather than a postprocessor (which would collide with `KeyEnforcer` — see
constraint 2), apply nulls in the engine right after each generator returns its
values, in both `_generate_columns` (`engine.py:843-865`) and `_recall_columns`
(`engine.py:952-977`). The per-column config gains an optional sibling key to
`args`:

```jsonc
"column_generator": { "udf_name": "...", "args": { ... } },
"null_spec": { "rate": 0.1 }                          // MCAR: 10% missing at random
// or correlated (MAR): null when a condition on already-built columns holds
"null_spec": { "rate": 0.3, "when": "_table['status'] == 'pending'" }
```

Add a helper `_apply_null_mask(series, null_spec, rng, partial_table, is_unicity_key)`
(in `engine.py`, or `dtypes.py` next to the other frame helpers):

- Draw the mask from the **same `_unit_rng`** already in scope → byte-identical,
  shard-safe, `INPUT_SCOPE`-neutral (purely batch-local).
- **Hard guard: never null a unicity key** (`is_unicity_key=True`). Nulls would
  break the `unique(subset=unicity_keys)` dedup and corrupt FK pools. Reject at
  config time in `validate()` if a `null_spec` is attached to a key column.
- Apply *before* the dedup/sort in `_generate_columns` so non-key nulls flow
  through naturally.
- **Drawing order matters for determinism:** draw the mask immediately after the
  column's values, consuming `_unit_rng` in a fixed sequence. Document that
  adding `null_spec` to a column shifts the RNG stream for that unit — acceptable
  because it is a config change, so the config hash changes and checkpoints
  invalidate correctly.

### Files

- `engine.py` — mask helper + call sites in `_generate_columns`/`_recall_columns`
  + `validate()` guard against `null_spec` on key columns.
- `scaffolding.py` — `Columns`/`Configs` optionally emit `null_spec` from an IR
  hint, e.g. `{"name": ..., "type": ..., "null_rate": 0.1}`.
- `_validate_output_schema` (`engine.py:507`) already tolerates nulls, but verify
  a *fully*-null column's dtype stays compatible (`dtypes._categories_compatible`)
  — add a regression test.
- CLAUDE.md / skill reference note on `null_spec`.

### Testing

- Determinism: same seed → identical null positions.
- Shard-equivalence: full run vs 2 shards byte-identical with nulls present.
- Key-column rejection (config-time error).
- Correlated `when` mask produces nulls only on matching rows.
- Fully-null column edge case (dtype compatibility).
- Golden manifest unaffected (new opt-in field).

### Effort / risk

~0.5–1 day. **Low risk.**

---

## Feature 2 — Distribution fitting / "profile from a sample"

**Goal.** `datagenic profile <sample> -o ir.json [--config config.json]` inspects
real (or sample) data and emits IR + a scaffolded config with generator args
pre-filled from observed statistics. Closes the loop so users stop hand-tuning
every config.

### Design — pure scaffold-time tool (zero determinism impact)

New module `datagenic/profiling.py`:

```python
def profile_frame(df: pl.DataFrame, table_name, keys=None) -> dict   # IR table entry + generator hints
def profile_paths(paths: list[str], storage) -> dict                 # full IR {name, tables:[...]}
```

Per column, infer:

- **dtype** → IR type (`string` / `int` / `float` / `date` / `datetime` / `array(...)`).
- **cardinality** (`n_unique`) → sets `cardinality`; low-cardinality categoricals
  → emit `list_of_values` + observed-frequency `probabilities` (consumed directly
  by `ListOfValues`/`RandomStrings`/`RandomIntegers` — `generators.py:67,121,149`).
- **floats** → `mu`/`sigma` (mean/std) for `RandomFloats`/`BrownianFloats`.
- **dates/datetimes** → min/max range.
- **null fraction** → emit `null_spec.rate` (ties into Feature 1).
- **key candidates** → columns where `n_unique == height` flagged as `keys`.
- **FK candidates** (multi-file mode) → a child column whose value set ⊆ another
  table's key set → emit `foreign_keys` (consumed by `KeyEnforcer` via the
  existing IR → config path).

Then reuse the existing `Configs(ir).dump()` path unchanged — profiling only
*fills the IR and overrides generator args*; it never touches the engine.

### Files

- New `datagenic/profiling.py`.
- `cli.py` — new `profile` subcommand. Read via `iokit.storages.Storages`
  (`read_parquet_as_polars`, `ls_dirs`/`ls_files`) so it works on cloud samples too.
- `__init__.py` — lazy export (e.g. `profile_paths`).
- Stats helpers factored to be shared with Feature 3.

### Testing

- Round-trip: generate → profile → scaffold → regenerate → assert dtypes,
  cardinality, and numeric ranges land in the expected band.
- Categorical frequency recovery (`probabilities` ≈ observed).
- FK-candidate detection on the `samples/` model.
- Marked non-integration (local parquet).

### Effort / risk

~2–3 days. **Low risk** (read-only, scaffold-side). Note: this is
descriptive-statistics matching, **not** copula / joint-distribution synthesis —
set that expectation explicitly in docs.

---

## Feature 3 — Post-generation profiling / quality report

**Goal.** `datagenic profile-output <config.json>` (or a `--report` flag on
`generate-data`) emits `quality_report.json` next to `generation_summary.json`:
per-column stats, null counts, cardinality, numeric distribution summary, a
correlation matrix, and — the high-value part — **expectation checks** against
the config (observed null-rate vs configured `null_spec.rate`, observed
cardinality vs `cardinality` arg, FK orphan count = rows whose FK value is not in
the parent key pool).

### Design

New module `datagenic/quality.py` reusing Feature 2's stats helpers. Reads
generated output through the **storage layer** (`Storages.ls_dirs`/`ls_files`/
`read_parquet_as_polars`) using the same
`{prefix}{table}/{date}/{table}-{batch}.parquet` layout the engine writes
(`engine.py:_checkpoint_path`, `batch_output_paths`). Aggregates across
dates/batches per table. Output mirrors the `generation_summary.json` shape
(`engine.py:_write_summary`) so the two sit side by side:

```jsonc
{ "tables": { "orders": {
    "columns": { "amount": { "nulls": 251, "null_rate": 0.10, "n_unique": 1873,
                             "min": ..., "max": ..., "mean": ..., "std": ... } },
    "fk_checks": [ { "col": "customer_id", "parent": "customers.id", "orphans": 0 } ],
    "expectations": [ { "check": "null_rate", "col": "amount",
                        "want": 0.1, "got": 0.103, "ok": true } ]
} } }
```

**Integration choice.** Keep it a *separate command* (not inline in `generate()`)
so it never slows generation and can run against any existing output dir, local
or cloud. Optionally add a thin `--report` flag on `generate-data` that calls it
post-run.

### Files

- New `datagenic/quality.py`.
- `cli.py` — `profile-output` subcommand (+ optional `--report` on `generate-data`).
- `__init__.py` — lazy export.
- Shared stats module with Feature 2.

### Testing

- Run a golden config → assert FK orphan count is 0 (integrity holds).
- Null-rate expectation matches the Feature 1 config within tolerance.
- Cardinality within tolerance of the configured value.
- Cloud path behind `-m integration`.

### Effort / risk

~2 days. **Low risk.** Build *after* Feature 2 (share stats code) and *after*
Feature 1 (so null-rate expectations have something to check).

---

## Feature 4 — Cardinality-aware relationship enforcement

**Goal.** Go beyond "FK values are valid" to controlling relationship *shape*:
nullable/optional FKs, and fan-out distribution (each parent referenced ~N times).

### Design — extend `KeyEnforcer` rules (not a new postprocessor)

Add to the rule vocabulary in `postprocessors.py:_apply` (`:104-220`). The parent
table is already present in `_input_tables` for the existing sample rule, so no
new wiring is required.

- **Nullable / optional FK** — fully shardable, low risk:
  ```jsonc
  { "col": "shipment_id", "from": "shipments.id", "null_rate": 0.2 }
  ```
  After the existing pool-sample, draw a mask from the rule's deterministic
  `_seed`/`_rng` and null the chosen fraction. Same key-guard as Feature 1.

- **Fan-out / multiplicity** — *per-batch distributional* guarantee:
  ```jsonc
  { "col": "customer_id", "from": "customers.id",
    "fanout": { "dist": "uniform", "min": 1, "max": 5 } }
  ```
  Within a batch, sample parents so the per-parent child count follows the target
  distribution (build a weighted parent multiset, then `sample` to the row count)
  using the rule's deterministic `_seed`.

### The honest constraint (must be documented)

A **global** hard min/max fan-out ("every customer has 1–5 orders across the
*entire* dataset") spans batches, so it requires cross-batch state →
`INPUT_SCOPE="global"` → **rejected by the shardability gate**
(`engine.py:_validate_distributable`). Therefore:

- Nullable FK and *per-batch* distributional fan-out: ship as default
  (`INPUT_SCOPE="batch"`, stays shardable — the common useful case).
- Global hard cardinality: only offer on `single_batch` reference tables, or have
  `KeyEnforcer` set `INPUT_SCOPE="global"` *when a global-fanout rule is present*
  so `check-distributable` correctly refuses to shard it. Do **not** silently
  pretend it is shardable.

### Files

- `postprocessors.py` — `KeyEnforcer._apply` new rule branches + conditional
  `INPUT_SCOPE`.
- `scaffolding.py` — `_fk_rules` (`:606`) emits `null_rate`/`fanout` from IR FK hints.
- `engine.py` — only if the global-scope conditional needs a hook.
- CLAUDE.md / skill reference on the FK rule vocabulary.

### Testing

- Nullable-FK orphan check (orphans = 0, null-rate matches) via Feature 3.
- Per-batch fan-out distribution within tolerance.
- **Shard-equivalence for batch-scope rules** (full vs 2 shards byte-identical).
- `check-distributable` *rejects* a global-fanout sharded config (negative test).
- Golden manifest for existing FK rules unchanged (new branches are additive).

### Effort / risk

~3–4 days. **Highest risk** — touches the sharding invariant. Do it last and use
`orchestration.check_distributable`'s full-vs-2-shards proof as the gate.

---

## Feature 5 — PII-shaped generators

**Goal.** Realistic names/emails/addresses/phones/companies/UUIDs instead of
`prefix_N`, deterministic and dependency-light.

### Design — vendored value lists, sampled via `_rng` (recommended over `faker`)

New generator classes in `datagenic/generators.py` following the existing
contract (`__init__` sets `_generator_description`; `call(self,
_number_of_rows_per_batch, ..., **kwargs)`; randomness via `_rng_from(kwargs)`):

- `RandomNames`, `RandomFirstNames`, `RandomLastNames`, `RandomEmails` (derive
  from sampled names + a domain pool), `RandomAddresses`, `RandomPhoneNumbers`,
  `RandomCompanyNames`, `RandomUUIDs` (deterministic UUID derived from
  `_rng.integers`, **not** `uuid4`).
- Unicity-key variants must produce **disjoint pools per batch** — reuse the
  existing `_cardinality_range` offset trick (`generators.py:17`) so
  emails/usernames stay globally unique across batches/shards, exactly as
  `RandomStrings` does today.
- Curated word lists live in a small data module (e.g. `datagenic/_pii_data.py`)
  — **no `faker` dependency** imported eagerly, preserving the cheap
  `import datagenic`. Offer `faker` as an *optional* opt-in generator
  (lazy-imported, seeded deterministically from `_rng`) but flag that faker's
  cross-version output drift makes it unsafe for the golden baseline — keep the
  vendored generators as the determinism-stable default.

### Scaffolding wiring

Two ways to select PII generators in `Columns.__init__` (`scaffolding.py:130-175`):

1. **Explicit semantic hint in IR** (preferred, unambiguous):
   `{"name": "email", "type": "string", "semantic": "email"}` → map `semantic` →
   PII generator.
2. **Column-name heuristic** (convenience, opt-in via a `Configs` flag so it
   cannot silently change scaffolded output for existing users): name matches
   `email|name|address|phone|company` → pick the PII generator. Off by default.

### Files

- `datagenic/generators.py` (+ new `datagenic/_pii_data.py`).
- `scaffolding.py` — `Columns` selection on `semantic`/heuristic.
- CLAUDE.md / skill reference. New generators are auto-discoverable (config
  references `datagenic.generators`).

### Determinism note

Pure `_rng` sampling → byte-identical, shard-safe, fits the existing model with
zero engine changes. New classes do not touch the golden baseline; only a
*default-selection* change would, so gate the name-heuristic behind a flag.

### Testing

- Determinism per generator.
- Per-batch uniqueness for key variants (the disjoint-pool guarantee).
- Shard-equivalence.
- Email-derives-from-name consistency.
- Scaffolder picks the PII generator on a `semantic` hint.

### Effort / risk

~2–3 days. **Low risk.**

---

## Feature 5 (applied) — Financial-markets domain generators

The primary use case for Feature 5 is **financial-markets reference data**:
realistic, internally **coherent** asset class / instrument / risk-factor
taxonomies, plus market dimensions, synthetic identifiers, and org/counterparty
names. Coherence is the differentiator — an `instrument_type` must be valid for
its `asset_class`, and a `risk_factor_group` must align with both.

### Module layout

- `datagenic/market_finance/_data.py` — the curated vocabularies and the parent→children
  maps (pure data, numpy-only, cheap import).
- `datagenic/market_finance/generators.py` — the generator classes (referenced in config
  as `"udf_module": "datagenic.market_finance.generators"`). Same contract as
  `generators.py`: `__init__` sets `_generator_description`; `call(self,
  _number_of_rows_per_batch, ..., **kwargs)`; randomness via `_rng_from(kwargs)`.

### Curated taxonomy (FRTB / standard risk classes)

```
asset_class -> instrument_types                                              -> risk_factor_groups
Rates      : IR Swap, Swaption, Cap/Floor, FRA, OIS, XCcy Swap, Govt Bond,    : IR Curve, IR Volatility, Basis Spread
             Bond Future
Credit     : CDS, CDS Index, Corporate Bond, CLN, Total Return Swap           : Credit Spread, Credit Volatility, Recovery Rate
Equity     : Equity Spot, Equity Option, Equity Future, Equity Swap,          : Equity Spot, Equity Volatility, Dividend Yield,
             Variance Swap, Dividend Future, ETF                                Equity Repo
FX         : FX Spot, FX Forward, FX Swap, FX Option, NDF                     : FX Spot, FX Volatility
Commodity  : Commodity Future, Commodity Option, Commodity Swap,              : Commodity Price, Commodity Volatility
             Commodity Forward
Inflation  : Inflation Swap, Inflation-Linked Bond, Inflation Cap/Floor       : Inflation Curve, Inflation Volatility
```

Market dimensions:

- **Currencies** (ISO 4217): USD, EUR, GBP, JPY, CHF, CAD, AUD, NZD, SEK, NOK,
  HKD, SGD, CNY, ... (weighted toward majors).
- **Currency pairs**: EUR/USD, USD/JPY, GBP/USD, USD/CHF, ... (G10 majors).
- **Tenors**: ON, 1W, 2W, 1M, 3M, 6M, 9M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 15Y, 20Y, 30Y.
- **Credit ratings**: S&P scale AAA..D (and an optional Moody's scale Aaa..C).
- **GICS sectors**: Energy, Materials, Industrials, Consumer Discretionary,
  Consumer Staples, Health Care, Financials, Information Technology,
  Communication Services, Utilities, Real Estate.

### Generators

**Coherent hierarchy (read the partial `_table`):**

- `FinancialAssetClass` — samples an asset class. Args: optional
  `list_of_values`/`probabilities` to restrict/weight.
- `FinancialInstrumentType` — arg `asset_class_column="asset_class"`. Reads
  `_table[asset_class_column]` and, per row, samples an instrument type valid for
  *that row's* asset class. Falls back to a flat sample over all instrument types
  if the parent column is absent (safe superset).
- `FinancialRiskFactorGroup` — arg `asset_class_column="asset_class"`. Same
  pattern, sampling a coherent risk-factor group.

Vectorize per distinct parent value (group rows by asset class, draw children for
each group from `_rng`) so the cost is O(distinct asset classes), not O(rows).

**Independent market dimensions:**

- `Currency`, `CurrencyPair`, `Tenor`, `CreditRating` (arg `scale="SP"|"Moody"`),
  `GICSSector` — each a weighted sample from its vocabulary via `_rng`.

**Composed human-readable names (read `_table`):**

- `InstrumentName` — e.g. `"EUR IR Swap 5Y"`, composed from the row's currency +
  instrument type + tenor columns (args name those source columns; each defaults
  to the conventional name, falls back gracefully if absent).
- `RiskFactorName` — e.g. `"EUR IR Curve 10Y"`, composed from currency +
  risk-factor group + tenor.

**Synthetic identifiers (valid check digits):**

- `ISIN` — 12 chars: 2-letter country prefix + 9-char body + 1 check digit
  (Luhn mod-10 over the digit-expanded body).
- `CUSIP` — 9 chars with mod-10 check digit.
- `SEDOL` — 7 chars with weighted mod-10 check digit.
- Identifier bodies are generated from `_rng`; for **unicity-key** identifier
  columns, generate the numeric body with the disjoint per-batch offset trick
  (`generators.py:_cardinality_range`) so codes stay globally unique across
  batches/shards, then append the deterministic check digit. The result both
  looks real and passes standard validators.

**Org / counterparty:**

- `Counterparty` / `LegalEntity` — sample from a vendored list of synthetic
  (clearly fictional) entity names.
- `TradingBook` / `Desk` — composed, e.g. `"Rates Trading EUR"`,
  `"FX Options EMEA"` = asset class + style + region.

### Scaffolding wiring — automatic semantic inference (no IR hand-tagging)

The user does **not** annotate the IR. During `generate-data-config` the
scaffolder *infers* each column's finance semantic automatically, wires the
matching `datagenic.market_finance.generators` generator, and annotates the choice so
the user can review and hand-edit the result. Scaffolding is not covered by the
byte-identical invariant (only *generation* is), so fuzzy/LLM inference is safe
here: once the config is frozen and reviewed, generation stays deterministic.

#### The `semantic_inferrer` hook (mirrors the existing `adaptor`)

Add a pluggable inferrer to `Configs`, exactly parallel to the existing
`adaptor=fn` extension point (`scaffolding.py:324`, CLI `--adaptor
module:function` at `cli.py:146`):

```python
Configs(input=ir, semantic_inferrer=fn, ...)   # fn(table_name, columns, samples=None) -> mapping
```

`fn` returns, per column, a record:

```python
{ "asset_class":     {"semantic": "asset_class",     "confidence": 0.95, "parent": None,           "alternatives": []},
  "instrument_type": {"semantic": "instrument_type", "confidence": 0.88, "parent": "asset_class",  "alternatives": ["product_type"]},
  "ccy":             {"semantic": "currency",        "confidence": 0.99, "parent": None,           "alternatives": []} }
```

`Columns.__init__` (`scaffolding.py:61`) gains a `semantic` branch that
short-circuits the type-based default selection when a semantic was inferred,
mapping it (and `parent` → the generator's `*_column` arg) to a finance generator.

#### Three inference strategies (selected by `--infer-semantics`)

| Strategy | Deps | Basis | Determinism |
|----------|------|-------|-------------|
| `heuristic` (default) | none | column name/type synonyms (`ccy`/`cur`→currency, `isin`→isin, `asset_class`/`assetclass`→asset_class, `tenor`/`maturity`→tenor, `rating`→credit_rating, …) | deterministic |
| `values` | none (reuses Feature 2 stats) | sample-backed: values match the ISO-4217 set, ISIN/CUSIP/SEDOL regex+checksum, or a taxonomy vocabulary | deterministic |
| `agent` | `anthropic`, **lazy** | LLM over names + sample values + any column descriptions; resolves ambiguity and infers the hierarchy (links `instrument_type`→`asset_class`) | non-deterministic → frozen, reviewable config |

**Package boundary:** datagenic ships the interface + the `heuristic` and
`values` inferrers (dependency-free, so `import datagenic` stays cheap). The
`agent` inferrer is opt-in — a lazily-imported module behind an extra
(`pip install datagenic[agent]`) or injected from the platform package — so a
bare import never pulls `anthropic`. The CLI resolves
`--infer-semantics heuristic|values|agent|module:function` the same way it
resolves `--adaptor`.

#### Annotation for manual fine-tuning

The scaffolder writes the inference into the config so the user can see and
override every guess. Each auto-selected column carries its existing
`description` plus an `_inferred` block:

```jsonc
{ "col_name": "instrument_type", "col_type": "string", "is_unicity_key": false,
  "column_generator": {
    "udf_module": "datagenic.market_finance.generators",
    "udf_name": "FinancialInstrumentType",
    "args": { "asset_class_column": "asset_class" },
    "description": "inferred semantic 'instrument_type' (confidence 0.88)"
  },
  "_inferred": { "semantic": "instrument_type", "confidence": 0.88,
                 "alternatives": ["product_type"], "by": "agent" } }
```

`_inferred` is ignored by the engine (it lives beside `column_generator`, not
inside it, and is not a `_`-prefixed *udf_arg*, so it is not subject to the
reserved-prefix rule). The user fine-tunes by editing plain JSON: swap the
`udf_name`, restrict `args.list_of_values`, set `probabilities`, or fix a wrong
guess. A `datagenic review-config config.json` helper lists every inferred
column sorted by confidence so low-confidence guesses surface first.

**Ordering constraint (must document):** a dependent column must appear *after*
its parent in the config `columns` list, because the engine builds columns
top-to-bottom and a hierarchy-aware generator reads only the columns built
before it (`engine.py:855`). The inferrer must therefore not name a `parent`
ordered after its dependent; `validate()` raises a clear error if it does
(mirroring the existing "must appear earlier" checks for dynamic
`number_of_rows`). When inference comes from a sample, the scaffolder can also
*reorder* columns so parents precede dependents.

### Determinism / sharding

All finance generators draw from the injected `_unit_rng` and read only the
current unit's partial `_table` → byte-identical, shard-safe, `INPUT_SCOPE`-
neutral, zero engine changes. New classes do not affect the golden baseline
unless a *default-selection* change is made; the `semantic` path is opt-in.

### Testing

- Coherence invariant: for generated rows, every `(asset_class, instrument_type)`
  and `(asset_class, risk_factor_group)` pair is in the taxonomy map (no
  cross-asset leakage).
- Determinism + shard-equivalence for the hierarchy generators.
- Identifier check digits validate (ISIN/CUSIP/SEDOL); key-variant identifiers
  unique per batch.
- Composed names reflect their source columns.
- Scaffolder maps `semantic` hints to the right generators; `validate()` rejects a
  `parent` that is ordered after its dependent.

### Effort / risk

~3–4 days (curating the vocabularies + the hierarchy/identifier logic). **Low
risk** — additive, no engine or invariant changes. The work is mostly data
curation and the `_table`-reading sample logic (the `InternalMappings`/
`DerivedColumn` pattern already proven in `postprocessors_udfs/generic_udfs.py`).

---

## Recommended sequencing & shared foundations

| Order | Feature | Why this order | Risk |
|-------|---------|----------------|------|
| 1 | **PII generators** | Fully isolated, no engine change, builds confidence | Low |
| 2 | **Null injection** | Central mask in engine; Feature 4's nullable-FK reuses the masking helper | Low |
| 3 | **Profile-from-sample** | Stats module that Feature 3 reuses; scaffold-side only | Low |
| 4 | **Quality report** | Reuses #3's stats + checks #2's null-rates and FK integrity | Low |
| 5 | **Cardinality enforcement** | Touches the sharding invariant — do last, gate with `check-distributable` | High |

**Shared building blocks to factor once:**

- A column **null-mask helper** (Feature 1) reused by Feature 4's nullable-FK.
- A **column-stats module** (`profiling.py`) reused by Feature 3's quality report.
- An **IR semantic/hint vocabulary** (`null_rate`, `semantic`, FK
  `null_rate`/`fanout`) extended consistently across `Configs`.

**Optional enabling refactor (do only if clean composition is wanted):** make
`table["post_processor"]` accept a **list** (a postprocessor chain), run in order
in `engine.py`. This would let null injection and cardinality enforcement be
standalone postprocessors alongside `KeyEnforcer` instead of being folded in. It
is the more general design but a bigger, invariant-sensitive change
(`_build_postprocessor_args`, `validate`, the dump loop, golden baseline). The
plans above deliberately avoid requiring it.

---

## Verification checklist (per feature, before merge)

- `ruff check .` and `ruff format --check .` both pass.
- `pytest tests/` green, including `test_2_safety_net.py` (golden manifest).
- Shard-equivalence (`check-distributable`) holds for any feature touching the
  generation path.
- `import datagenic` stays cheap (no new eager heavy imports).
- New public API added to the lazy `_LAZY` map in `__init__.py`.
