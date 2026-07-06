# Implementation Plan: realistic, conditional numeric fact values

Make quantitative fact columns (notional, price, rate, PV, quantity, …) come out
**realistic by default** and, where it matters, **conditioned on other columns** —
e.g. a `notional` whose magnitude/shape depends on `asset_class` (an equity notional
≠ a swap notional). This extends the automatic realism the finance pack already gives
*categoricals/identifiers* to *numbers*, and makes those numbers row-aware.

This is the numeric counterpart of the coherent-taxonomy hierarchy
(`asset_class → instrument_type`), reusing the same machinery. Scaffold-time only —
the resulting config is frozen, so generation stays byte-identical regardless.

---

## Background: what already works (no new primitive needed)

A column value can already depend on other columns, two ways:

1. **A generator reads the partial row.** The engine injects `_table` — the frame of
   columns generated *before* this one (config order) — into every generator
   (`engine._generate_columns`). `FinancialInstrumentType` already uses this to read
   `asset_class` and sample a coherent child; `InternalMappings` / `Evals` use it for
   derived columns.
2. **A post-processor reads the whole frame** (+ declared `_input_tables`) after
   generation (e.g. `amount = qty × price`).

So conditional numbers need a *generator that conditions on a parent column*, not a new
engine capability.

## The gap

`Columns` (`scaffolding.py`) maps any non-key float to `BrownianFloats(mu=1000,
sigma=100)`: wrong **magnitude** (≈1000, a notional is millions) and wrong **family**
(normal; notionals are positive and right-skewed → lognormal). Nothing conditions the
distribution on the row's category. Semantic inference (`semantics.py`) covers only
categorical/identifier semantics today.

---

## Design

### 1. A distribution-aware float generator (`datagenic/generators.py`)

Add a `distribution` parameter to the float generators (`RandomFloats` /
`BrownianFloats`), or a small `LognormalFloats`:

- `distribution="normal"` (**default → existing behaviour, byte-identical, golden-safe**)
- `distribution="lognormal"` — positive, right-skewed; parameterised by a median +
  log-σ (so config reads in real units, not log space).

All draws stay on the injected `_rng` (determinism/shardability unchanged). Encode the
per-quantity time behaviour via the existing `recall` split: a `notional` is **fixed**
across dates (stateless, no `recall` → values persist), a `price`/`pv` **evolves**
(Brownian `recall`).

### 2. A conditional (parent-aware) numeric generator (`market_finance/generators.py`)

`ConditionalFloats` (or a finance `Notional`) — the numeric twin of
`FinancialInstrumentType`:

```jsonc
{"udf_name": "ConditionalFloats",
 "args": {"by_column": "asset_class",
          "cases": {"Equity":   {"dist": "lognormal", "median": 1.0e5, "sigma": 0.8},
                    "Rates":    {"dist": "lognormal", "median": 1.0e7, "sigma": 1.0},
                    "FX":       {"dist": "lognormal", "median": 5.0e6, "sigma": 0.9}},
          "default": {"dist": "lognormal", "median": 1.0e6, "sigma": 1.0}}}
```

Per row it reads `_table[by_column]` and draws from the matching case on `_rng`
(vectorised per distinct category, like `_sample_by_parent`). Falls back to `default`
when the parent column is absent or a category is unmapped.

### 3. Quantity priors (`datagenic/market_finance/_data.py`)

Pure-data tables of plausible defaults:

- **Flat priors** per quantity semantic — `{family, location, scale, sign, time-behaviour}`:
  `notional`, `price`, `quantity`, `rate`/`yield`/`coupon`, `spread`, `pv`/`mtm`,
  `fx_rate`.
- **Conditional priors** keyed by parent category — e.g.
  `NOTIONAL_BY_ASSET_CLASS = {"Equity": …, "Rates": …, "FX": …, …}`.

These are *book-agnostic* defaults (see Accuracy ladder).

### 4. Semantic map + wiring (`semantics.py`, `scaffolding.py`)

- Add numeric entries to `SEMANTIC_GENERATORS` (`notional`, `price`, `rate`, `spread`,
  `pv`, `quantity`, `fx_rate`). A semantic that is conditional (notional) maps to
  `ConditionalFloats` with a `parent_arg = "by_column"`; flat ones map to the
  distribution-aware float generator with the flat prior.
- `build_generator(semantic, parent)` already wires a parent into the `*_column` arg —
  reuse it for `by_column`, and merge the prior's `cases`/params into `args`.
- `Configs._apply_semantic` overrides the column exactly as it does for categoricals;
  `_order_for_coherence` already guarantees the parent (`asset_class`) is generated
  **before** the dependent (`notional`), and `validate()` errors if a named parent is
  ordered later.

### 5. Inference across the three existing strategies (`semantics.py`, `semantics_agent.py`)

- **heuristic** — name patterns **gated to numeric type**: `notional|principal|face` →
  `notional`; `price|px` → `price`; `qty|quantity|size` → `quantity`;
  `rate|yield|coupon` → `rate`; `spread` → `spread`; `pv|mtm|npv` → `pv`. Hierarchy
  linking (`_link_hierarchy_parents`) extended so `notional` links to the table's
  `asset_class` column when present.
- **values** — extend `profiling.profile_frames` to (a) pick the **family** (all-positive
  + spans orders of magnitude → lognormal) and (b) **fit per group**: `group_by` the
  detected parent (`asset_class`) and fit median/σ per category → an authoritative
  `cases` map straight from the sample.
- **agent** — map a numeric column to a quantity semantic *and* propose family/range
  (and conditioning column) from name + table context — covers the long tail with no
  curated prior.

---

## Two kinds of dependence (keep them distinct)

- **Categorical conditioning** — *which* distribution a value is drawn from depends on a
  category (notional scale by asset class). → `ConditionalFloats` reading `_table`.
- **Arithmetic derivation** — a value *computed* from others (`amount = qty × price`,
  `pv` from rate + notional). → `Evals` / `InternalMappings`, or a post-processor.

Both compose: a coherent conditional `notional` can feed a derived `pv`.

---

## Constraints & invariants

- **Column order**: the dependent column must follow its parent (the engine builds
  columns top-to-bottom; a generator sees only earlier columns). Handled by the existing
  `_order_for_coherence` reorder + the `validate()` "must appear earlier" check.
- **Single post-processor slot**: on fact tables `KeyEnforcer` already owns the
  `post_processor` slot, so conditional numbers must be a **column generator** (reads
  `_table`), not a post-processor — which is also the shardable choice.
- **Determinism / shardability**: reads `_table` + draws from the per-unit `_rng` →
  batch-local, byte-identical; `check-distributable` stays green.
- **Golden baseline**: opt-in via `--infer-semantics` (default scaffold unchanged), and
  `distribution="normal"` default keeps existing float output byte-identical.

---

## Accuracy ladder (the honest part)

- **heuristic priors** — one-click *plausible* (notional ≈ millions, lognormal,
  scaled by asset class) — but a guess; an equity notional prior isn't *your* book.
- **values / profile-from-sample** — *accurate*: per-group fit from real data is the
  authoritative path for both family and the conditional `cases`.
- **agent** — flexible for columns with no curated prior.

Same three-strategy layering already shipped — now numeric and conditional. Every pick
is annotated under `_inferred` for `review-config`, so guesses are visible and editable.

---

## Phasing

1. ✅ **Distribution-aware float generator** (`distribution=`) + tests; golden-neutral
   (default normal). `RandomFloats`/`BrownianFloats` gained `distribution="normal"|
   "lognormal"`; `_draw_floats` helper.
2. ✅ **Flat quantity priors** (`market_finance._data.QUANTITY_PRIORS`) + numeric entries in
   `SEMANTIC_GENERATORS` / `build_generator` + float-gated heuristic numeric patterns.
3. ✅ **`ConditionalFloats`** (generic, in `generators.py`) + conditional priors keyed by
   `asset_class` (`market_finance._data.CONDITIONAL_QUANTITY_PRIORS`); heuristic links `notional`
   → the asset-class column. `notional` maps to `ConditionalFloats` whose `default` case
   is the flat prior, so a standalone notional stays realistic.
4. ✅ **`values` per-group fit** — `semantics.fit_distribution` (family detection +
   real-unit param recovery, round-trips with `_draw_floats`) fits flat quantities and,
   for conditional ones, a per-parent-category `cases` map (`_fit_conditional`);
   `profiling._profile_column` gains the family-aware flat fit. Fitted params override
   the curated priors via `build_generator(overrides=…)`.
5. ✅ **agent** numeric/conditional proposals — the LLM inferrer labels numeric columns
   with a quantity semantic and proposes distribution family + scale (and a conditioning
   column) via `_numeric_args`; proposals are sanitised (positive median for lognormal,
   non-negative sigma) and override the curated prior through `build_generator`.

---

## Testing

- distribution-aware generator: lognormal is positive/right-skewed; determinism;
  `distribution="normal"` byte-identical (golden unchanged).
- `ConditionalFloats`: per-row distribution matches the row's category (e.g. Equity rows
  ~1e5, Rates rows ~1e7); falls back to default when parent absent; deterministic;
  batch-local (shard-equivalent).
- inference: `notional`(float) → notional semantic, linked to `asset_class`, ordered
  after it; non-numeric `notional` (string) not matched.
- profiling: per-group fit recovers distinct per-asset-class scales from a sample.
- end-to-end: scaffold → generate → Equity vs Rates notionals differ by ~2 orders of
  magnitude, positive, coherent with `asset_class`.

---

## Risks / open questions

- **Conditioning key**: `asset_class` is the natural default, but `instrument_type`
  (swap vs equity option) discriminates more finely — support either via `by_column`.
- **Prior curation is subjective**: keep priors clearly labelled as defaults; lean on
  profiling for fidelity. Don't over-fit the curated table to one asset class.
- **Ambiguous names** (`amount`, `value`): heuristic guesses, agent does better,
  profiling is authoritative — surfaced via `_inferred` for review.
- **Lognormal in config units**: express as median + σ (real units) so a hand-editor
  reads sensible numbers, not log-space parameters.
