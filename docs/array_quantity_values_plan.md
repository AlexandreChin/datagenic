# Implementation Plan: realistic array-type quantity values (e.g. PnL time series)

Make **array-typed** quantitative columns (a per-row `pnl` vector, a rate `curve`, a
returns `series`) come out realistic by default and, where it matters, **conditioned on
another column** — the array counterpart of the scalar work already shipped
(`RandomFloats`/`BrownianFloats` distributions, `ConditionalFloats`, fit-from-sample,
agent proposals). Same machinery: a generator that reads the partial `_table`,
scaffold-time inference across heuristic/values/agent, `_inferred` annotations.
Scaffold-time only — the resulting config is frozen, so generation stays byte-identical.

---

## Background: what already works

- **Array generators exist.** `BrownianArrayFloats` emits a fixed-size float array per row
  and evolves it across dates via `recall`; also `RandomArrayFloats`,
  `RandomMultivariateArrayFloats`, array-date generators (`generators.py`).
- **Type-based scaffolding wires them.** An IR column typed `array(float)` →
  `Columns` picks `BrownianArrayFloats` (`size_of_arrays=100, mu=1000, sigma=100`,
  normal). So an array column gets an array generator — but with generic, un-tuned params.
- **Determinism reshape identity.** `rng.normal(size=(rows, k))` draws `rows*k` values in
  C-order and reshapes; it is byte-identical to `rng.normal(size=rows*k).reshape(rows, k)`.
  So routing array draws through the scalar `_draw_floats(rng, rows*k, …)` + reshape keeps
  the existing `normal` output **byte-for-byte** (golden-neutral).

## The gap

1. **No domain/semantic coverage for arrays.** Every `SEMANTIC_GENERATORS` entry (finance
   categoricals/identifiers + the numeric quantities) is **scalar**; nothing maps to an
   array generator, so `--infer-semantics` never detects a `pnl` time series or tunes it.
2. **Array generators missed Phases 1–5.** No `distribution` arg (no lognormal), no
   conditioning (`ConditionalFloats` is scalar-only), no fit-from-sample.
3. **Profiling can't type arrays.** `profile_frames._ir_type` falls back to `"string"` for
   a list dtype; `fit_distribution` only runs on scalar floats.
4. **Latent mismatch.** The scalar numeric gate uses `"float" in type`, which is also true
   for `"array(float)"` — an `array(float)` named like a scalar quantity could be mis-wired
   to a scalar generator. Must be fixed.

---

## Design

### 1. Distribution-aware array generators (`datagenic/generators.py`)

Add `distribution: str = "normal"` to `RandomArrayFloats` and `BrownianArrayFloats`,
routing the draw through the existing `_draw_floats` helper then reshaping:

```python
flat = _draw_floats(_rng_from(kwargs), _number_of_rows_per_batch * size_of_arrays, mu, sigma, distribution)
return np.asarray(flat).reshape(_number_of_rows_per_batch, size_of_arrays).tolist()
```

- `"normal"` (default) is byte-identical to today (reshape identity above → golden-safe).
- `"lognormal"` — positive, right-skewed elements (`mu` = median in real units).
- **Optional intra-array path** (`path="iid" | "walk"`): `"iid"` (default = today, each
  element independent); `"walk"` makes the array a **cumulative** series along its own axis
  (`base + cumsum(shocks)`) — a genuine per-row time path, the realism lever for a PnL/price
  series. Kept opt-in so the default stays byte-identical.
- `BrownianArrayFloats.recall` (cross-date evolution) unchanged in default behaviour; note
  the **two time axes** distinction below.

### 2. Conditional array generator (`datagenic/generators.py`)

`ConditionalArrayFloats` — the array twin of `ConditionalFloats`: reads `_table[by_column]`
and, per row, draws an array from the matching `cases` entry
(`{value: {distribution, mu, sigma, size_of_arrays}}`), falling back to `default`. Same
vectorised-per-distinct-parent, `_rng`-only, batch-local pattern (shard-safe).

### 3. Array quantity priors (`datagenic/_finance_data.py`)

`ARRAY_QUANTITY_PRIORS` (flat) and `CONDITIONAL_ARRAY_QUANTITY_PRIORS` (keyed by parent
category), shaped like the generator args — `{distribution, mu, sigma, size_of_arrays,
path?}`. E.g. a `pnl` vector: normal, mean 0, a book-plausible daily-PnL sigma, `size` = a
default window (e.g. 250 business days); a positive `curve`: lognormal.

### 4. Semantic map + wiring (`semantics.py`, `scaffolding.py`)

- Add array-quantity semantics (`pnl`, `pnl_vector`, `curve`, `series`, `returns`) to
  `SEMANTIC_GENERATORS` → the array generators, base_args from the priors; conditional ones
  → `ConditionalArrayFloats` with `parent_arg = "by_column"`.
- **Type gating (critical fix).** Introduce `_is_float_array_type(t)` (`"array(float)"`) and
  tighten `_is_float_type` to exclude array types, so scalar semantics never grab an array
  column and vice-versa. `_match_by_type` dispatches: string → categorical; `float` (scalar)
  → scalar quantity; `array(float)` → array quantity.
- `build_generator` already merges `base_args` + parent + `overrides`; array size/params flow
  through unchanged. `_order_for_coherence` already orders a conditional child after its parent.

### 5. Inference across the three strategies

- **heuristic** — array-quantity name patterns (`pnl`, `pnl_vector`, `curve`, `series`,
  `returns`), gated to `array(float)`. A conditional one (`pnl`) links to the table's
  `asset_class` column via the generalised `_link_hierarchy_parents`.
- **values / profiling** — teach `profile_frames._ir_type` to map polars `List(Float*)` →
  `array(float)` (and `List(Int*)` → `array(int)`); extend fitting to arrays: flatten the
  per-row arrays to fit element `distribution`+`mu`+`sigma` (reuse `fit_distribution`), and
  set `size_of_arrays` from the observed (modal/median) array length. Per-group fit when a
  parent category is present → an authoritative `cases` map.
- **agent** — propose an array semantic + element family/scale + `size_of_arrays` (and a
  conditioning column) from name + table context; sanitise like the scalar `_numeric_args`.

---

## Two "time" axes (keep them distinct)

- **Intra-array axis** — the array *is* a series (daily PnL over a window). Realism lives in
  `path="walk"` (cumulative) vs `"iid"`, element family, and length.
- **Cross-date axis** — the same row's array **evolving across generation dates** via
  `recall` (today: additive shocks to the whole vector). Independent of the intra-array axis.

A PnL column typically wants a `walk` intra-array path; whether it also drifts across dates
is a separate `recall` choice. Document both so a user tunes the right one.

---

## Constraints & invariants

- **Determinism / shardability** — all draws on the per-unit `_rng`; reshape/cumsum are
  deterministic; `ConditionalArrayFloats` reads `_table` + draws batch-local → byte-identical,
  `check-distributable` stays green.
- **Golden baseline** — opt-in via `--infer-semantics`; `distribution="normal"` + `path="iid"`
  defaults keep existing array output byte-identical (reshape identity).
- **Single post-processor slot** — conditional arrays must be a **column generator** (reads
  `_table`), not a post-processor (`KeyEnforcer` owns the slot on fact tables).
- **Type-gate correctness** — scalar and array numeric semantics must be mutually exclusive
  (the `_is_float_type` fix); add a regression test that a scalar quantity name on an
  `array(float)` column is **not** wired to a scalar generator, and vice-versa.

## Accuracy ladder

- **heuristic** — one-click plausible (a PnL vector: normal, mean 0, book-plausible sigma,
  250-day window) — a guess.
- **values / profile** — accurate: element family/params + array length fitted from a real
  sample (per-group for conditional).
- **agent** — flexible for array columns with no curated prior.

Every pick annotated under `_inferred` for `review-config`.

---

## Phasing

1. ✅ **Distribution-aware array generators** — `RandomArrayFloats`/`BrownianArrayFloats`
   gained `distribution` (`normal` default / `lognormal`) and `path` (`iid` default /
   `walk`, a cumulative per-row series) via `_draw_float_array`; golden-neutral (reshape
   identity keeps `normal`+`iid` byte-identical).
2. ✅ **Flat array-quantity priors** (`ARRAY_QUANTITY_PRIORS`: `pnl_series`, `returns`,
   `curve`) + array semantics in `SEMANTIC_GENERATORS`; array-gated heuristic patterns; the
   **scalar/array type-gate fix** (`_is_float_array_type` + tightened `_is_float_type`, so
   `pnl`/float → scalar signed `pnl` and `pnl`/array(float) → `pnl_series`, never crossing).
   Added a scalar signed `pnl` prior (distinct from the positive `pv`).
3. ✅ **`ConditionalArrayFloats`** (generic, in `generators.py`) + conditional array priors
   keyed by `asset_class` (`CONDITIONAL_ARRAY_QUANTITY_PRIORS`). `pnl_series` moved from flat
   to conditional (per-asset-class PnL volatility); `_link_hierarchy_parents` generalised to
   link conditional *array* children to `asset_class` too; `size_of_arrays`/`path` shared
   across cases, `default` used for absent/unmapped parents.
4. ✅ **profiling/values**: `profiling._ir_type` types a polars `List(Float)` column as
   `array(float)`; `fit_array_distribution` fits element family/params + `size_of_arrays`
   from the sample, and `_fit_conditional_array` fits per-parent-category `cases`. Wired into
   `_profile_column` (flat, no-inference path) and `values_inferrer._fit_numeric` (flat +
   conditional). FK-candidate detection skips array columns (unhashable).
5. ✅ **agent** array proposals — the LLM inferrer labels an array(float) column with an
   array semantic and proposes element distribution/mu/sigma **plus `size_of_arrays`** (and a
   conditioning column); `_numeric_args` maps it to overrides (flat spec / `{default}` for
   conditional, both carrying `size_of_arrays`) and sanitises invalid proposals.

---

## Testing

- Array generator: `distribution="normal"` byte-identical to today (reshape identity, golden
  unchanged); lognormal elements positive/right-skewed; `path="walk"` monotone-in-expectation
  cumulative vs `"iid"`; determinism.
- `ConditionalArrayFloats`: per-row array scale matches the row's category; fallback to
  `default`; deterministic; batch-local (shard-equivalent).
- Type gate: `pnl`(`array(float)`) → array generator; a scalar quantity name on `array(float)`
  is **not** wired to a scalar generator, and `notional`(`float`) is **not** wired to an array
  generator.
- Inference: `pnl` linked to `asset_class`, ordered after it. profiling recovers per-group
  element scales + array length from a sample.
- End-to-end: scaffold → generate → each row's `pnl` is an array of the fitted length, values
  coherent with the row's asset class, positive-vs-signed per family.

---

## Risks / open questions

- **What "realistic" means for a series.** `iid` elements aren't a real path; `path="walk"`
  gives a cumulative walk but still no vol-clustering/autocorrelation. Decide how far to model
  (a simple walk may be enough for a fixture; GARCH-like dynamics are out of scope).
- **Ragged arrays.** Real samples may have variable-length arrays; scaffold a fixed
  `size_of_arrays` (modal length) and `log()` the choice rather than silently truncating.
- **recall + lognormal.** Additive cross-date shocks to a positive (lognormal) array can go
  negative; use multiplicative shocks for positive families, or restrict cross-date evolution
  to `normal`.
- **Naming ambiguity.** `pnl` vs `pnl_scalar` — the type gate resolves scalar-vs-array, but a
  column named `pnl` typed `float` is a scalar quantity (add a scalar `pnl` prior too, or map
  it to `pv`).
