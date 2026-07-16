---
name: reusing-existing-code-in-datagenic
description: Use when you have existing Python data-generation code (pandas/polars/numpy functions or classes) and want to reuse it inside datagenic as a column generator or post-processor. Covers code that returns arrays, lists, one DataFrame, a list/dict of DataFrames, or arbitrary objects, and porting it to be deterministic (uses np.random.seed / uuid / datetime.now / global state, or needs to become shardable).
---

# Reusing Existing Code in datagenic

## Overview

datagenic has exactly **two plug-in points**, both just a class with a `call()` method
referenced from config by import path. Reusing existing code = (1) wrap it in the right
shape, (2) route its randomness through the injected `_rng`, (3) wire it in config. Nothing
in the datagenic package changes.

**The core invariant you must protect:** same seed + config → *byte-identical* output,
whether run sequentially, on N cores (`--parallel`), or sharded (`--shard`). Most porting
work is making the code honor this.

## Choose the plug point by OUTPUT SHAPE (not "function vs class")

| Legacy code returns | Plug point | Returns |
|---|---|---|
| one column: numpy array / list / Series / one-value-per-row | **column generator** | array/list/Series (normalized to a Polars Series) |
| one table: a DataFrame | **post-processor** | one `pl.DataFrame` (pandas coerced) |
| **many tables: list/dict of DataFrames** | **fan-out** (§Complex outputs) | one datagenic *table per output*, each a post-processor returning its slice |
| **arbitrary object** (nested dict, model, …) | **reducer** (§Complex outputs) | a `pl.DataFrame` — datagenic only persists tabular data |

Most "I wrote a function that builds a DataFrame" maps to a **post-processor**; per-field
value factories map to **column generators**.

## The contract

UDF classes are instantiated with **no constructor args** — every parameter arrives as a
config arg or a framework kwarg. Always `**kwargs`; name only what you use.

```python
# column generator — one column
class CustomerIds:
    def call(self, _number_of_rows_per_batch, _batch_number, prefix="CUST-", **kwargs):
        start = (_batch_number - 1) * _number_of_rows_per_batch   # batch-disjoint → unique
        idx = range(start, start + _number_of_rows_per_batch)
        return [f"{prefix}{i:08x}" for i in idx]

# post-processor — one table
class OrdersBuilder:
    INPUT_SCOPE = "reference"   # reads a broadcast dimension table; see §Shardability
    def call(self, _table, _input_tables, _rng, _date, _number_of_rows_per_batch, **kwargs):
        cat = _input_tables["catalog"]                 # current-batch slice, already Polars
        n = _number_of_rows_per_batch
        row = _rng.integers(0, cat.height, n)          # sample a catalog ROW (price stays aligned)
        qty = _rng.poisson(3, n) + 1
        total = cat["price"].to_numpy()[row] * qty * (1 + _rng.normal(0, 0.05, n))
        return pl.DataFrame({
            "order_id": [f"ORD-{i:012d}" for i in range(n)],
            "created_at": [_date] * n,                  # the asof date — never datetime.now()
            "product": cat["product"].to_numpy()[row], "quantity": qty, "total_price": total,
        })
```

Framework kwargs (both): `_rng` (seeded `numpy.random.Generator` — **use for all randomness**),
`_seed`, `_date`, `_batch_number` (1-based), `_number_of_rows_per_batch`,
`_start_date`/`_end_date`/`_previous_date`, `_number_of_batches`. Post-processors also get
`_table`, `_input_tables`, `_unicity_keys`. Column generators also get `_global_seed` and
`input_array` (in `recall`). **`_global_seed` is injected into column generators only, not
post-processors** — this matters for fan-out (§Complex outputs).

## The determinism port — the real work

Go through this checklist against the legacy code; each item is a byte-identical-output bug:

1. **Randomness → `_rng` only.** Delete `np.random.seed(...)`, `np.random.<fn>` (global
   state), `random`, `Faker` default seeding. Draw from the injected `_rng`. (Seeding the
   numpy global leaks across every other `(table, date, batch)` unit and breaks shards.)
2. **`uuid` / auto-increment ids → deterministic ids.** `uuid4()` is neither reproducible
   nor seed-tied. Use a **batch-disjoint** integer window `(_batch_number-1)*n … +n` rendered
   as a string (reproducible *and* globally unique across batches/shards), or draw from `_rng`.
3. **Wall-clock → `_date`.** Replace `datetime.now()` / `time.time()` with the injected
   `_date`.
4. **No cross-batch / global state.** No module-level accumulators, no reading other batches'
   output files, no state on `self` between calls. Each unit must be independent.
5. **Positional data args → framework kwargs.** `f(n, catalog)` becomes: size from
   `_number_of_rows_per_batch`; upstream tables via declared `_input_tables` (a batch-local
   slice), not a positional argument.
6. **Time evolution → `recall`.** If a value must persist/drift across dates, keep first-date
   logic in `call` and add `recall` (same signature) that rolls the prior frame forward —
   don't hold state on the instance.
7. **Return Polars** (pandas is coerced, but Polars is native).

## Complex outputs

### Function returns numpy array / list — no work
A column generator's `call` may return a numpy array, Python list, or Series; all are
normalized. Just route randomness through `_rng`.

### Class returns a list/dict of DataFrames → fan-out
There is no plug point that emits several tables at once. Split into **one datagenic table per
output frame**, each a post-processor returning just its slice, all driven by the *same*
producer. Determinism requires every slice to see the **identical** producer output, so the
producer must be seeded **independently of the table name** — but post-processors don't get
`_global_seed`, so pass a shared seed constant through config (`producer_seed`) and combine it
with the injected `_date` / `_batch_number`:

```python
import hashlib, functools
import numpy as np, polars as pl
from legacy_gen import MarketProducer            # .build(n, rng) -> {"instruments": df, "quotes": df}

def _producer_rng(producer_seed, date, batch):
    key = f"{producer_seed}|{date}|{batch}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

@functools.lru_cache(maxsize=16)                 # pure-function cache: keyed on ALL deterministic inputs
def _run(producer_seed, date, batch, n):
    return MarketProducer().build(n, _producer_rng(producer_seed, date, batch))

class ProducerSlice:                             # same class for every output table
    INPUT_SCOPE = "batch"
    def call(self, _date, _batch_number, _number_of_rows_per_batch,
             producer_seed, output_key, **kwargs):
        frames = _run(producer_seed, _date, _batch_number, _number_of_rows_per_batch)
        df = frames[output_key]
        return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
```

Config: one table per output, same `udf_name`, differing `output_key`, **shared
`producer_seed`**:
```json
{ "table_name": "instruments", "columns": [], "post_processor": {
    "udf_module": "adapters", "udf_name": "ProducerSlice",
    "udf_args": { "producer_seed": 20240101, "output_key": "instruments" } } }
{ "table_name": "quotes", "columns": [], "post_processor": {
    "udf_module": "adapters", "udf_name": "ProducerSlice",
    "udf_args": { "producer_seed": 20240101, "output_key": "quotes" } } }
```
The `lru_cache` is a *pure-function* cache (its key is the full set of deterministic inputs),
so it never breaks determinism — one worker computes the producer once and every slice reuses
it; other workers/shards recompute identically. Omit it and each slice recomputes — still
correct, just slower. If the outputs share one schema (mere partitions), prefer a single
table over fan-out.

### Class returns an arbitrary object → reducer
datagenic persists parquet, so tabular data only. Add a projection step in `call` that pulls
the columns you want out of the object and returns a `pl.DataFrame`. Keep the extraction pure
(no I/O, no clock, randomness from `_rng`).

## Wire it in config

Module must be importable (on `PYTHONPATH`).

```json
"column_generator": { "udf_module": "mypkg.gen", "udf_name": "CustomerIds", "args": { "prefix": "CUST-" } }
"post_processor":   { "udf_module": "mypkg.gen", "udf_name": "OrdersBuilder",
                      "udf_args": { "_input_tables": ["catalog"] } }
```
A post-processor that fabricates the whole table sits on a columns-less table (`"columns": []`).
Reference/dimension tables it reads are typically `single_batch: true, single_date: true`
(generated once and broadcast).

## Shardability

Declare `INPUT_SCOPE`: `"batch"` (default, own batch only), `"reference"` (also reads small
broadcast tables), `"global"` (whole-table/cross-batch — **rejected** for sharded runs). Then
**prove it**:

```bash
datagenic check-distributable config.json
```
Static `INPUT_SCOPE` gate + determinism probe (perturbs the numpy global RNG; output must not
change → confirms no ambient randomness leaked) + full-run-vs-2-shards byte equality. This is
the fastest way to catch a bad port.

## Common mistakes

| Mistake | Fix |
|---|---|
| Kept `np.random.seed(42)` / used `np.random.<fn>` | Draw from injected `_rng` |
| `uuid4()` / `datetime.now()` for ids/timestamps | Batch-disjoint integer ids; `_date` for time |
| Wrote a bare function | Wrap in a class with `call` (+ `recall` if time-evolving) |
| Passed constructor args to the UDF | Instantiated with no args — pass via config `udf_args`/`args` |
| Fan-out slices seeded from `_rng`/`_seed` | Table-dependent → slices diverge; use shared `producer_seed` + `_date`/`_batch_number` |
| Expected `_global_seed` in a post-processor | Not injected there (column generators only) |
| Returned a dict/list of DataFrames from one post-processor | One table per output (fan-out) |
| Sized output to a hardcoded `n` | Use `_number_of_rows_per_batch` |
| Skipped verification | Run `datagenic check-distributable` |
