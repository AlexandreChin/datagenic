# Post-processors & Custom Generators Reference

A post-processor replaces or augments a table's column-generated data. Common uses: load a
reference file, join/merge declared input tables, enforce keys (`KeyEnforcer`), apply business
rules. Built-ins live in `datagenic.postprocessors` (`KeyEnforcer`, `TemplatePostProcessor`);
dev/sample ones in `postprocessors_udfs.generic_udfs`.

## Contract

```python
class MyPostProcessor:
    INPUT_SCOPE = "batch"          # optional: "batch" (default) | "reference" | "global"
    REQUIRED_ARGS = ["my_param"]   # optional: validate() rejects configs missing these udf_args

    def call(self, _table, _input_tables: dict, **kwargs): ...     # first date
    def recall(self, _table, _input_tables: dict, **kwargs): ...   # later dates (optional)
```

- `_table` — this table's column-generated rows (empty frame if the table has no columns).
- `_input_tables` — dict of the tables you declared in `udf_args._input_tables`, resolved to the
  **current batch's slice**. Exactly those, `{}` if none declared.
- Return a `pl.DataFrame` (preferred) or `pd.DataFrame` (coerced at the write boundary).
- Accept `**kwargs` so new framework args never break you. Name only what you use.

Wire it in the config:

```json
"post_processor": {
  "udf_module": "mypkg.postprocessors",
  "udf_name": "MyPostProcessor",
  "udf_args": { "_input_tables": ["products"], "my_param": 42 }
}
```

## Framework-injected kwargs

| Kwarg | Type | Description |
|---|---|---|
| `_table` | frame | This table's column-generated rows |
| `_input_tables` | `dict[str, frame]` | The declared upstream tables (current batch slice) |
| `_unicity_keys` | `list[str]` | This table's unicity-key columns |
| `_date` | `str` | Current business date |
| `_previous_date` | `str` | Previous business date (`_start_date` on the first date) |
| `_start_date` / `_end_date` | `str` | Config date range |
| `_batch_number` | `int` | Current batch (**1-based**, offset applied) |
| `_number_of_batches` | `int` | Total batches |
| `_number_of_rows_per_batch` | `int` | `ceil(number_of_rows / number_of_batches)` for this table |
| `_seed` | `int` | Deterministic per-unit seed (`sha256(seed \| table:post \| date \| batch)`) |
| `_rng` | `numpy.random.Generator` | Seeded `default_rng(_seed)` — **use this for all randomness** |

All `default_parameters.postprocessor_args` keys are also injected (minus `_input_tables`), and
all your own `udf_args` keys (minus `_input_tables`).

## Using the technical arguments

- **`call` vs `recall`** — `call` runs on the first date; if you define `recall` (same
  signature) it runs on every later date. Use the pair for time-evolution: read `_previous_date`
  / the prior frame and roll values forward so a series drifts day to day.
- **dates** — only introduce new entities on `_start_date`, carry them forward otherwise.
- **`_batch_number` / `_number_of_batches`** — batch-aware sizing only; never state that must
  persist across batches (see Batch isolation).
- **`_number_of_rows_per_batch`** — size any rows you fabricate to match the batch.
- **`_seed` / `_rng`** — reproducible randomness; draw everything from `_rng`.

```python
import polars as pl

class DriftPrices:
    """Seed a price per row, then nudge it a little each day (reproducibly)."""
    def call(self, _table, _rng, _number_of_rows_per_batch, **kwargs):
        return _table.with_columns(price=pl.Series(_rng.normal(100, 5, _number_of_rows_per_batch)))

    def recall(self, _table, _rng, **kwargs):
        prev = _table["price"].to_numpy()
        return _table.with_columns(price=pl.Series(prev * (1 + _rng.normal(0, 0.01, len(prev)))))
```

## Batch isolation & distributability

Byte-identical output across sequential / `--parallel` / `--shard` runs requires each
`(table, date, batch)` unit to be independent:

1. **Randomness only from `_rng`.** Never `numpy.random.*` global state, `random`, `uuid`, or
   `time`.
2. **No cross-batch state.** No module globals accumulated across calls; don't read other
   batches' output files from disk.
3. **Reading `_input_tables` is fine** — the framework injects only the current batch's slice,
   so a join against a declared input table is batch-local.

Declare dependency scope so the gate can reason about it:

| `INPUT_SCOPE` | Meaning | Shardable? |
|---|---|---|
| `"batch"` (default) | depends only on its own batch | ✅ |
| `"reference"` | also reads small reference tables every worker regenerates identically | ✅ |
| `"global"` | whole-table / cross-batch state | ❌ — `validate(distributed=True)` rejects it for a sharded table |

If a table genuinely needs whole-table data, use a `single_batch` reference table (generated
once and broadcast), not `INPUT_SCOPE="global"`. Verify before distributing:

```bash
datagenic check-distributable config.json
```

(static `INPUT_SCOPE` gate + determinism probe under a perturbed global RNG + full-vs-2-shards
byte-equality check).

## `KeyEnforcer` (built-in)

Wired automatically onto fact tables when scaffolding with FK enforcement. Rule kinds:
`{"col","copy"}` (copy another column), `{"col","literal"}`, `{"col","from":"Table.col"}`
(sample from a pool; intersection if a list), `{"cols":[...],"from":"Table"}` (copy whole key
tuples), and `+ "cross"` for the cartesian key space. Duplicate rows are dropped on the unicity
keys afterward. Reference-pool tables are typically `single_batch`.

## Custom column generators

A generator is a class with `call(self, _number_of_rows_per_batch, **kwargs)` returning a
column (numpy array, list, or pandas Series — normalised to a Polars Series). Implement
`recall(self, input_array, **kwargs)` for time-propagated columns. Reference it per column via
`column_generator.udf_module`/`udf_name`/`args`.

```python
# mypkg/my_generators.py
import numpy as np

class MyGenerator:
    def call(self, prefix, _number_of_rows_per_batch, _rng=None, **kwargs):
        rng = _rng or np.random.default_rng()
        return np.char.add(prefix, rng.integers(0, 1000, _number_of_rows_per_batch).astype(str))
```

**Faker** maps onto this pattern (one provider value per row). Caveats: Faker has its own seed
(`Faker.seed(n)`) separate from numpy — for reproducibility pass `_seed` and seed it explicitly,
or draw indices from `_rng`; it's a Python loop, so fine for small reference tables, not
million-row fact tables; for columns that must be stable across dates, implement `recall`.
