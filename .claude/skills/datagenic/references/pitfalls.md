# Pitfalls Reference

## Determinism & batch isolation

| Symptom | Root cause | Fix |
|---|---|---|
| `check-distributable` fails the determinism probe | A generator/post-processor used `numpy.random.*`, `random`, `uuid`, or `time` instead of the injected `_rng` | Draw all randomness from `_rng` (seeded `default_rng(_seed)`) |
| Union of shards ≠ single run | Post-processor keeps cross-batch state or reads other batches' files | Make it batch-local; only read declared `_input_tables` (current-batch slice) |
| `validate(distributed=True)` rejects a sharded table | Its post-processor declares `INPUT_SCOPE="global"` | Make it batch-independent, or mark the reference table `single_batch` (broadcast) |
| Golden test fails after an *unintended* change | A determinism regression | Investigate — don't just re-run `capture_baseline.py` |

## Config & the IR

| Symptom | Root cause | Fix |
|---|---|---|
| `kind` / `table_type` ignored or errors | `kind` was renamed to `type`; `table_type` was removed (inert) | Use `type: "fact" \| "referential"` in the IR; drop `table_type` |
| A table's `foreign_keys` are ignored | The table isn't `type: "fact"` — FK enforcement is only wired for fact tables | Set `type: "fact"`, or move the FK to the fact side |
| A fact table comes out tiny / a reference table huge | Row-count scale follows `type` (fact → `number_of_rows_in_fact_tables`, referential → `..._non_fact_tables`) | Set the correct `type`, or override `number_of_rows` per table |
| `ValueError: type must be 'fact' or 'referential'` | Typo in the IR `type` value | Fix the value; omit `type` to default to `referential` |
| `udf_args._input_tables` table is empty / missing | Declared table appears **later** in the config, or isn't declared | List dependencies in `udf_args._input_tables` and place them earlier |
| A `_`-prefixed `udf_args` key is rejected by `validate()` | Leading `_` is reserved for framework args | Rename the key (only `_input_tables` is allowed) |
| `validate()` rejects `null_spec` on a column | `null_spec` is on a unicity key, or `rate` is outside [0, 1] | Move it to a non-key column; keep `rate` in [0, 1] (nullable FKs → a `KeyEnforcer` rule) |
| `null_spec` nulls produce no nulls / wrong rows | `rate` ≤ 0, or `when` references a column generated *after* this one (not yet in `_table`) | Set `rate` in (0, 1]; reference only earlier columns in `when` (reorder if needed) |
| A column-level `null_spec` on an FK column is overwritten | FK enforcement (`KeyEnforcer`) runs after generation and resamples the column | Express nullable FKs via the FK `null_rate` hint, not `null_spec` |
| FK `null_rate` / `fanout` has no effect | The hint is on a unicity-key FK column (ignored — nulls/repeats would break dedup), or on a `referential` table's FK (only fact tables get `KeyEnforcer`) | Put it on a non-key FK column of a `fact` table |
| `fanout` doesn't give a hard global min/max | It's a per-batch (distributional) guarantee — across batches counts sum | Expected; for a true global bound make the parent/child a `single_batch` table |

## Domain generators & semantic inference

| Symptom | Root cause | Fix |
|---|---|---|
| `instrument_type` / `risk_factor_group` not coherent with `asset_class` | The parent column is generated *after* the child (it reads only earlier columns) | List `asset_class` before its dependents; with inference the scaffolder reorders, and `args.asset_class_column` must name the parent |
| `--infer-semantics agent` raises ImportError | The `anthropic` SDK isn't installed | `pip install 'datagenic[agent]'` (+ `ANTHROPIC_API_KEY`), or use `heuristic` / `values` |
| `--infer-semantics values` infers nothing | No `samples` provided (or no matching `<table>.parquet`/`.csv` in the directory) | Pass `--samples <dir>` / `samples=`; it falls back to name heuristics per column otherwise |
| Inference picked the wrong generator | Heuristic/agent guess; the config is the source of truth | `datagenic review-config` to spot low-confidence picks, then hand-edit `udf_name`/`args`, or set an explicit `"semantic"` on the IR column |
| A finance-categorical unicity key collapses the row count | A small fixed domain can't fill a high-cardinality key | Pair it with an id/incremental key (`keys: ["trade_id", "asset_class"]`); the generator now declares `cardinality()` so the engine budgets the id correctly — a lone categorical key warns with its achievable ceiling |
| A finance categorical ignores a large `cardinality` | Fixed-vocabulary generators cap `cardinality` at their domain (no fabricated values) | Expected — to widen the domain, supply `list_of_values`, or pick a higher-cardinality generator |

## Post-processor authoring

| Symptom | Root cause | Fix |
|---|---|---|
| `_input_tables` is the wrong shape / `_tables` is undefined | The injected dict is `_input_tables` (not `_tables`, which was the old name) | Use `_input_tables` |
| Post-processor receives unexpected kwargs | All `postprocessor_args` + your `udf_args` are injected | Absorb with `**kwargs`; name only what you use |
| `validate()` reports a missing required arg | The class declares `REQUIRED_ARGS` and the config omits one | Add the arg to `udf_args`, or drop it from `REQUIRED_ARGS` |
| Referencing a table before it's produced raises | `_result` holds an `_Ungenerated` sentinel until the table runs | Declare it in `_input_tables` and order it earlier |

## Distribution / parallel

| Symptom | Root cause | Fix |
|---|---|---|
| Deadlock / hang under `--parallel` | `fork` start method with numpy/polars threads | datagenic forces `forkserver`/`spawn`; don't re-enable `fork` |
| Re-run regenerates everything instead of resuming | The config (or seed/args) changed → new config hash | Intentional; only relocating output or extending dates preserves reuse |
| Re-run reuses stale data after editing a generator | (shouldn't happen) the data spec change must change the hash | Confirm the edit is inside the content-config, not an IO/location field |

## Environment

| Symptom | Root cause | Fix |
|---|---|---|
| `numpy 2.5.0 cannot be used` / mypy stub syntax error on Python 3.11 | datagenic requires **Python ≥ 3.12** (the locked numpy needs it) | Use Python 3.12+ |
| Cloud backend can't find credentials | `Storages` loads a `.env` via `load_dotenv()` and reads vars with `os.getenv` | Put creds in `.env` (or real env vars); see `docs/cookbook_storages.md` |
| `import datagenic` pulls in a cloud SDK | Something eagerly imported a backend | Keep `__init__` lazy; cloud SDKs import per `storage_type` only |
