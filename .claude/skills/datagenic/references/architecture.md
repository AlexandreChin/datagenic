# Architecture Reference

## The two-stage pipeline

Everything flows through **scaffold → generate**.

1. **Scaffold** (`datagenic/scaffolding.py`, `Configs`): IR → a full generation config JSON.
   `Configs` picks a default column generator per `(type, is_unicity_key, n_keys)` via the
   `Columns` class, wires foreign-key enforcement (`KeyEnforcer`) onto fact tables, and
   optionally attaches a no-op `TemplatePostProcessor` to every table. Output is dumped with
   `.dump(file_path=...)`.
2. **Generate** (`datagenic/engine.py`, `Generator`): config → data files.
   `Generator(config).generate()` runs the nested loop **batch → date → table → columns →
   post-processor** and writes parquet (or csv/avro) to
   `output_folder_path/{tables_prefix}{table}/{date}/{table}-{batch}.parquet`.

The **IR** is the native input — a flat
`{name, tables:[{type, name, keys, columns:[{name, type}], foreign_keys}]}`. To scaffold from
another schema, pass `adaptor=fn` where `fn(source) -> IR` (a small one-time mapper). The IR
carries **no generators**; generators are assigned during scaffolding. See `config.md`.

## How a table is built, per `(date, batch)` unit

For each unit the engine:

1. Resolves the per-batch row count (`_number_of_rows_per_batch`), honouring a dynamic
   `number_of_rows: {from_table, multiplier}`.
2. **Column generation** — iterates `table.columns` in config order; for each, instantiates
   the generator and calls `generator.call(_number_of_rows_per_batch, **args, **framework_kwargs)`
   on the first date, `recall(...)` on later dates if defined. Results assemble into a Polars
   DataFrame.
3. **Post-processor** — if `table.post_processor` is set, calls `call(_table, _input_tables, **udf_args, **framework_kwargs)`
   on the first date and `recall(...)` on later dates; the return value replaces the table's data.

`_result` (a dict, one entry per table) holds the current unit's frame for each table; each
table is seeded with an `_Ungenerated` sentinel, so referencing a table **before it is produced
raises** instead of silently yielding an empty frame. `_input_tables` is built from the
`udf_args._input_tables` list a post-processor declares (resolved to the current batch's slice).

**Framework-injected kwargs** (every `call`/`recall`, leading `_`): `_table`, `_input_tables`,
`_unicity_keys`, `_date`, `_previous_date`, `_start_date`, `_end_date`, `_batch_number`
(1-based), `_number_of_batches`, `_number_of_rows_per_batch`, `_seed`, `_rng`. User `udf_args`
arrive as plain kwargs alongside them; a `_`-prefixed config key is reserved and rejected by
validation (except `_input_tables`).

## Determinism model (the core invariant)

- **Per-unit seeded RNG.** Each `(table, date, batch)` unit gets its own
  `numpy.random.default_rng` seeded from `sha256(seed | table | date | batch)`
  (post-processors use `sha256(seed | table:post | date | batch)`). Generators and
  post-processors must draw from the **injected `_rng`** — never `numpy.random.*` global state,
  `random`, `uuid`, or `time`. An extra draw in one place then cannot perturb any other unit.
- **Config is read-only.** `Generator` keeps a pristine deep copy and rebuilds a fresh working
  config at construction *and* at the start of every `generate()`. Repeated `generate()` calls,
  and reusing one config dict across instances, are byte-identical. Don't mutate the caller's dict.
- **Distribution is master-free.** Sharding splits the batch space into disjoint slices
  (`--shard i --of N`, 1-based) injected as `batch_range` into a *copy* of the config; seeds are
  untouched, so the union of all shards equals a single run byte-for-byte. `--parallel N` does the
  same across local processes (using `forkserver`/`spawn`, never `fork`).
- **Shardability gate.** A post-processor declares `INPUT_SCOPE` = `"batch"` (default) /
  `"reference"` / `"global"`. `validate(distributed=True)` statically rejects a sharded table
  whose post-processor declares `"global"`. `datagenic check-distributable` proves it
  dynamically: a determinism probe (regenerate under a perturbed global RNG, require identical
  output) plus a sample full-run-vs-2-shards byte-equality check.

## Checkpointing & resume

Every dumped parquet embeds key-value metadata (`datagenic_config_hash`, `datagenic_seed`,
`datagenic_end_date`). On re-run a unit is skipped (its frame restored) only if a matching
parquet already exists with the same config hash. The hash deliberately **excludes IO/location
fields** (output path, storage type, format, log paths, batch offset, `end_date`) so the same
data written elsewhere — or with an extended date range — stays valid. Works for cloud output
too: the storage layer validates via the parquet footer without downloading the whole object.
`generate(force_regenerate=True)` bypasses; `generate(extend=True)` requires existing data.

## Module map (`datagenic/`)

- `__init__.py` — lazy (PEP 562) public API re-exports; keep `import datagenic` cheap.
- `scaffolding.py` — `Configs` (IR → config; optional `semantic_inferrer=`/`samples=` for
  domain-aware generator selection) and `Columns` (default-generator selection).
- `engine.py` — `Generator` (config resolution, validation, the loop, checkpointing) and
  `_Journal` (append-only JSONL run log + derived `generation_summary.json`).
- `generators.py` — the column-generator library (`RandomFloats`, `BrownianFloats`,
  `UniqueIncrementalStrings`, array/timestamp/multivariate, …). Each is a class with `call()`
  and optionally `recall()` / `cardinality()`.
- `generators_finance.py` — financial-markets domain generators (coherent asset-class →
  instrument → risk-factor hierarchy, currency/tenor/rating/sector, ISIN/CUSIP/SEDOL with valid
  check digits, counterparty/book/desk); fixed-vocabulary ones declare `cardinality()`.
  Vocabularies in `_finance_data.py`.
- `semantics.py` / `semantics_agent.py` — scaffold-time semantic inference that maps columns to
  domain generators (no IR hand-tagging): `heuristic` / `values` inferrers (dependency-free) and
  the opt-in LLM `agent` (`[agent]` extra, lazy `anthropic`). See `config.md`.
- `postprocessors.py` — the two built-ins the scaffolder wires: `TemplatePostProcessor`
  (no-op hook to copy) and `KeyEnforcer` (FK / key-alignment).
- `protocols.py` — runtime-checkable `Protocol` contracts (generators, post-processors) checked
  in `Generator.validate()`, including the `INPUT_SCOPE` declaration. **Read this for the UDF contract.**
- `dtypes.py` — frame/dtype helpers and the `_Ungenerated` sentinel.
- `dependency_graph.py` — `GraphDependencies` builds a mermaid flowchart (`dump_file`) or an
  interactive HTML viewer (`dump_html`) of table/file/post-processor dependencies.
- `orchestration.py` — generation-side distribution: `run_shard`, `run_parallel`,
  `shard_batch_range`, `check_distributable`.
- `iokit/` — the storage layer (`Storages`, `cloud_providers/`); cloud SDKs are imported lazily
  per `storage_type`.

## Package boundary

datagenic is a **standalone, reusable leaf**. Importing it must NOT pull in any
platform/integration code, any cloud SDK, or even the generation engine. New public API → add
to the lazy map in `__init__.py`. Schema adaptors and any consume/ingest orchestration live in
the consuming package, not here. The `postprocessors_udfs/` package is a **dev/sample/test
fixture only** (resolved via `pythonpath = .` in `pytest.ini`); it re-exports `KeyEnforcer`/
`TemplatePostProcessor` from `datagenic.postprocessors` so older configs still resolve.
