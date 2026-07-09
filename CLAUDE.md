# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`datagenic` is a standalone Python package for **deterministic, distributable synthetic-data generation**. It takes a neutral description of tables (the *IR*), scaffolds a generation config, then executes that config to produce realistic, reproducible data. The defining invariant across the whole codebase: **the same seed + config always yields byte-identical output** — whether run sequentially, on N local cores, or sharded across a cluster. Most non-obvious design choices exist to protect that invariant.

Requires Python ≥ 3.12.

> **Skill available.** A `datagenic` skill lives in `.claude/skills/datagenic/` (SKILL.md + references on architecture, config/fine-tuning, post-processors & batch isolation, testing, pitfalls). Use it for deeper task-specific guidance.

## Commands

```bash
uv sync                              # install all deps incl. dev + cloud groups
# or: pip install -e ".[cloud]"      # without uv
# or: pip install -e ".[cloud,agent]" # + the anthropic SDK for `--infer-semantics agent`

pytest tests/                        # full suite
pytest tests/engine/                 # one feature area (tests are grouped by feature)
pytest tests/engine/test_determinism.py::test_same_seed_is_byte_identical   # one test
pytest -m "not integration"          # skip tests needing cloud creds/network
pytest -m cloud                      # slice by trait marker (also: slow, agent, integration)

ruff check .                         # lint
ruff format .                        # format (line-length 100, double quotes)
mypy datagenic                       # type-check (non-strict)
```

CLI (installed as `datagenic`, entry point `datagenic.cli:main`):

```bash
datagenic generate-data-config samples/ir/solutions_ir.json -o config.json --output-folder ./out
datagenic generate-data-config ir.json -o config.json --infer-semantics heuristic       # auto-detect domain columns (heuristic|empirical|agent|module:function)
datagenic generate-data-config ir.json -o config.json --infer-semantics empirical --samples ./sample_data  # value-based, from real sample files
datagenic profile ./sample_data -o config.json      # scaffold a config straight from sample files (no IR) — infers keys/FKs + fits generator args
datagenic review-config config.json                  # list inferred columns (lowest confidence first) to fine-tune
datagenic generate-data config.json                  # sequential
datagenic generate-data config.json --parallel 8     # all local cores (byte-identical to sequential)
datagenic generate-data config.json --shard 1 --of 8 # one worker of a distributed run
datagenic generate-data config.json --tables orders  # a "job": only this table + its upstream closure
datagenic check-distributable config.json            # verify sharding is byte-safe
```

### Golden baseline

`tests/engine/test_determinism.py::test_golden_manifest_matches` asserts generated parquet hashes against `tests/golden/baseline_manifest.json`. If you intentionally change generation output, regenerate the manifest:

```bash
python tests/golden/capture_baseline.py
```

The fixtures feeding both capture and assertion live in `tests/golden/golden_fixtures.py` (single source of truth, so the manifest and the test run identical configs). A failure here that you did *not* intend means a determinism regression — investigate, don't just refresh.

## The two-stage pipeline

Everything flows through **scaffold → generate**:

1. **Scaffold** (`datagenic/scaffolding.py`, `Configs`): IR → a full generation config JSON. `Configs` picks a default column generator per `(type, is_unicity_key, n_keys)` via the `Columns` class, wires foreign-key enforcement (`KeyEnforcer`) onto fact tables, and optionally attaches a no-op `TemplatePostProcessor` to every table. Optionally, `semantic_inferrer=` (CLI `--infer-semantics`) auto-detects domain columns — e.g. financial-markets (asset class, instrument, ISIN) — and wires realistic generators, annotating each choice under `_inferred` for review. This covers *numeric* facts too, scalar and array: a `notional`/`price`/`rate` gets a plausible distribution family + scale (lognormal in the millions, not `N(1000,100)`), a `notional` is made **conditional** on `asset_class` (`ConditionalFloats`), and an `array(float)` `pnl` becomes a per-row time series scaled per asset class (`ConditionalArrayFloats`). `empirical`/`profile` fit these from a real sample (element params + array length), `agent` proposes them. Inference is **scaffold-time only**, so the resulting config is frozen and generation stays byte-identical regardless. Per-column IR hints `null_rate`/`null_when` become a `null_spec` the engine applies as deterministic, shard-safe missing-value injection (`Generator._apply_null_mask`; never on a unicity key). Output is dumped with `.dump(file_path=...)`.
2. **Generate** (`datagenic/engine.py`, `Generator`): config → data files. `Generator(config).generate()` runs the nested loop **batch → date → table → columns → postprocessor**.

The **IR** is the native input: a flat `{name, tables:[{type, name, keys, columns:[{name, type}], foreign_keys}]}`. To scaffold from another schema, pass `adaptor=fn` where `fn(source) -> IR` — a small one-time mapper from your source format to the IR. Such adaptors live in the consuming/platform package, not here — this repo stays platform-free on purpose (see "Package boundary" below).

## Module map (`datagenic/`)

- `__init__.py` — lazy (PEP 562) re-exports of the public API: `Configs`, `Columns` (scaffolding), `Generator` (engine), `GraphDependencies` (dependency_graph). A bare `import datagenic` must stay cheap — it pulls in no polars/pandas/cloud SDK. Keep it that way.
- `scaffolding.py` — `Configs` (IR → config; optional `semantic_inferrer=`/`samples=` for domain-aware generator selection) and `Columns` (default-generator selection). Opt-in `Configs(reference_rows="auto")` sizes each **reference/dimension table to its strict minimal grain** instead of a flat count: `number_of_rows = ∏ cardinality(unicity key)` — a bounded key (fixed-vocabulary generator / `list_of_values`) uses its real domain, a free (synthetic-id) key uses the `number_of_rows_in_non_fact_tables` budget, and only when ≥2 keys are free is that budget split geometrically (the sole case cardinalities are equalized). Auto references become broadcast dimensions (`single_batch`/`single_date`), and composite keys enable `scale_unique_by_construction` so the grain is enumerated *exactly*. Default (`reference_rows=None`) keeps the legacy flat sizing, so it's additive/golden-safe. A **sole bounded-vocabulary key** is sampled, not enumerated, so it can undershoot its domain (~0.63× for N-from-N) — this warns; use a composite key or explicit `number_of_rows` for exact coverage. Reference size is independent of the fact tables (or how many) that reference it (`_key_cardinality`/`_resolve_grain`/`_size_reference_table`).
- `engine.py` — the heart. `Generator` (config resolution, validation, the generation loop, checkpointing) and `_Journal` (append-only JSONL run log + derived `generation_summary.json`).
- `generators.py` — the library of column generators (`RandomFloats`, `BrownianFloats`, `UniqueIncrementalStrings`, array/timestamp/multivariate generators, etc.). Each is a class with `call()` and optionally `recall()` / `cardinality()`. The float generators (scalar **and** array: `RandomArrayFloats`/`BrownianArrayFloats`) take a `distribution` arg (`"normal"` default — byte-identical — or `"lognormal"`, median + log-σ via `_draw_floats`); the array ones also take `path` (`"iid"` default / `"walk"`, a cumulative per-row series). `ConditionalFloats` / `ConditionalArrayFloats` draw each row from a per-category distribution read off a parent column (`by_column`) in the partial `_table` — the numeric analogue of the coherent hierarchy — falling back to a `default` when the parent is absent.
- `domains/market_finance/generators.py` — financial-markets domain generators: a coherent asset-class → instrument-type → risk-factor-group hierarchy (children read the partial `_table`), currency/tenor/rating/sector, ISIN/CUSIP/SEDOL with valid check digits, coherent dates (`TradeDate`/`MaturityDate`/`SettlementDate`, `maturity = trade_date + tenor`), and counterparty/book/desk. Also the **rolling book** family (`RollingTradeDate`/`RollingTenor`/`RollingMaturityDate`/`RollingTradeId`, opt-in via `Configs(rolling_book=True)`): each row is a slot running a deterministic renewal process seeded date-independently from `(global_seed, batch, slot)` — at any as-of date the slot emits the trade whose `[trade_date, maturity)` interval contains it, so the book is a constant-size set of *live* trades (`trade_date ≤ as-of < maturity`), short-tenor slots roll (new tenor + new id) while long ones persist, all byte-reproducible across dates/shards. These re-evaluate per date via `recall`, and the scaffolder skips the key sort (`scale_skip_global_key_sort`) so row-index → slot stays stable. Fixed-vocabulary categoricals declare `cardinality()` so the engine budgets unicity keys correctly (they can't exceed their domain). Also **tenor-bucketed sensitivities** (`TenorSensitivities` + `SensiType`/`TenorGrid`): a risk ladder — an `array(float)` aligned to a tenor grid — where each bucket is a signed value scaled by the row's `asset_class`, and every bucket beyond the instrument's remaining maturity (`maturity_date − as-of`, snapped to the nearest grid bucket) is zeroed. The shape keys on `sensi_type` **and** instrument (precedence *spot → bullet → ladder*): a **spot-risk sensi** (equity/FX `delta`/`gamma`, `SPOT_SENSIS_BY_ASSET_CLASS`) collapses to a single short-end bucket (a spot is a point, not a curve); a zero-coupon-like instrument (`BULLET_INSTRUMENTS`) gets a *bullet* at its maturity bucket; everything else (rates/credit delta, vega by expiry, rho…) gets a *ladder* across all live buckets. So the same FX forward gives a spot-point `delta` but a maturity-bullet `rho`. `sensi_type` is read per-row from a column or set by a fixed arg. It reads `asset_class`/`instrument_type`/`maturity_date` from the partial `_table` and re-evaluates per date via `recall`, so the live region shrinks as the trade ages; the scaffolder orders it after `maturity_date`. Vocabularies live in `domains/market_finance/_data.py` (no third-party imports) — which also holds the numeric priors: `QUANTITY_PRIORS`/`CONDITIONAL_QUANTITY_PRIORS` (scalar, e.g. notional scaled by asset class) and `ARRAY_QUANTITY_PRIORS`/`CONDITIONAL_ARRAY_QUANTITY_PRIORS` (array, e.g. a PnL series whose vol scales by asset class), plus the sensitivity priors (`SENSI_TYPES`, `DEFAULT_TENOR_GRID`, `BULLET_INSTRUMENTS`, `SENSI_MAGNITUDE_BY_ASSET_CLASS`). `COUNTERPARTIES` is loaded (stdlib `importlib.resources`) from bundled real large-cap CSVs in `datagenic/domains/market_finance/data/counterparties/*.csv` (CAC 40, DAX 40, EURO STOXX 50, SMI, FTSE 100, S&P 500, Nikkei 225, TSX 60, ASX, HSI, Nifty 50, KOSPI, Ibovespa, Taiwan, + European extras — a snapshot, **~1,040 names** with sector + country), with a fictional fallback if the package data is absent. `domains/market_finance/_semantics.py` is the pack's `register(registry)` hook (exposed as `datagenic.domains.market_finance.register`): it maps all the finance semantics/patterns/links/value-classifiers into the `semantics_registry`, so `semantics.py` stays domain-free and the pack is discovered automatically.
- `semantics.py` — scaffold-time semantic inference: maps each column to a domain generator automatically so the IR needs no hand-tagging. **Domain-agnostic — it names no specific domain**; all domain content is contributed by domain packs via `semantics_registry` (see below). Exposes `SEMANTIC_GENERATORS` (an alias for the registry's generator map), `resolve_inferrer`, `build_generator`, and the dependency-free `heuristic` (name-based, **type-gated** off the registry's per-type patterns: string→categorical/identifier, scalar `float`→scalar quantity, `array(float)`→array quantity — width-agnostic, so `array(float32)`/`array(float64)` count, scalar `int`→a `product_id`-style identifier) and `empirical` (sample-value based) inferrers. `fit_distribution` / `fit_array_distribution` fit a numeric (scalar or array) column's family + params — and, for arrays, `size_of_arrays` — from a sample (flat or per-parent-category `cases`), parameterised to round-trip with `_draw_floats`; `_fit_numeric` dispatches on the registry's per-semantic numeric fit-kind. Used by `Configs`; produces a fully hand-editable config (`_inferred` annotations).
- `semantics_registry.py` — the **domain-pack registry** decoupling core inference from any domain. A `SemanticRegistry` holds semantic→generator specs, ordered per-type name patterns, parent-coherence links, per-semantic numeric fit-kinds, and lazy value-classifier factories. `load()` (called once at `import datagenic.semantics`) registers a few domain-neutral core generics (`identifier`/`identifier_int`) then **auto-discovers** every `datagenic/domains/*` pack via `pkgutil` and calls its `register(registry)` hook. Dependency-free and numpy-free (registration records strings/patterns + each pack's pure data; generator classes load only when `build_generator` instantiates one). A new domain is a folder with a `register` hook — no core edit.
- `semantics_agent.py` — the opt-in LLM `agent` inferrer (Claude via the `anthropic` SDK, `[agent]` extra). Labels columns and, for numeric quantities (scalar or array), proposes a distribution family + scale (array: + `size_of_arrays`) and a conditioning column, sanitised before use. Lazy-imported only when invoked, so a bare import — and `import datagenic.scaffolding` — pulls in no SDK.
- `profiling.py` — **profile-from-sample** scaffolding (CLI `profile`): `build_config(sources)` reads sample files, synthesises an IR (tables, key & FK candidates; polars `List` columns typed `array(float)`) and fits generator args (cardinality, family-aware `distribution`+mu/sigma via `fit_distribution`, array element params + length via `fit_array_distribution`, value frequencies → `list_of_values`/`probabilities`, null rate) from the data, then scaffolds via `Configs`. The IR is internal (optionally emitted); the user supplies only files. Scaffold-time only, so generation stays deterministic.
- `postprocessors.py` — the two built-ins the scaffolder wires: `TemplatePostProcessor` (no-op hook to copy/customize) and `KeyEnforcer` (FK / key-alignment).
- `protocols.py` — runtime-checkable `Protocol` contracts for column generators and postprocessors, enforced in `Generator.validate()`. **Read this to understand the UDF contract**, including the `INPUT_SCOPE` shardability declaration.
- `dtypes.py` — frame/dtype helpers and the `_Ungenerated` sentinel (referencing a table before it's produced raises, instead of silently yielding an empty frame).
- `dependency_graph.py` — `GraphDependencies` builds a mermaid flowchart of table/file dependencies from a config.
- `orchestration.py` — generation-side distribution: `run_shard`, `run_parallel`, `shard_batch_range`, and `check_distributable`.
- `iokit/` — the storage layer. `storages.py` (`Storages`, the unified local/cloud interface) and `cloud_providers/` (azure_blob, aws_s3, sftp). Cloud SDKs are imported **lazily per `storage_type`** so local generation needs none of them.

## Determinism model (the core invariant)

- **Per-unit seeded RNG.** Each `(table, date, batch)` unit gets its own `numpy.random.default_rng` seeded from `sha256(global_seed | table | date | batch)` (`Generator._make_rng` / `_derive_seed`). Generators and postprocessors must draw from the **injected `_rng`** — never `numpy.random.*` global state, `random`, `uuid`, or `time`. An extra draw in one place then cannot perturb any other unit, and shards/parallel workers stay independent. The engine also injects the raw `_global_seed` into generator `call`/`recall`: use it (not `_rng`) only when a value must be **date-independent** yet still seed-tied — e.g. the rolling-book renewal, keyed on `sha256(global_seed | batch | slot)` so a slot's schedule is identical on every date and shard.
- **Config is read-only.** `Generator` keeps a pristine `_source_config` deep copy and rebuilds a fresh working config at construction *and* at the start of every `generate()`. Repeated `generate()` calls and reusing one config dict across `Generator` instances are all byte-identical. Don't introduce mutation of the caller's dict.
- **Distribution is master-free.** Sharding splits the batch space into disjoint slices (`shard_batch_range`, 1-based) injected as `batch_range` into a *copy* of the config; seeds are untouched, so the union of all shards equals a single run byte-for-byte. `run_parallel` does the same across local processes — and deliberately avoids the `fork` start method (numpy/polars threads + fork = deadlock), forcing `forkserver`/`spawn`.
- **Shardability gate.** A postprocessor declares `INPUT_SCOPE` = `"batch"` (default) / `"reference"` / `"global"`. `validate(distributed=True)` statically rejects a sharded table whose postprocessor declares `"global"`. `orchestration.check_distributable` proves it dynamically: a determinism probe (regenerate under a perturbed global RNG, require identical output) plus a sample full-run-vs-2-shards byte equality check.

## Checkpointing & resume

Every dumped parquet embeds key-value metadata: `datagenic_config_hash`, `datagenic_seed`, `datagenic_end_date`. On re-run, a unit is skipped (its frame restored) only if a matching parquet already exists with the same `_config_hash` (`Generator._load_checkpoint`). The config hash deliberately **excludes IO/location fields** (output path, storage type, format, log paths, batch offset, end_date) so the same data written elsewhere — or with an extended date range — stays valid. This works identically for cloud output: the storage layer validates via the parquet footer/metadata without downloading the whole object. `force_regenerate=True` bypasses; `extend=True` requires existing data.

## Package boundary (important)

This repo was split out to be a **standalone, reusable leaf**. Importing `datagenic` must NOT pull in any platform/integration code, any cloud SDK, or even the generation engine. Honor this when adding code:

- New public API → add to the lazy `_LAZY` map in `__init__.py`, don't eagerly import.
- Platform/ingestion concerns (schema adaptors, the generate→consume→reclaim orchestrator, push/ingest CLI commands) belong in the consuming/platform package, not here.
- The `postprocessors/` package is a **dev/sample/test fixture only** — not shipped. It's resolved via `pythonpath = .` in `pytest.ini`. It re-exports `KeyEnforcer`/`TemplatePostProcessor` from `datagenic.postprocessors` so older configs referencing `postprocessors_udfs.generic_udfs.*` still resolve.

## Conventions

- **Polars is the working frame type** in the engine; pandas is accepted at boundaries and normalized (`postprocessors._as_pl`, `dtypes._as_pl_series`). New generation/postprocessing code should be Polars-first.
- **Framework-injected UDF args use a leading `_`** (`_table`, `_input_tables`, `_rng`, `_date`, `_seed`, …). User config `udf_args` arrive as plain kwargs alongside them; a `_`-prefixed key in config is reserved and rejected by validation (except `_input_tables`). UDFs should accept `**kwargs` so new framework args never break them.
- `tests/` are grouped into **feature directories that mirror the module map** — `engine/` (+ `engine/runtime/` for the journal/checkpoint lifecycle), `contracts/`, `scaffolding/`, `semantics/`, `quantities/{scalar,array}/`, `market_finance/`, `postprocessors/`, `profiling/`, `scale/`, `storage/`, `packaging/`. Import mode is `importlib` (see `pytest.ini`), so the same test-file basename may recur across directories. `tests/conftest.py` puts `tests/golden` on `sys.path` (so any test can `import golden_fixtures`) and auto-applies trait markers by path (`cloud`→`storage/`, `agent`, `slow`); `integration` stays explicit. `tests/golden/` holds the byte-equality baseline; `samples/` holds frozen IR, sample configs, and generated output (the latter is gitignored and regenerated by the suite).
