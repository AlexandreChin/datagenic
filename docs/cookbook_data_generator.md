# Cookbook — Data Generator

---

## Quickstart: scaffold a config and generate data

`Configs` scaffolds from a neutral **internal representation (IR)** — a flat,
source-agnostic description of the tables to generate. Pass an IR (a dict or a JSON
path) directly; no adaptor needed. A frozen sample IR ships in `samples/ir/`:

```python
from datagenic import Configs, Generator

ir_path = "samples/ir/solutions_ir.json"
config_path = "samples/data_generation_config/my_config.json"

# 1 — Scaffold a config from the data model (native IR input)
Configs(
    input=ir_path,                # an IR dict or a path to IR JSON
    storage_type="local",         # "local" or "azure_blob"
    output_format="parquet",      # "parquet", "csv", or "avro"
    output_folder_path="samples/data_generator/output",
    start_date="2024-01-01",
    end_date="2024-01-05",
    number_of_batches=2,
    number_of_rows_in_fact_tables=2500,
    number_of_rows_in_non_fact_tables=100,
    seed=0,
).dump(file_path=config_path)

# 2 — Generate the data
Generator(data_generator_config_file=config_path).generate()
```

> Already have your tables described in another format? Map it to the IR with a small
> `adaptor` function — see _scaffold from another schema_ below.

---

## The native input: the IR (and scaffolding from any JSON)

`Configs` scaffolds from a **neutral internal representation (IR)** — a flat,
source-agnostic description of the tables to generate. The IR is the _native_ input:
pass it directly, no adaptor.

```python
from datagenic import Configs

source = {
    "name": "shop",
    "tables": [
        {"type": "fact", "name": "orders", "keys": ["order_id"],
         "columns": [{"name": "order_id", "type": "int"}, {"name": "amount", "type": "float"}]},
        {"type": "referential", "name": "products", "keys": ["sku"],
         "columns": [{"name": "sku", "type": "string"}]},
    ],
}
Configs(input=source).dump(file_path="config.json")  # no adaptor
```

The IR shape (an invalid shape fails fast with a clear error naming the problem):

```python
{
  "name": "<model name>",
  "tables": [
    {
      "name": "orders",
      "type": "fact",                  # "fact" or "referential" (optional, default "referential")
      "keys": ["order_id"],            # unicity-key column names
      "columns": [{"name": "order_id", "type": "int"}, …],
      "foreign_keys": [                # optional, fact tables — drives FK enforcement
        {"column": "sku", "references": "products.sku"}   # references: str | [str]
      ],
    },
    …
  ],
}
```

### `type`: fact vs referential

`type` tells the scaffolder how to treat each table — it changes two things in the
generated config (and nothing else; `type` itself is not written into the config):

- **`fact`** — the large, _referencing_ table(s) of your model (trades, orders, events…).
  Scaffolded with `number_of_rows_in_fact_tables` rows (default **2500**), and its
  `foreign_keys` are honored: a `KeyEnforcer` post-processor is wired on so every FK
  column draws only from the referenced table's value pool (referential integrity).
- **`referential`** (the default) — a smaller reference/lookup table that fact tables
  point at (currencies, products, counterparties…). Scaffolded with
  `number_of_rows_in_non_fact_tables` rows (default **100**). A `foreign_keys` block on
  a referential table is ignored.

So: mark a table `fact` to make it big and FK-enforced; leave it `referential` (or omit
`type`) for a small pool that fact tables reference. Both row counts are configurable on
`Configs(...)` (and overridable per table in the generated config).

**Scaffold from another schema — pass `adaptor=fn`**, where `fn(source) -> IR` maps
your structure into the IR. The adaptor is a plain function: take whatever shape your
schema has and return the IR dict. For example, suppose your models are described as
entities with a `role`, a primary key, and a `name: type` field map:

```python
from datagenic import Configs

# Your own schema (any shape you like — this is just an example)
my_schema = {
    "dataset": "shop",
    "entities": [
        {"table": "orders", "role": "fact", "pk": ["order_id"],
         "fields": {"order_id": "int", "sku": "string", "amount": "float"},
         "refs": {"sku": "products.sku"}},
        {"table": "products", "role": "dimension", "pk": ["sku"],
         "fields": {"sku": "string", "name": "string"}},
    ],
}

def my_adaptor(source: dict) -> dict:
    """Map a custom schema into datagenic's IR (`fn(source) -> IR`)."""
    return {
        "name": source["dataset"],
        "tables": [
            {
                "name": e["table"],
                "type": "fact" if e["role"] == "fact" else "referential",
                "keys": e.get("pk", []),
                "columns": [{"name": n, "type": t} for n, t in e["fields"].items()],
                "foreign_keys": [
                    {"column": col, "references": ref}
                    for col, ref in e.get("refs", {}).items()
                ],
            }
            for e in source["entities"]
        ],
    }

Configs(input=my_schema, adaptor=my_adaptor).dump(file_path="config.json")
```

Or from the CLI (`--adaptor module:function`, pointing at your adaptor):

```bash
# native IR — no adaptor
uv run python -m datagenic.cli generate-data-config '{"name":"shop","tables":[...]}' \
  -o config.json

# custom schema — via your own adaptor
uv run python -m datagenic.cli generate-data-config my_schema.json \
  -o config.json --adaptor mypkg.adaptors:my_adaptor
```

---

## Scaling up: one config, generate anywhere

Write **one config**, then scale by adding a flag. Every form below produces
**byte-identical** output — distributing is a work split, not a data change.

> Commands run through the project environment with `uv run` (one-time setup:
> `uv sync`). If you've activated the venv instead, drop the `uv run` prefix.

Copy-paste, end to end (uses the sample IR shipped in the repo):

```bash
# 1 — scaffold a config from the sample IR (native input, no adaptor)
uv run python -m datagenic.cli generate-data-config samples/ir/solutions_ir.json \
  -o my_config.json --output-folder ./out \
  --rows-fact 2000 --rows-dict 100 --batches 8 \
  --start-date 2024-01-01 --end-date 2024-01-05 --seed 0

# 2 — generate (single machine, sequential)
uv run python -m datagenic.cli generate-data my_config.json
```

**Use all local cores** (one machine, multicore):

```bash
uv run python -m datagenic.cli generate-data my_config.json --parallel 8
```

**Distribute across machines/workers** — no master, no per-worker config edits.
Worker `i` of `N` generates a disjoint batch slice (shards are **1-based**: `1 … N`);
all write into the same `--output-folder`, so the union is one dataset (point
downstream at `./out`):

```bash
# launch one per worker (SLURM array, k8s Job, etc.)
uv run python -m datagenic.cli generate-data my_config.json --shard 1 --of 8
uv run python -m datagenic.cli generate-data my_config.json --shard 2 --of 8
# ...through --shard 8 --of 8, or run all 8 shards locally in parallel:
seq 1 8 | xargs -P8 -I{} uv run python -m datagenic.cli generate-data my_config.json --shard {} --of 8
```

**Check it's safe to distribute** before spending cluster time (run it on your
config; green = the union of shards is provably byte-identical to a single run):

```bash
uv run python -m datagenic.cli check-distributable my_config.json
```

### Uniqueness guarantees (and their limits) — read before trusting keys

| Scope                                   | Unicity keys unique?                                                         |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| Within a `(date, batch)` partition      | **Yes**                                                                      |
| Across batches / shards (same date)     | **Yes** — disjoint per-batch key ranges                                      |
| Across dates                            | **Only if the key includes the `Date` column** (the default for fact tables) |
| Tables produced by a **post-processor** | **No** — the framework enforces nothing; your post-processor owns uniqueness |

`check-distributable` is what verifies a post-processor is actually shardable
(deterministic + each shard independent). A post-processor that uses `np.random`
(instead of the injected `_rng`), keeps cross-batch state, or reads other batches
from disk will fail it — fix it before distributing.

### Huge runs: lower peak memory (optional `default_parameters` flags)

Both default to off and leave output byte-identical until enabled:

- `"scale_skip_global_key_sort": true` — drop the per-partition key sort (needed to
  stream very large partitions).
- `"scale_unique_by_construction": true` — make composite keys unique by
  construction instead of a de-dup pass.

---

## Logging & recovery

### Logs

Every run writes two files under `output_folder_path` (relocatable via
`default_parameters`):

| File                      | Contents                                                                                 | Override key              |
| ------------------------- | ---------------------------------------------------------------------------------------- | ------------------------- |
| `generation.log`          | one JSON line per unit — `START` / `DONE` (rows) / `ERROR` (with the table, date, batch) | `generation_log_path`     |
| `generation_summary.json` | per-table totals (rows, units, errors, elapsed) + an `errors` list                       | `generation_summary_path` |

The summary is rewritten **atomically after every unit**, so even if a run crashes
the partial summary on disk is always valid JSON — read it to see exactly how far a
run got and which table/date/batch failed. (Set `logger_folder_path` to also get a
classic text log.) Distributed runs write one log/summary **per shard**
(`generation.shard{i}-of{N}.log`) so workers don't clobber each other.

### Recovery — just re-run

Generation is **checkpointed**: every parquet embeds a hash of the content-config,
and a re-run **reuses any file whose hash matches**, regenerating only what's
missing. So after a crash (or an OOM, or a killed worker) you **re-run the same
command** and it resumes where it stopped — no flags, no manual cleanup. The same
holds per shard in a distributed run.

```bash
# crashed at batch 5 of 8 → just run it again; batches 1–4 are reused
uv run python -m datagenic.cli generate-data my_config.json
```

Recovery works for **local _and_ cloud** output. For a cloud target (`azure_blob`)
the resume check reads only each finished file's parquet **footer** (a tiny ranged
read — not a full download) to validate its hash, so already-uploaded units are
skipped, not regenerated and re-uploaded.

Finer control is available from the Python API (`Generator.generate(...)`):

```python
from datagenic import Generator

gen = Generator(data_generator_config_file="my_config.json")

gen.generate()                        # resume (default): reuse valid checkpoints
gen.generate(force_regenerate=True)   # ignore checkpoints, regenerate everything
gen.generate(extend=True)             # add newly-appended dates to an existing run
gen.generate(dry_run=True)            # print the plan (tables × dates × batches), write nothing
```

Note: changing the data spec (generators, seed, row counts) changes the config hash,
so stale files are **not** reused — you won't silently mix old and new data. Only
re-locating the output / changing dates preserves reuse (that's what `extend` is for).

---

## Config file structure

### `default_parameters`

| Parameter                 | Description                                                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_format`           | `"parquet"`, `"csv"`, or `"avro"`                                                                                                                          |
| `storage_type`            | `"local"` or `"azure_blob"` (the generator's config validator rejects other values; `"s3"` and `"sftp"` are available in `Storages` directly but not here) |
| `output_folder_path`      | Root folder for generated files                                                                                                                            |
| `tables_prefix`           | Prefix prepended to every table name in file paths                                                                                                         |
| `start_date` / `end_date` | Date range — data is generated for every business day                                                                                                      |
| `number_of_batches`       | Independent shards per date (enables parallelism)                                                                                                          |
| `offset_batches`          | Skip the first N batch numbers (useful for resuming)                                                                                                       |
| `postprocessor_args`      | Shared args passed to all post-processors, including `fk_enforcement`                                                                                      |
| `seed`                    | Random seed (reserved for future use)                                                                                                                      |
| `scale`                   | Multiply `number_of_rows` and `cardinality` uniformly — popped after first use                                                                             |

### FK enforcement

Ensures FK columns in the fact table draw values exclusively from the referenced table's column pool, maintaining referential integrity across tables.

```json
"postprocessor_args": {
    "fk_enforcement": {
        "enabled": true,
        "trade_fact": {
            "Date":        ["fx_dim.fx_date"],
            "srcCurrency": ["fx_dim.srcCurrency"],
            "csaId":       ["csa_dim.csaId"],
            "productType": ["riskfactors_zip.productType"]
        }
    }
}
```

`Configs.dump()` builds this block automatically from the IR's `foreign_keys` (each fact-table FK becomes a reference into the named referential table's column pool).

### Table entry

```json
{
    "table_name": "trade_fact",
    "number_of_rows": 2500,
    "single_batch": false,
    "single_date": false,
    "dump_results": true,
    "post_processor": {},
    "columns": [...]
}
```

| Field            | Description                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| `number_of_rows` | Total rows across all batches for one date                                        |
| `single_date`    | If `true`, only generate for `start_date`                                         |
| `single_batch`   | If `true`, only generate for the first batch                                      |
| `dump_results`   | If `false`, keep in memory only (useful for intermediate tables)                  |

> **Memory model.** The engine keeps one frame per table in memory (the current
> `(date, batch)` unit), overwritten each unit — so peak memory is bounded by
> _number of tables × rows-per-unit_, not by the number of dates/batches. `dump_results:
> false` saves **disk I/O, not memory**: an intermediate table stays resident for the
> whole run (it may be read by any later table's post-processor on any date), and frames
> are reclaimed only when overwritten, not evicted once a downstream consumer is done.

---

## Built-in column generators (`datagenic.generators`)

| Generator                   | Type          | Description                                                   |
| --------------------------- | ------------- | ------------------------------------------------------------- |
| `UniqueIncrementalIntegers` | int           | Monotonically increasing integers per batch                   |
| `UniqueIncrementalStrings`  | string        | `{prefix}{n}` with no repeats across batches                  |
| `UniqueIncrementalDates`    | date          | Sequential dates starting from `start_date`                   |
| `RandomIntegers`            | int           | Random integers with configurable cardinality                 |
| `RandomStrings`             | string        | Random `{prefix}{n}` with configurable cardinality            |
| `RandomDates`               | date          | Random dates from an optional fixed list                      |
| `RandomTimeStamps`          | datetime      | Random timestamps                                             |
| `BrownianFloats`            | float         | Normal distribution with Brownian shocks across time          |
| `BrownianArrayFloats`       | array(float)  | Same, returning a fixed-size array per row                    |
| `RandomArrayStrings`        | array(string) | Random string arrays with configurable cardinality            |
| `IncrementalArrayDates`     | array(date)   | Arrays of incrementally spaced dates                          |
| `CurrentAsofDates`          | date          | Always the current batch date (used for the `Date` column)    |
| `SingleValues`              | any           | Fills the column with a single constant value                 |
| `ValuesFromFile`            | any           | Reads values from a CSV or Parquet file                       |
| `ListOfValues`              | any           | Samples from a user-supplied list with optional probabilities |

Private arguments injected automatically into every generator call:

| Argument                    | Value                                          |
| --------------------------- | ---------------------------------------------- |
| `_date`                     | Current business date                          |
| `_previous_date`            | Previous business date                         |
| `_start_date` / `_end_date` | Config range                                   |
| `_batch_number`             | Current batch (1-indexed, with offset applied) |
| `_number_of_batches`        | Total batches                                  |
| `_number_of_rows_per_batch` | `ceil(number_of_rows / number_of_batches)`     |

---

## Custom column generator

```python
# mypkg/my_generators.py  (any module importable on your PYTHONPATH)
class MyGenerator:
    def call(self, prefix: str, _number_of_rows_per_batch: int, **kwargs):
        import numpy as np
        return np.char.add(prefix, np.arange(1, _number_of_rows_per_batch + 1).astype(str))
```

Reference it in the config:

```json
{
  "col_name": "myColumn",
  "col_type": "string",
  "is_unicity_key": false,
  "column_generator": {
    "udf_name": "MyGenerator",
    "udf_module": "mypkg.my_generators",
    "args": { "prefix": "item_" }
  }
}
```

---

## Post-processors

A post-processor replaces or augments the column-generated data for a table. Common uses: loading a real reference file, joining two intermediate tables, applying business rules.

> **Uniqueness & sharding are your responsibility here.** The framework's unicity-key
> handling applies to column-generated data only — a post-processor's output is written
> as-is. For distributable runs, use the injected `_rng` (never `np.random`/`uuid`/`time`),
> avoid cross-batch state, and verify with `check-distributable` (see _Scaling up_). Declare
> `INPUT_SCOPE = "global"` on the class if it truly needs whole-table/cross-batch data — that
> marks the table non-shardable.

```json
{
  "table_name": "FinancialInstruments",
  "number_of_rows": 0,
  "columns": [],
  "single_batch": true,
  "single_date": true,
  "dump_results": true,
  "post_processor": {
    "udf_module": "postprocessors_udfs.generic_udfs",
    "udf_name": "ImportFile",
    "udf_args": { "file_path": "tests/source/financial_instruments.csv" }
  }
}
```

Extra private arguments available inside post-processor `call()`:

| Argument                    | Value                                                       |
| --------------------------- | ----------------------------------------------------------- |
| `_table`                    | The `DataFrame` generated so far for this table             |
| `_input_tables`             | Dict of the upstream tables this post-processor declared in `udf_args._input_tables`, resolved to their data (current batch's slice) — exactly those, `{}` if none declared |
| `_unicity_keys`             | List of unicity key column names for this table             |
| `_date`                     | Current business date being generated                       |
| `_previous_date`            | Previous business date (`_start_date` on the first date)    |
| `_start_date`               | Config range start date                                     |
| `_end_date`                 | Config range end date                                       |
| `_batch_number`             | Current batch number (1-indexed, offset applied)            |
| `_number_of_batches`        | Total number of batches                                     |
| `_number_of_rows_per_batch` | `ceil(number_of_rows / number_of_batches)` for this table   |
| `_seed`                     | Deterministic per-unit seed (`sha256(seed \| table \| date \| batch)`) |
| `_rng`                      | A seeded `numpy.random.Generator` (`default_rng(_seed)`) — **use this for any randomness**, never the global `numpy.random` |

A post-processor class must expose a `call(**kwargs)` method. Optionally implement `recall(**kwargs)` to apply time-evolution logic on subsequent dates (same signature as `call`).

---

## Generating the workflow graph

```python
from datagenic import GraphDependencies

GraphDependencies(
    "samples/data_generation_config/data_generator_config_post_processor.json"
).dump_file(folder_path="samples/graphs", mermaid_header=False)
# Output: samples/graphs/data_generator_dependencies.md
```

The graph shows which files and tables feed which post-processors. Only tables that use a post-processor appear in the graph.

For the `shop` model above (a `fact` table `orders` with an FK to the `referential`
table `products`, plus a file-loaded table), `dump_file` renders a flowchart like this —
each table's pipeline is grouped in its own box, and node type shows in both colour
**and** shape — column generators are amber stadiums, post-processors pink hexagons,
temp results grey, tables green, files blue (see the legend). When a table has **both**
column generators and a post-processor, the column data is a *temp result* the
post-processor consumes:

```mermaid
---
title: Data Generator Workflow
---
%%{init: {'flowchart': {'curve': 'basis', 'nodeSpacing': 45, 'rankSpacing': 70}}}%%
flowchart LR
    classDef table fill:#bbf7d0,stroke:#16a34a,stroke-width:2.5px,color:#052e16
    classDef generator fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#451a03
    classDef postprocessor fill:#fce7f3,stroke:#db2777,stroke-width:1.5px,color:#500724
    classDef intermediate_results fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,stroke-dasharray:4 3,color:#0f172a
    classDef file fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554

    %% referential table: column generators feed it directly
    subgraph products_pipe["products · referential"]
        direction LR
        pg(["⚙️ column generators"]):::generator --> products[("🗄️ products")]:::table
    end

    %% fact table: generators -> temp result -> KeyEnforcer -> table
    subgraph orders_pipe["orders · fact"]
        direction LR
        og(["⚙️ column generators"]):::generator --> tmp[("🧪 temp result")]:::intermediate_results
        tmp --> ke{{"🧩 KeyEnforcer<br/>datagenic.postprocessors"}}:::postprocessor --> orders[("🗄️ orders")]:::table
    end

    %% post-processor-only table loaded from a file
    subgraph instruments_pipe["instruments · file-loaded"]
        direction LR
        csv["📄 financial_instruments.csv"]:::file --> imp{{"🧩 ImportFile<br/>generic_udfs"}}:::postprocessor --> instruments[("🗄️ instruments")]:::table
    end

    %% FK pool: products feeds the enforcer that fills orders' FK columns
    products -->|FK pool| ke

    subgraph legend["legend"]
        direction LR
        lf["📄 file"]:::file ~~~ lg(["⚙️ generator"]):::generator ~~~ lp{{"🧩 post-processor"}}:::postprocessor ~~~ lt[("🧪 temp result")]:::intermediate_results ~~~ ltab[("🗄️ table")]:::table
    end

    %% subgraphs: no fill on the pipelines, only the legend is filled
    style products_pipe fill:transparent
    style orders_pipe fill:transparent
    style instruments_pipe fill:transparent
    style legend fill:#f1f5f9,stroke:#94a3b8
```

Read the edges as _"feeds into"_: for `orders`, the column generators produce a temp
result that `KeyEnforcer` post-processes — and `products ---> KeyEnforcer` means the
`products` value pool is what the enforcer draws FK values from. For `instruments`,
`financial_instruments.csv ---> ImportFile ---> instruments` loads the table straight
from the file. (The real `dump_file` output uses this exact style — one left-to-right
subgraph per table, the legend, and the colours/shapes above. Two cosmetic differences:
its subgraph titles are just the table names, since `fact`/`referential` isn't carried
in the config, and cross-table edges like the FK one are unlabeled.)

### One unit at a time, replayed across dates and batches

This DAG is the dependency structure for a **single `(date, batch)` unit** — the engine
walks it once per unit, inside the nested loop **batch → date → table → columns →
post-processor**. The *shape* never changes from unit to unit; what changes is the
per-unit seeded RNG (`sha256(seed | table | date | batch)`), which is exactly what keeps
each unit independent and reproducible.

- **Across dates** — the first date invokes each generator/post-processor's `call()`;
  later dates invoke `recall()`, which propagates state forward (e.g. a Brownian series
  evolves from the previous date, and unicity keys stay stable). `single_date` tables are
  produced only for `start_date`.
- **Across batches** — every batch replays the same DAG with its own seed and a disjoint
  key range, so batches are fully independent: byte-identical on re-run and safe to shard
  across workers. `single_batch` tables (small reference pools) are generated once on the
  first batch and reused by the rest.

So the graph shows _what depends on what_ within a unit; the full dataset is that DAG
executed once per `(date, batch)`, with `call()`/`recall()` selecting first-date vs
time-propagated behaviour.
