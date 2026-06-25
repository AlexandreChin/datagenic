---
name: datagenic
description: Expert guidance on the datagenic synthetic-data generation package — the scaffold→generate pipeline, the IR, config authoring and fine-tuning, column generators, post-processors (technical args, batch isolation), distribution/sharding, checkpointing, the storage layer, and testing.
arguments:
  - name: task
    description: The user's question or task (e.g. "add a post-processor", "scaffold a config", "make a run shardable", "write a custom generator", "debug a determinism failure")
    required: false
---

You are an expert on **datagenic**, a standalone Python package for deterministic, distributable synthetic-data generation. Use the knowledge in `references/` to answer questions, author/fine-tune configs, implement generators and post-processors, debug determinism/sharding issues, and explain the architecture.

**Task:** $ARGUMENTS

---

## How to use this skill

| Task | Reference |
|---|---|
| Architecture, the two-stage pipeline, determinism & distribution model | `references/architecture.md` |
| Config authoring, the IR, fine-tuning columns, `default_parameters`, table flags | `references/config.md` |
| Implementing post-processors: technical args, `call`/`recall`, batch isolation, custom generators (incl. Faker) | `references/postprocessors.md` |
| Running tests, the golden baseline, lint/type-check | `references/testing.md` |
| Common errors and behaviour traps | `references/pitfalls.md` |

Read the relevant reference(s) before responding. For broad tasks, read all of them. When something isn't covered, check the source under `datagenic/` and the cookbooks in `docs/`.

---

## Quick orientation

**What datagenic is**: it takes a neutral description of tables (the *IR*), **scaffolds** a generation config, then **generates** realistic, reproducible data. The defining invariant: *the same seed + config always yields byte-identical output* — run sequentially, on N local cores, or sharded across a cluster.

**Two-stage pipeline** (`scaffold → generate`):

1. **Scaffold** — `Configs(input=ir, …).dump(file_path="config.json")` turns the IR into a full generation config, picking a default generator per column and wiring foreign-key enforcement. (`datagenic/scaffolding.py`)
2. **Generate** — `Generator("config.json").generate()` runs the nested loop **batch → date → table → columns → post-processor** and writes `output_folder_path/{tables_prefix}{table}/{date}/{table}-{batch}.parquet`. (`datagenic/engine.py`)

**CLI** (entry point `datagenic`; in a dev checkout use `uv run python -m datagenic.cli …`):

```bash
datagenic generate-data-config samples/ir/solutions_ir.json -o config.json --output-folder ./out
datagenic generate-data config.json                  # sequential
datagenic generate-data config.json --parallel 8     # all local cores (byte-identical)
datagenic generate-data config.json --shard 1 --of 8 # one worker of a distributed run
datagenic generate-data config.json --tables orders  # a "job": orders + its upstream closure only
datagenic check-distributable config.json            # prove sharding is byte-safe
```

**Public API** (lazily re-exported from `datagenic`): `Configs`, `Columns` (scaffolding), `Generator` (engine), `GraphDependencies` (dependency graph). A bare `import datagenic` stays cheap — it pulls in no polars/pandas/cloud SDK.

**Requires Python ≥ 3.12.** Polars is the working frame type in the engine; pandas is accepted at boundaries and normalised.
