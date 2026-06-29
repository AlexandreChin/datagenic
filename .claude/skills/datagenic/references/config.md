# Config Reference

datagenic has **two** documents: the **IR** (input you describe) and the **generation config**
(output of scaffolding, consumed by `Generator`). The IR has no generators; scaffolding assigns
them.

## The IR (native input)

```json
{
  "name": "shop",
  "tables": [
    {
      "name": "orders",
      "type": "fact",                  // "fact" or "referential" (optional, default "referential")
      "keys": ["order_id"],            // unicity-key column names
      "columns": [{"name": "order_id", "type": "int"}, {"name": "amount", "type": "float"}],
      "foreign_keys": [                // optional, fact tables — drives FK enforcement
        {"column": "sku", "references": "products.sku"}   // references: str | [str]
      ]
    },
    {"name": "products", "type": "referential", "keys": ["sku"],
     "columns": [{"name": "sku", "type": "string"}]}
  ]
}
```

`type` is validated (`"fact"` | `"referential"`, default `"referential"`) and drives two things
at scaffold time: **row-count scale** (fact tables get `number_of_rows_in_fact_tables`, default
2500; referential get `number_of_rows_in_non_fact_tables`, default 100) and **FK enforcement**
(only fact tables' `foreign_keys` are honoured — a `KeyEnforcer` post-processor is wired on).
`type` is consumed during scaffolding and is **not** written into the generation config.

Scaffold from the IR:

```python
from datagenic import Configs
Configs(input="ir.json").dump(file_path="config.json")          # IR dict or JSON path
Configs(input=src, adaptor=my_adaptor).dump(file_path="config.json")  # any schema via fn(source)->IR
```

## Generation config — top-level structure

```json
{
  "default_parameters": {
    "output_format": "parquet",          // "parquet" | "csv" | "avro"
    "storage_type": "local",             // "local" | "azure_blob"  (generator validator)
    "output_folder_path": "./out",
    "tables_prefix": "",
    "start_date": "2024-01-01",
    "end_date": "2024-01-05",
    "number_of_batches": 8,
    "offset_batches": 0,
    "postprocessor_args": { "fk_enforcement": { "...": "..." } },
    "seed": 0
  },
  "tables": [ ... ]
}
```

Optional `default_parameters`: `scale` (multiplies row counts + non-key cardinalities, popped
after first use), `batch_range` (injected by sharding), `generation_log_path` /
`generation_summary_path` / `logger_folder_path`, and the peak-memory flags
`scale_skip_global_key_sort` / `scale_unique_by_construction` (both default off; off ⇒ output
unchanged).

## Table entry

```json
{
  "table_name": "trade_fact",
  "number_of_rows": 2500,                 // int, or {"from_table": "x", "multiplier": 1.0}
  "single_batch": false,
  "single_date": false,
  "dump_results": true,
  "post_processor": {},
  "columns": [ ... ]
}
```

| Flag | Default | Effect |
|---|---|---|
| `single_batch` | `false` | Generate once (batch 1) and broadcast — the pattern for reference pools shared across batches |
| `single_date` | `false` | Generate only on `start_date`; reused on later dates |
| `dump_results` | `true` | `false` = keep in memory only (intermediate table), write no parquet |

There is **no `table_type` field** — behaviour comes entirely from these three flags. (Memory
note: the engine keeps one frame per table in memory, overwritten each unit, so peak memory is
bounded by *tables × rows-per-unit*; `dump_results: false` saves disk I/O, not memory.)

## A scaffolded column, and how to fine-tune it

```json
{
  "col_name": "fx_rate",
  "col_type": "float32",
  "is_unicity_key": false,
  "column_generator": {
    "udf_name": "BrownianFloats",
    "udf_module": "datagenic.generators",
    "args": { "mu": 1000, "sigma": 100, "list_of_values": null, "probabilities": null },
    "description": "..."
  }
}
```

Fine-tune by editing this block in place — the keys under `args` are the generator's tunable
parameters (pre-filled with defaults):

- **Tune args** — `mu`/`sigma` (float distribution), `cardinality` (distinct values), `prefix`
  (string labels), `size_of_arrays` (array columns).
- **Pin exact values** — most generators accept `list_of_values` + `probabilities`, e.g.
  `"list_of_values": ["EUR","USD","GBP"], "probabilities": [0.5,0.3,0.2]`.
- **Swap the generator** — change `udf_name` to any built-in in `datagenic.generators` (or
  `datagenic.generators_finance` for realistic financial-markets values) and set its `args`.
- **Use your own** — point `udf_module`/`udf_name` at a custom generator on the PYTHONPATH.

Editing `args`, `seed`, or row counts changes the **config hash**, so a re-run regenerates the
affected files instead of reusing stale checkpoints. Relocating output or only extending the
date range preserves reuse (that's what `extend=True` is for).

## Domain-aware scaffolding & semantic inference

Beyond the type-based default generators, the scaffolder can detect a column's **domain
meaning** and wire a realistic generator automatically — currently financial markets
(`datagenic.generators_finance`). This happens **at scaffold time only**, so the result is a
normal frozen config; generation determinism is unaffected.

**Finance generators** (`udf_module: "datagenic.generators_finance"`):

- Coherent hierarchy — `FinancialAssetClass` → `FinancialInstrumentType` /
  `FinancialRiskFactorGroup`. The children read the partial row and sample values valid for
  that row's asset class, so the parent column must be **generated first** (list it earlier;
  the scaffolder reorders inferred ones). The child's `args.asset_class_column` names the parent.
- Market dims — `Currency`, `CurrencyPair`, `Tenor`, `CreditRating` (`scale`), `GICSSector`.
- Identifiers — `ISIN`/`CUSIP`/`SEDOL` (valid check digits), `InstrumentName`, `RiskFactorName`.
- Org — `Counterparty`, `TradingBook`, `Desk`.

**Inference** — `Configs(input=ir, semantic_inferrer=..., samples=...)`, or CLI
`--infer-semantics` / `--samples`:

| Strategy | Basis | Needs |
|---|---|---|
| `heuristic` | column-name patterns | nothing (dependency-free) |
| `values` | sample values (vocab membership, ISIN/CUSIP/SEDOL checksum); name fallback | `samples=` (dict of frames/paths, or a directory of `<table>.parquet`/`.csv`) |
| `agent` | LLM (Claude) for ambiguous columns | `pip install 'datagenic[agent]'` + `ANTHROPIC_API_KEY` |
| `module:function` | your own `fn(table, columns, samples=None) -> {col: {semantic, confidence, parent, ...}}` | — |

An explicit `"semantic"` on an IR column **overrides** inference. Each inferred column gets an
`_inferred` block (semantic, confidence, source) beside `column_generator` — ignored by the
engine, surfaced by `datagenic review-config config.json` (lowest confidence first) so you can
fine-tune. The inferrer never hand-edits the config beyond the generator choice; everything
stays editable.

**Cardinality on fixed-domain categoricals.** A finance categorical has a small fixed
vocabulary, so it declares a `cardinality()` method: the engine consults it when distributing
the per-key cardinality budget across unicity keys (generalising the datetime special case to
any generator that defines `cardinality()`), and the generator honors an assigned `cardinality`
as a **cap** (it slices its vocab; it cannot exceed the domain — no fabricated values). As a
unicity key, pair it with a high-cardinality key (an id) — a lone small-domain key can't reach a
large row count, and generation warns with the achievable ceiling rather than silently dedup-
collapsing the table.

## `number_of_rows` as a dynamic reference

```json
"number_of_rows": {"from_table": "products", "multiplier": 1.0}
```

Sized from the referenced table's per-batch row count × `multiplier`. The referenced table must
appear **earlier** in the config list (validation enforces this).

## FK enforcement block

`Configs.dump()` builds `default_parameters.postprocessor_args.fk_enforcement` from the IR's
`foreign_keys`, and wires a `KeyEnforcer` post-processor onto each fact table so its FK columns
draw only from the referenced table's value pool:

```json
"fk_enforcement": {
  "enabled": true,
  "trade_fact": { "srcCurrency": ["fx_dim.srcCurrency"], "csaId": ["csa_dim.csaId"] }
}
```

## `udf_args` naming convention

- Keys starting with `_` are **reserved** for framework-injected args and are rejected by
  `validate()` — **except** `_input_tables`, the one framework-consumed key you set in
  `udf_args` to declare which tables to inject as `_input_tables`.
- Declare upstream tables a post-processor reads via `udf_args._input_tables: ["a", "b"]`; they
  must appear **earlier** in the config (validation enforces this).
