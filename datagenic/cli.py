"""datagenic CLI — generation only.

Commands: generate-data, generate-data-config, check-distributable.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="datagenic",
        description="datagenic — synthetic data generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate-data ────────────────────────────────────────────────────────
    gen_parser = sub.add_parser("generate-data", help="Generate synthetic data")
    gen_parser.add_argument("config", help="Path to generator config JSON")
    gen_parser.add_argument(
        "--shard",
        type=int,
        default=1,
        help="This worker's index in a sharded run (1-based: 1..N). Generates a "
        "disjoint batch slice; the union of all shards equals a single run.",
    )
    gen_parser.add_argument(
        "--of",
        type=int,
        default=1,
        help="Total number of shards/workers (default 1 = no sharding, today's run).",
    )
    gen_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Generate on N local processes (single-machine multicore). Output is "
        "byte-identical to a sequential run. Mutually exclusive with --shard/--of.",
    )
    gen_parser.add_argument(
        "--tables",
        metavar="A,B,…",
        help="Run only a job: the given root table(s) plus their full transitive "
        "upstream dependency closure. Tables outside the closure are not generated.",
    )

    check_parser = sub.add_parser(
        "check-distributable",
        help="Check whether a generator config can be sharded safely",
    )
    check_parser.add_argument("config", help="Path to generator config JSON")

    gen_config_parser = sub.add_parser(
        "generate-data-config",
        help="Create a generator config from an IR (or any JSON via --adaptor)",
    )
    gen_config_parser.add_argument(
        "input",
        help="Path to an IR JSON file, or an inline JSON object string. To scaffold "
        "from a non-IR schema, pass --adaptor.",
    )
    gen_config_parser.add_argument("-o", "--output", required=True, help="Output config file path")
    gen_config_parser.add_argument(
        "--storage-type", default="local", help="Storage type (default: local)"
    )
    gen_config_parser.add_argument(
        "--format", default="parquet", dest="output_format", help="Output format (default: parquet)"
    )
    gen_config_parser.add_argument(
        "--output-folder", default="samples/data_generator/output", help="Folder for generated data"
    )
    gen_config_parser.add_argument(
        "--rows-fact", type=int, default=2500, help="Rows per fact table (default: 2500)"
    )
    gen_config_parser.add_argument(
        "--rows-dict", type=int, default=100, help="Rows per non-fact table (default: 100)"
    )
    gen_config_parser.add_argument(
        "--batches", type=int, default=2, help="Number of batches (default: 2)"
    )
    gen_config_parser.add_argument("--offset-batches", type=int, default=0)
    gen_config_parser.add_argument("--start-date", default="2024-01-01")
    gen_config_parser.add_argument("--end-date", default="2024-01-02")
    gen_config_parser.add_argument("--seed", type=int, default=0)
    gen_config_parser.add_argument(
        "--no-fk",
        dest="enforce_fk",
        action="store_false",
        help="Disable foreign-key enforcement on fact tables",
    )
    gen_config_parser.add_argument(
        "--no-postprocessor-templates",
        dest="template_postprocessors",
        action="store_false",
        help="Do not wire a no-op postprocessor template onto each table",
    )
    gen_config_parser.add_argument(
        "--adaptor",
        metavar="module:function",
        help="Map a non-IR input JSON to the IR, e.g. mypkg.adaptors:my_adaptor. "
        "Omit when input is already IR.",
    )
    gen_config_parser.add_argument(
        "--infer-semantics",
        dest="infer_semantics",
        metavar="heuristic|values|agent|module:function",
        default=None,
        help="Auto-detect each column's domain meaning during scaffolding and wire "
        "a realistic generator (e.g. financial-markets asset class / instrument / "
        "ISIN). 'heuristic' uses column-name rules; 'module:function' supplies a "
        "custom inferrer. Inferred columns are annotated for manual review "
        "(see 'datagenic review-config'). Omit to disable.",
    )
    gen_config_parser.add_argument(
        "--samples",
        metavar="DIR | {table:path}",
        help="Sample data for value-based inference ('--infer-semantics values'): a "
        "directory holding <table>.parquet/<table>.csv per table, or an inline/"
        "file JSON map of table->path. Lets the inferrer read real values.",
    )

    review_parser = sub.add_parser(
        "review-config",
        help="List columns whose generator was inferred during scaffolding "
        "(lowest confidence first), so you can fine-tune the config by hand",
    )
    review_parser.add_argument("config", help="Path to generator config JSON")

    args = parser.parse_args()

    # ── Dispatch ───────────────────────────────────────────────────────────
    if args.command == "generate-data":
        from datagenic import orchestration

        tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None

        if args.parallel > 1:
            if args.of > 1:
                parser.error(
                    "--parallel (single-machine multicore) is mutually exclusive with --of"
                )
            orchestration.run_parallel(args.config, args.parallel, tables=tables)
        else:
            orchestration.run_shard(args.config, shard=args.shard, of=args.of, tables=tables)

    elif args.command == "check-distributable":
        from datagenic import orchestration

        verdict = orchestration.check_distributable(args.config)
        if verdict["ok"]:
            print("OK — config is distributable (shardable).")
        else:
            print("NOT distributable:")
            for failure in verdict["failures"]:
                print(f"  - {failure}")
            sys.exit(1)

    elif args.command == "generate-data-config":
        import importlib
        import json

        from datagenic import Configs

        # accept the source as a file path or an inline JSON object string
        source = args.input
        if source.lstrip().startswith("{"):
            source = json.loads(source)

        # optional adaptor "module:function" to map a non-IR source to the IR
        adaptor = None
        if args.adaptor:
            module_name, _, func_name = args.adaptor.partition(":")
            if not func_name:
                parser.error("--adaptor must be 'module:function'")
            adaptor = getattr(importlib.import_module(module_name), func_name)

        # samples: a directory string, an inline JSON map, or a .json map file
        samples = args.samples
        if samples and samples.lstrip().startswith("{"):
            samples = json.loads(samples)
        elif samples and samples.endswith(".json"):
            with open(samples) as f:
                samples = json.load(f)

        Configs(
            input=source,
            adaptor=adaptor,
            storage_type=args.storage_type,
            output_format=args.output_format,
            output_folder_path=args.output_folder,
            number_of_rows_in_fact_tables=args.rows_fact,
            number_of_rows_in_non_fact_tables=args.rows_dict,
            number_of_batches=args.batches,
            offset_batches=args.offset_batches,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
            enforce_fk=args.enforce_fk,
            template_postprocessors=args.template_postprocessors,
            semantic_inferrer=args.infer_semantics,
            samples=samples,
        ).dump(file_path=args.output)

    elif args.command == "review-config":
        import json

        with open(args.config) as f:
            config = json.load(f)

        rows = []
        for table in config.get("tables", []):
            for col in table.get("columns", []):
                inferred = col.get("_inferred")
                if inferred:
                    rows.append((table["table_name"], col["col_name"], inferred))

        if not rows:
            print("No inferred columns in this config (scaffolded without --infer-semantics).")
        else:
            rows.sort(key=lambda r: r[2].get("confidence", 0.0))
            print(f"{len(rows)} inferred column(s), lowest confidence first:")
            for table_name, col_name, inferred in rows:
                parent = f", parent={inferred['parent']}" if inferred.get("parent") else ""
                alts = (
                    f", alternatives={inferred['alternatives']}"
                    if inferred.get("alternatives")
                    else ""
                )
                print(
                    f"  [{inferred.get('confidence', 0.0):.2f}] {table_name}.{col_name} "
                    f"-> {inferred['semantic']} (by {inferred['by']}{parent}{alts})"
                )


if __name__ == "__main__":
    main()
