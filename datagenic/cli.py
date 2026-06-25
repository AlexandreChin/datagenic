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
        ).dump(file_path=args.output)


if __name__ == "__main__":
    main()
