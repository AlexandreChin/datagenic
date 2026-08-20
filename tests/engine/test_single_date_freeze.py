"""A single_date table is frozen at start_date — including its columns.

``single_date`` means "generated on start_date, reused on later dates". The
postprocessor and the dump have always honoured that, but column ``recall()``
did not: a date-aware generator kept advancing the table in memory while only
the start_date frame was ever written, so consumers were fed values no
persisted file contains.
"""

from pathlib import Path

import polars as pl

from datagenic import Generator

DATES = ["2024-01-01", "2024-01-02", "2024-01-03"]


def _cfg(out, *, single_date):
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": str(out),
            "tables_prefix": "",
            "start_date": DATES[0],
            "end_date": DATES[-1],
            "number_of_batches": 1,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "dim",
                "number_of_rows": 2,
                "columns": [
                    {
                        "col_name": "asof",
                        "col_type": "date",
                        "is_unicity_key": False,
                        "column_generator": {
                            "udf_name": "CurrentAsofDates",  # advances on recall()
                            "udf_module": "datagenic.generators",
                            "args": {},
                        },
                    }
                ],
                "post_processor": {},
                "single_batch": True,
                "single_date": single_date,
                "dump_results": True,
            },
            {
                "table_name": "fact",
                "number_of_rows": 1,
                "columns": [],
                "post_processor": {
                    "udf_module": "postprocessors_udfs.generic_udfs",
                    "udf_name": "DateEchoPostProcessor",
                    "udf_args": {"_input_tables": ["dim"], "column": "asof"},
                },
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            },
        ],
    }


def _seen(out):
    frames = [pl.read_parquet(p) for p in sorted((Path(out) / "fact").rglob("*.parquet"))]
    return pl.concat(frames).sort("date")


def test_single_date_columns_do_not_advance(tmp_path):
    """Every date's consumer sees start_date's frame — what the dim's one dumped
    partition actually holds."""
    Generator(_cfg(tmp_path, single_date=True)).generate(force_regenerate=True)

    seen = _seen(tmp_path)
    assert seen["date"].to_list() == DATES
    assert seen["seen"].to_list() == [f"{DATES[0]} 00:00:00"] * len(DATES)

    # ...and the dim itself was written once, for start_date only.
    assert [p.name for p in sorted((tmp_path / "dim").iterdir())] == [DATES[0]]


def test_non_single_date_columns_still_advance(tmp_path):
    """The freeze is the flag's doing, not a lost capability: without it the same
    column tracks the date as before."""
    Generator(_cfg(tmp_path, single_date=False)).generate(force_regenerate=True)

    assert _seen(tmp_path)["seen"].to_list() == [f"{d} 00:00:00" for d in DATES]
