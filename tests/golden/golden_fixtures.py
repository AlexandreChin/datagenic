"""Single source of truth for the golden-baseline configs (G1).

Both ``capture_baseline.py`` and ``test_2_safety_net.py`` use these so the
captured manifest and the assertion run the exact same configs. Configs are
deterministic (fixed seed) and self-contained (no external file dependencies).

The ``dms`` fixture scaffolds from a frozen *neutral IR* (``samples/ir/solutions_ir.json``).
That IR is the exact output the Opensee ``dms_to_ir`` adaptor produced for the sample
DMS — frozen here so ``datagenic`` needs no platform dependency. ``Configs(IR)`` is
byte-identical to ``Configs(dms, adaptor=dms_to_ir)``, so the baseline hashes are
unchanged from when the adaptor produced them.
"""

import copy
import hashlib
import json
from pathlib import Path

import polars as pl

from datagenic import Columns, Configs, Generator

REPO_ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS_IR = REPO_ROOT / "samples" / "ir" / "solutions_ir.json"


def _col(table, name, col_type, is_key, n_keys, rows):
    c = Columns(
        table,
        name,
        col_type,
        is_key,
        n_keys,
        number_of_rows=rows,
        _number_of_rows_per_batch=rows,
    )
    return {
        "col_name": name,
        "col_type": col_type,
        "is_unicity_key": is_key,
        "column_generator": {
            "udf_name": c.default_generator_name,
            "udf_module": "datagenic.generators",
            "args": c.default_arguments,
            "description": c.default_generator_description,
        },
    }


def _rng_config():
    rows = 200
    return {
        "default_parameters": {
            "output_format": "parquet",
            "storage_type": "local",
            "output_folder_path": "",
            "tables_prefix": "",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "number_of_batches": 2,
            "offset_batches": 0,
            "postprocessor_args": {},
            "seed": 0,
        },
        "tables": [
            {
                "table_name": "t_rng",
                "number_of_rows": rows,
                "post_processor": {},
                "columns": [
                    _col("t_rng", "id", "int", True, 1, rows),
                    _col("t_rng", "val", "float", False, 1, rows),
                ],
                "single_batch": False,
                "single_date": False,
                "dump_results": True,
            }
        ],
    }


def _dms_config():
    """Full sample column-generator config (small, deterministic).

    Exercises the real generator set — composite unicity keys, dates,
    datetimes, arrays — by scaffolding from the frozen sample IR (the native
    datagenic input; no adaptor needed).
    """
    ir = json.loads(SOLUTIONS_IR.read_text())
    return Configs(
        input=ir,
        output_format="parquet",
        storage_type="local",
        number_of_rows_in_fact_tables=100,
        number_of_rows_in_non_fact_tables=30,
        number_of_batches=2,
        start_date="2024-01-01",
        end_date="2024-01-02",
        seed=0,
    ).config_template


def fixtures():
    """Named reference configs included in the golden baseline."""
    return {"rng": _rng_config(), "dms": _dms_config()}


def run(cfg, out_dir):
    cfg = copy.deepcopy(cfg)
    cfg["default_parameters"]["output_folder_path"] = str(out_dir)
    Generator(data_generator_config_file=cfg).generate()


def _logical_digest(path):
    """Hash a parquet file's *logical content* (schema + values), not its raw bytes.

    The baseline is committed and re-checked across time, so it must be stable under
    pyarrow/python upgrades. Raw parquet bytes are not: every file's footer embeds the
    writer version (``parquet-cpp-arrow version X.Y.Z``) and encoding defaults shift
    between Arrow releases, so hashing bytes flags a mismatch on every toolchain bump
    even when the generated data is identical. Hashing the schema (column names +
    dtypes) and the values in row order tests what the project actually guarantees —
    deterministic *data* — independent of how Arrow happens to serialise it.
    """
    df = pl.read_parquet(path)
    schema = [(name, str(dtype)) for name, dtype in df.schema.items()]
    payload = repr(schema) + "|" + repr(df.to_dict(as_series=False))
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_tree(root):
    root = Path(root)
    return {str(p.relative_to(root)): _logical_digest(p) for p in sorted(root.rglob("*.parquet"))}
