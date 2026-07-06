"""Flat array-quantity semantics + scalar/array type gate (Phase 2 of array values).

Inference maps array(float) columns whose name matches a series/curve quantity
(pnl_series, returns, curve) to a distribution-aware array generator wired with a
book-agnostic prior. The column's *type* — not its name — decides scalar vs array:
`pnl` as float is a signed scalar, `pnl` as array(float) is a time series, and they
never cross-wire.
"""

import polars as pl

import datagenic.market_finance._data as fd
from datagenic import Configs, Generator
from datagenic.semantics import SEMANTIC_GENERATORS, build_generator, heuristic_inferrer


def _infer(name, col_type):
    rec = heuristic_inferrer("t", [{"name": name, "type": col_type}]).get(name)
    return rec["semantic"] if rec else None


# ── array names map to array semantics ────────────────────────────────────────


def test_array_names_map_to_array_semantics():
    assert _infer("pnl", "array(float)") == "pnl_series"
    assert _infer("pnl_vector", "array(float)") == "pnl_series"
    assert _infer("daily_returns", "array(float)") == "returns"
    assert _infer("yield_curve", "array(float)") == "curve"


# ── the scalar/array type gate ────────────────────────────────────────────────


def test_type_decides_scalar_vs_array_for_same_name():
    # 'pnl' resolves by type: scalar float -> signed scalar; array(float) -> series
    assert _infer("pnl", "float") == "pnl"
    assert _infer("pnl", "array(float)") == "pnl_series"


def test_scalar_semantic_does_not_grab_array_column():
    # 'notional' is a scalar-only semantic; on an array(float) column it must NOT
    # be wired to the scalar ConditionalFloats (no array pattern matches it either)
    assert _infer("notional", "array(float)") is None
    assert _infer("notional", "float") == "notional"


def test_array_semantic_does_not_grab_scalar_column():
    # a series/curve name on a scalar float column never yields an *array* semantic
    # (it may match a scalar one — e.g. "yield_curve" -> the scalar `rate` via "yield")
    array_semantics = set(fd.ARRAY_QUANTITY_PRIORS)
    assert _infer("daily_returns", "float") not in array_semantics
    assert _infer("yield_curve", "float") not in array_semantics
    assert _infer("pnl_series", "float") not in array_semantics


# ── priors → generator specs ──────────────────────────────────────────────────


def test_every_array_prior_has_a_semantic_entry():
    for semantic, prior in fd.ARRAY_QUANTITY_PRIORS.items():
        assert semantic in SEMANTIC_GENERATORS
        module, cls, base_args, parent_arg = SEMANTIC_GENERATORS[semantic]
        assert module == "datagenic.generators"
        assert cls == prior["generator"]
        assert parent_arg is None
        assert set(base_args) == {"distribution", "mu", "sigma", "size_of_arrays", "path"}


def test_scalar_pnl_is_signed_normal():
    cg = build_generator("pnl")
    assert cg["udf_name"] == "RandomFloats"
    assert cg["args"]["distribution"] == "normal"
    assert cg["args"]["mu"] == 0.0  # signed, mean ~0 — not the positive `pv` prior


def test_build_generator_wires_flat_array_prior():
    # `returns` is a flat (non-conditional) array quantity; pnl_series is conditional (test_48)
    cg = build_generator("returns")
    assert cg["udf_name"] == "RandomArrayFloats"
    assert cg["args"]["path"] == "iid"
    assert cg["args"]["size_of_arrays"] == 250


# ── scaffolding integration + end-to-end ──────────────────────────────────────


def _returns_ir():
    return {
        "name": "risk",
        "tables": [
            {
                "type": "fact",
                "name": "positions",
                "keys": ["pos_id"],
                "columns": [
                    {"name": "pos_id", "type": "int"},
                    {"name": "daily_returns", "type": "array(float)"},
                ],
            }
        ],
    }


def test_scaffold_wires_array_generator_and_annotates():
    template = Configs(
        input=_returns_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    col = {c["col_name"]: c for c in template["tables"][0]["columns"]}["daily_returns"]
    assert col["column_generator"]["udf_name"] == "RandomArrayFloats"
    assert col["column_generator"]["args"]["size_of_arrays"] == 250
    assert col["_inferred"]["semantic"] == "returns"


def test_end_to_end_returns_is_an_array_series(tmp_path):
    cfg = Configs(
        input=_returns_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=100,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "positions").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])
    first = df["daily_returns"].to_list()[0]
    assert isinstance(first, list) and len(first) == 250  # a per-row series
