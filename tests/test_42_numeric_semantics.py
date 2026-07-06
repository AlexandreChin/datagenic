"""Flat numeric quantity semantics (Phase 2 of conditional numeric values).

Extends automatic scaffold-time inference to *numbers*: name patterns gated to
float columns map quantities (notional, price, rate, spread, pv, quantity,
fx_rate) to a distribution-aware float generator wired with a book-agnostic
prior, so magnitudes are plausible out of the box (a notional ≈ millions,
lognormal) instead of the generic N(1000, 100).

Here the IR carries no parent category, so ``notional`` (conditional in Phase 3)
falls back to its flat default. The per-category scaling lives in test_43.
"""

import numpy as np
import polars as pl

import datagenic.market_finance._data as fd
from datagenic import Configs, Generator
from datagenic.semantics import (
    SEMANTIC_GENERATORS,
    build_generator,
    heuristic_inferrer,
)

# ── heuristic maps numeric names, gated to float type ─────────────────────────


def test_numeric_names_map_to_quantity_semantics():
    columns = [
        {"name": "notional", "type": "float"},
        {"name": "clean_price", "type": "float"},
        {"name": "credit_spread", "type": "float"},
        {"name": "pv", "type": "float"},
        {"name": "coupon_rate", "type": "float"},
        {"name": "fx_rate", "type": "float"},
    ]
    inferred = heuristic_inferrer("trades", columns)
    assert inferred["notional"]["semantic"] == "notional"
    assert inferred["clean_price"]["semantic"] == "price"
    assert inferred["credit_spread"]["semantic"] == "spread"
    assert inferred["pv"]["semantic"] == "pv"
    assert inferred["coupon_rate"]["semantic"] == "rate"
    # fx_rate is tested before the generic rate pattern
    assert inferred["fx_rate"]["semantic"] == "fx_rate"
    # numeric semantics are flat (no hierarchy parent) in this phase
    assert all(rec["parent"] is None for rec in inferred.values())


def test_numeric_patterns_are_gated_to_float():
    # same names as ints/strings must NOT be matched by the numeric patterns
    columns = [
        {"name": "notional", "type": "int"},  # a key/count, left alone
        {"name": "price", "type": "string"},  # a label, not a quantity
        {"name": "quantity", "type": "int"},
        {"name": "trade_id", "type": "int"},
        {"name": "amount", "type": "float"},  # ambiguous name, no pattern
    ]
    assert heuristic_inferrer("trades", columns) == {}


# ── priors → generator specs ──────────────────────────────────────────────────


def test_every_prior_has_a_semantic_entry():
    for semantic in fd.QUANTITY_PRIORS:
        assert semantic in SEMANTIC_GENERATORS
        module, _cls, base_args, parent_arg = SEMANTIC_GENERATORS[semantic]
        assert module == "datagenic.generators"
        assert parent_arg is None
        assert set(base_args) == {"mu", "sigma", "distribution"}


def test_build_generator_wires_flat_prior():
    cg = build_generator("quantity")  # a stateless, flat quantity
    assert cg["udf_module"] == "datagenic.generators"
    assert cg["udf_name"] == "RandomFloats"
    assert cg["args"]["distribution"] == "lognormal"
    assert cg["args"]["mu"] == fd.QUANTITY_PRIORS["quantity"]["mu"]
    assert cg["args"]["sigma"] == fd.QUANTITY_PRIORS["quantity"]["sigma"]


def test_build_generator_price_evolves():
    # a price walks over dates -> BrownianFloats (has recall)
    assert build_generator("price")["udf_name"] == "BrownianFloats"


# ── scaffolding integration ───────────────────────────────────────────────────


def _quant_ir():
    # no asset_class column -> notional falls back to its flat default
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "notional", "type": "float"},
                    {"name": "clean_price", "type": "float"},
                ],
            }
        ],
    }


def test_scaffold_wires_numeric_generator_and_annotates():
    template = Configs(
        input=_quant_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    cols = {c["col_name"]: c for c in template["tables"][0]["columns"]}

    notional = cols["notional"]["column_generator"]
    assert notional["udf_name"] == "ConditionalFloats"
    assert notional["args"]["default"]["distribution"] == "lognormal"
    assert cols["notional"]["_inferred"]["semantic"] == "notional"
    assert cols["notional"]["_inferred"]["parent"] is None  # no asset_class to link
    assert cols["clean_price"]["column_generator"]["udf_name"] == "BrownianFloats"


def test_end_to_end_notional_falls_back_to_flat_default(tmp_path):
    cfg = Configs(
        input=_quant_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=5000,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])

    notional = df["notional"].to_numpy()
    assert (notional > 0).all()  # lognormal is positive
    # with no parent category, the flat default (median ≈ $1M) applies
    assert 0.5e6 < float(np.median(notional)) < 2.0e6
    assert float(notional.mean()) > float(np.median(notional))  # right-skewed
