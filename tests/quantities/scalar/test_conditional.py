"""Conditional (parent-aware) numeric fact values (Phase 3 of conditional numerics).

``ConditionalFloats`` draws each row's value from a per-category distribution read
off another column (``by_column``) in the partial table — the numeric twin of the
coherent ``asset_class -> instrument_type`` taxonomy. Inference maps ``notional``
to it, keyed on the row's ``asset_class`` (an Equity notional ≈ $100k, a Rates
notional ≈ tens of millions). Falls back to a flat ``default`` when the parent
column is absent or the category is unmapped.
"""

import numpy as np
import polars as pl

import datagenic.domains.market_finance._data as fd
from datagenic import Configs, Generator
from datagenic.generators import ConditionalFloats
from datagenic.semantics import SEMANTIC_GENERATORS, build_generator, heuristic_inferrer

_CASES = {
    "Equity": {"distribution": "lognormal", "mu": 1.0e5, "sigma": 0.8},
    "Rates": {"distribution": "lognormal", "mu": 2.5e7, "sigma": 1.0},
}
_DEFAULT = {"distribution": "lognormal", "mu": 1.0e6, "sigma": 1.0}


def _call(parent_values, rng, **over):
    table = pl.DataFrame({"asset_class": parent_values})
    kw = dict(by_column="asset_class", cases=_CASES, default=_DEFAULT)
    kw.update(over)
    return ConditionalFloats().call(
        _number_of_rows_per_batch=len(parent_values), _table=table, _rng=rng, **kw
    )


# ── generator: per-row distribution matches the row's category ────────────────


def test_value_scale_matches_row_category():
    parent = np.array(["Equity"] * 4000 + ["Rates"] * 4000)
    out = _call(parent, np.random.default_rng(0))
    assert (out > 0).all()  # lognormal is positive
    equity = out[parent == "Equity"]
    rates = out[parent == "Rates"]
    assert 0.5e5 < float(np.median(equity)) < 2.0e5  # ≈ mu for Equity
    assert 1.5e7 < float(np.median(rates)) < 4.0e7  # ≈ mu for Rates
    # ~2 orders of magnitude apart, coherent with the category
    assert float(np.median(rates)) / float(np.median(equity)) > 50


def test_pv_is_asset_class_conditional_per_unit():
    # pv/mtm is a per-unit-of-instrument valuation, now conditioned on asset_class
    # (like notional) rather than a single flat prior.
    inferred = heuristic_inferrer(
        "trades",
        [{"name": "asset_class", "type": "string"}, {"name": "mtm", "type": "float"}],
    )
    assert inferred["mtm"]["semantic"] == "pv"
    assert inferred["mtm"]["parent"] == "asset_class"  # linked to the category column

    spec = build_generator("pv", parent="asset_class")
    assert spec["udf_name"] == "ConditionalFloats"
    assert spec["args"]["by_column"] == "asset_class"
    cases = spec["args"]["cases"]
    assert {"Equity", "Rates", "FX"} <= set(cases)
    # per unit of instrument: price-like magnitudes, NOT notional-scale (millions)
    assert all(c["mu"] < 1.0e4 for c in cases.values())


def test_unmapped_category_uses_default():
    parent = np.array(["Commodity"] * 4000)  # not in _CASES
    out = _call(parent, np.random.default_rng(0))
    assert 0.5e6 < float(np.median(out)) < 2.0e6  # ≈ default median


def test_absent_parent_column_falls_back_to_default():
    out = ConditionalFloats().call(
        _number_of_rows_per_batch=100,
        by_column="asset_class",
        cases=_CASES,
        default=_DEFAULT,
        _table=pl.DataFrame({}),  # no by_column
        _rng=np.random.default_rng(0),
    )
    assert len(out) == 100
    assert (out > 0).all()


def test_is_deterministic():
    parent = np.array(["Equity", "Rates"] * 500)
    a = _call(parent, np.random.default_rng(7))
    b = _call(parent, np.random.default_rng(7))
    assert np.array_equal(a, b)


# ── inference: notional -> ConditionalFloats, linked to asset_class ───────────


def test_notional_maps_to_conditional_floats():
    module, cls, base_args, parent_arg = SEMANTIC_GENERATORS["notional"]
    assert (module, cls, parent_arg) == ("datagenic.generators", "ConditionalFloats", "by_column")
    assert set(base_args) == {"cases", "default"}


def test_inference_links_notional_to_asset_class():
    columns = [
        {"name": "asset_class", "type": "string"},
        {"name": "notional", "type": "float"},
    ]
    inferred = heuristic_inferrer("trades", columns)
    assert inferred["notional"]["semantic"] == "notional"
    assert inferred["notional"]["parent"] == "asset_class"

    cg = build_generator("notional", "asset_class")
    assert cg["udf_name"] == "ConditionalFloats"
    assert cg["args"]["by_column"] == "asset_class"
    assert cg["args"]["cases"] == fd.CONDITIONAL_QUANTITY_PRIORS["notional"]["cases"]


def _finance_ir():
    # notional listed before asset_class to exercise coherence reordering
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
                    {"name": "asset_class", "type": "string"},
                ],
            }
        ],
    }


def test_scaffold_orders_asset_class_before_notional():
    template = Configs(
        input=_finance_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    table = template["tables"][0]
    order = [c["col_name"] for c in table["columns"]]
    assert order.index("asset_class") < order.index("notional")

    notional = {c["col_name"]: c for c in table["columns"]}["notional"]
    assert notional["column_generator"]["udf_name"] == "ConditionalFloats"
    assert notional["column_generator"]["args"]["by_column"] == "asset_class"
    assert notional["_inferred"]["parent"] == "asset_class"


# ── end-to-end: notional scaled by asset_class, positive, coherent ────────────


def test_end_to_end_notional_scaled_by_asset_class(tmp_path):
    cfg = Configs(
        input=_finance_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=8000,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])

    assert (df["notional"].to_numpy() > 0).all()
    medians = (
        df.group_by("asset_class")
        .agg(pl.col("notional").median().alias("m"))
        .to_dict(as_series=False)
    )
    by_class = dict(zip(medians["asset_class"], medians["m"], strict=True))
    # Equity and Rates both present and ~2 orders of magnitude apart
    assert "Equity" in by_class and "Rates" in by_class
    assert by_class["Rates"] / by_class["Equity"] > 20
