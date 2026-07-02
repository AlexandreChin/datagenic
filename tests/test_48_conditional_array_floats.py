"""Conditional (parent-aware) array-quantity values (Phase 3 of array values).

`ConditionalArrayFloats` draws each row's array from a per-category distribution
read off a parent column — the array twin of `ConditionalFloats`. Inference maps a
`pnl` array(float) column to it, keyed on `asset_class` (a Rates book's PnL path
swings wider than an Equity book's). Falls back to `default` when the parent is
absent or the category unmapped.
"""

import numpy as np
import polars as pl

import datagenic._finance_data as fd
from datagenic import Configs, Generator
from datagenic.generators import ConditionalArrayFloats
from datagenic.semantics import SEMANTIC_GENERATORS, build_generator, heuristic_inferrer

_CASES = {
    "Equity": {"distribution": "normal", "mu": 0.0, "sigma": 5.0e4},
    "Rates": {"distribution": "normal", "mu": 0.0, "sigma": 2.0e5},
}
_DEFAULT = {"distribution": "normal", "mu": 0.0, "sigma": 5.0e4}


def _call(parent_values, rng, **over):
    table = pl.DataFrame({"asset_class": parent_values})
    kw = dict(
        size_of_arrays=250, by_column="asset_class", cases=_CASES, default=_DEFAULT, path="walk"
    )
    kw.update(over)
    return np.array(
        ConditionalArrayFloats().call(
            _number_of_rows_per_batch=len(parent_values), _table=table, _rng=rng, **kw
        )
    )


# ── generator: per-row array scale matches the row's category ─────────────────


def test_array_scale_matches_row_category():
    parent = np.array(["Equity"] * 3000 + ["Rates"] * 3000)
    out = _call(parent, np.random.default_rng(0))
    assert out.shape == (6000, 250)
    # a walk's terminal dispersion scales with the per-category sigma (Rates 4x Equity)
    eq = float(np.std(out[parent == "Equity"][:, -1]))
    ra = float(np.std(out[parent == "Rates"][:, -1]))
    assert 2.5 < ra / eq < 6.0


def test_unmapped_category_and_absent_parent_use_default():
    out = _call(np.array(["Commodity"] * 2000), np.random.default_rng(0))  # not in _CASES
    assert out.shape == (2000, 250)
    absent = ConditionalArrayFloats().call(
        size_of_arrays=10,
        _number_of_rows_per_batch=5,
        cases=_CASES,
        default=_DEFAULT,
        _table=pl.DataFrame({}),  # no by_column
        _rng=np.random.default_rng(0),
    )
    assert len(absent) == 5 and len(absent[0]) == 10


def test_is_deterministic():
    parent = np.array(["Equity", "Rates"] * 400)
    a = _call(parent, np.random.default_rng(7))
    b = _call(parent, np.random.default_rng(7))
    assert np.array_equal(a, b)


# ── inference wiring ──────────────────────────────────────────────────────────


def test_pnl_series_maps_to_conditional_array_floats():
    module, cls, base_args, parent_arg = SEMANTIC_GENERATORS["pnl_series"]
    assert (module, cls, parent_arg) == (
        "datagenic.generators",
        "ConditionalArrayFloats",
        "by_column",
    )
    assert set(base_args) == {"size_of_arrays", "path", "cases", "default"}


def test_inference_links_pnl_to_asset_class():
    columns = [{"name": "asset_class", "type": "string"}, {"name": "pnl", "type": "array(float)"}]
    inferred = heuristic_inferrer("positions", columns)
    assert inferred["pnl"]["semantic"] == "pnl_series"
    assert inferred["pnl"]["parent"] == "asset_class"

    cg = build_generator("pnl_series", "asset_class")
    assert cg["udf_name"] == "ConditionalArrayFloats"
    assert cg["args"]["by_column"] == "asset_class"
    assert cg["args"]["cases"] == fd.CONDITIONAL_ARRAY_QUANTITY_PRIORS["pnl_series"]["cases"]


def _pnl_ir():
    # pnl before asset_class to exercise coherence reordering
    return {
        "name": "risk",
        "tables": [
            {
                "type": "fact",
                "name": "positions",
                "keys": ["pos_id"],
                "columns": [
                    {"name": "pos_id", "type": "int"},
                    {"name": "pnl", "type": "array(float)"},
                    {"name": "asset_class", "type": "string"},
                ],
            }
        ],
    }


def test_scaffold_orders_asset_class_before_pnl():
    template = Configs(
        input=_pnl_ir(), output_format="parquet", number_of_batches=1, semantic_inferrer="heuristic"
    ).config_template
    table = template["tables"][0]
    order = [c["col_name"] for c in table["columns"]]
    assert order.index("asset_class") < order.index("pnl")
    pnl = {c["col_name"]: c for c in table["columns"]}["pnl"]
    assert pnl["column_generator"]["udf_name"] == "ConditionalArrayFloats"
    assert pnl["_inferred"]["parent"] == "asset_class"


# ── end-to-end ────────────────────────────────────────────────────────────────


def test_end_to_end_pnl_series_scaled_by_asset_class(tmp_path):
    cfg = Configs(
        input=_pnl_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=4000,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "positions").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])
    arr = np.array(df["pnl"].to_list())
    ac = np.array(df["asset_class"].to_list())
    assert arr.shape[1] == 250  # a per-row series
    eq = float(np.std(arr[ac == "Equity"][:, -1]))
    ra = float(np.std(arr[ac == "Rates"][:, -1]))
    assert ra > eq * 2  # Rates PnL path swings wider than Equity, coherent with the book
