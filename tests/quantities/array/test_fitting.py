"""Fitting array-quantity distributions from sample data (Phase 4 of array values).

`values`/`profile` learn an array column's element family + params **and its
length** from the sample — flat, and per-parent-category for a conditional series
(a Rates PnL path fitted wider than an Equity one). `profile` also learns to type a
polars List column as array(float).
"""

import numpy as np
import polars as pl

from datagenic import Generator
from datagenic.profiling import _ir_type, build_config
from datagenic.semantics import fit_array_distribution, values_inferrer


def _series(rng, sigma, n_rows, length):
    return [rng.normal(0.0, sigma, length).tolist() for _ in range(n_rows)]


# ── fit_array_distribution: element params + length ───────────────────────────


def test_fits_element_params_and_length():
    rng = np.random.default_rng(0)
    spec = fit_array_distribution(_series(rng, 0.02, 400, 200))
    assert spec["distribution"] == "normal"  # signed -> normal
    assert abs(spec["mu"]) < 0.01  # mean ~0
    assert 0.017 < spec["sigma"] < 0.023  # recovers the element sigma
    assert spec["size_of_arrays"] == 200  # recovers the array length


def test_fit_array_handles_empty():
    assert fit_array_distribution([]) is None
    assert fit_array_distribution([None, None]) is None


# ── profiling types + fits a List column ──────────────────────────────────────


def test_ir_type_maps_list_to_array_float():
    df = pl.DataFrame({"pnl": [[1.0, 2.0], [3.0, 4.0]]})
    assert _ir_type(df["pnl"].dtype) == "array(float)"


def test_profile_from_sample_fits_flat_array(tmp_path):
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({"id": np.arange(300), "metric": _series(rng, 5.0, 300, 40)})
    config, _ir = build_config(
        {"t": frame}, output_folder_path=str(tmp_path / "out"), number_of_batches=1
    )
    metric = {c["col_name"]: c for c in config["tables"][0]["columns"]}["metric"]
    args = metric["column_generator"]["args"]
    assert args["distribution"] == "normal"
    assert args["size_of_arrays"] == 40  # length learned from the sample
    assert 4.0 < args["sigma"] < 6.0


# ── values inference: conditional per-group array fit ─────────────────────────


def _conditional_pnl(rng, n=3000, length=120):
    ac = rng.permutation(np.array(["Equity"] * (n // 2) + ["Rates"] * (n // 2)))
    pnl = [
        (rng.normal(0, 5e4, length) if a == "Equity" else rng.normal(0, 2e5, length)).tolist()
        for a in ac
    ]
    return {"asset_class": ac.tolist(), "pnl": pnl}


def test_values_inferrer_fits_conditional_array_cases():
    samples = _conditional_pnl(np.random.default_rng(0))
    columns = [{"name": "asset_class", "type": "string"}, {"name": "pnl", "type": "array(float)"}]
    result = values_inferrer("positions", columns, samples=samples)
    args = result["pnl"]["args"]
    assert result["pnl"]["semantic"] == "pnl_series"
    assert result["pnl"]["parent"] == "asset_class"
    assert args["size_of_arrays"] == 120
    # per-class element volatility recovered from the sample (Rates ~4x Equity)
    assert 3.5e4 < args["cases"]["Equity"]["sigma"] < 6.5e4
    assert 1.6e5 < args["cases"]["Rates"]["sigma"] < 2.4e5


def test_end_to_end_profile_from_sample_conditional_pnl(tmp_path):
    rng = np.random.default_rng(0)
    samples = _conditional_pnl(rng, n=3000, length=120)
    frame = pl.DataFrame(
        {
            "pos_id": np.arange(3000),
            "asset_class": samples["asset_class"],
            "pnl": samples["pnl"],
        }
    )
    config, _ir = build_config(
        {"positions": frame},
        semantic_inferrer="values",
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_batches=1,
    )
    pnl = {c["col_name"]: c for c in config["tables"][0]["columns"]}["pnl"]
    cg = pnl["column_generator"]
    assert cg["udf_name"] == "ConditionalArrayFloats"
    assert cg["args"]["size_of_arrays"] == 120
    assert 1.6e5 < cg["args"]["cases"]["Rates"]["sigma"] < 2.4e5

    Generator(data_generator_config_file=config).generate()
    files = list((tmp_path / "out" / "positions").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])
    arr = np.array(df["pnl"].to_list())
    ac = np.array(df["asset_class"].to_list())
    assert arr.shape[1] == 120
    eq = float(np.std(arr[ac == "Equity"][:, -1]))
    ra = float(np.std(arr[ac == "Rates"][:, -1]))
    assert ra > eq * 2  # generated Rates PnL path swings wider, matching the sample
