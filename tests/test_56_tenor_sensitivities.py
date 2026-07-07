"""Tenor-bucketed sensitivities — a risk ladder aligned to a tenor grid.

`TenorSensitivities` emits an array(float) aligned to a tenor grid where each bucket is a
signed value scaled by the row's asset_class, the shape is a *bullet* (single maturity
bucket) for zero-coupon-like instruments and a *ladder* (all live buckets) for
swaps/options, and every bucket beyond the instrument's remaining maturity (snapped to the
nearest grid bucket) is zero — recomputed per as-of date via recall.
"""

import numpy as np
import polars as pl

import datagenic.market_finance.generators as gf
from datagenic import Configs, Generator
from datagenic.market_finance import _data as fd
from datagenic.semantics import heuristic_inferrer

_GRID = fd.DEFAULT_TENOR_GRID
_GRID_DAYS = np.array([gf._tenor_to_days(t) for t in _GRID])


def _ladder(
    asset_class, instrument, maturity, asof="2026-07-07", seed=0, n=None, grid=None, sensi=None
):
    """Run TenorSensitivities for hand-built rows and return (n, L) sensis + the grid."""
    acs = asset_class if isinstance(asset_class, list) else [asset_class]
    insts = instrument if isinstance(instrument, list) else [instrument]
    mats = maturity if isinstance(maturity, list) else [maturity]
    n = n or len(acs)
    data = {
        "asset_class": acs,
        "instrument_type": insts,
        "maturity_date": np.array(mats, dtype="datetime64[D]"),
    }
    if sensi is not None:
        data["sensi_type"] = sensi if isinstance(sensi, list) else [sensi] * n
    out = gf.TenorSensitivities().call(
        _number_of_rows_per_batch=n,
        _date=asof,
        tenor_grid=grid,
        _table=pl.DataFrame(data),
        _rng=np.random.default_rng(seed),
    )
    return np.array(out)


# ── shape: bullet vs ladder ────────────────────────────────────────────────────


def test_bullet_instrument_has_one_bucket_at_maturity():
    # an FX forward maturing in ~1M is a single-cashflow (bullet): one non-zero bucket, at 1M
    s = _ladder("FX", "FX Forward", "2026-08-06")[0]  # ~30 days from 2026-07-07
    nz = np.nonzero(s)[0]
    assert len(nz) == 1
    assert _GRID[nz[0]] == "1M"


def test_ladder_instrument_spreads_across_live_buckets():
    # a 5Y IR swap is a term-structure product: non-zero across every bucket up to 5Y
    s = _ladder("Rates", "IR Swap", "2031-07-07")[0]  # ~5Y
    nz = {_GRID[i] for i in np.nonzero(s)[0]}
    assert {"ON", "1M", "1Y", "5Y"} <= nz  # spread, not a single point
    assert len(nz) > 5


# ── shape keyed on sensi_type (spot vs curve risk) ─────────────────────────────


def test_spot_sensi_collapses_to_a_single_short_end_bucket():
    # equity/FX delta is a spot-price risk — a point, not a tenor ladder
    for ac, inst in [("Equity", "Equity Option"), ("FX", "FX Forward")]:
        s = _ladder(ac, inst, "2029-07-07", sensi="delta")[0]
        nz = np.nonzero(s)[0]
        assert len(nz) == 1 and _GRID[nz[0]] == "ON"  # one bucket, at the short (spot) end


def test_curve_sensi_stays_a_ladder_even_for_equity():
    # equity VEGA has a term structure (vol by expiry) -> a ladder, not a point
    s = _ladder("Equity", "Equity Option", "2029-07-07", sensi="vega")[0]
    assert np.count_nonzero(s) > 5


def test_same_instrument_different_sensi_type_differs():
    # an FX forward's delta is a spot point, its rho is a rates bullet at maturity
    delta = _ladder("FX", "FX Forward", "2026-10-07", sensi="delta")[0]
    rho = _ladder("FX", "FX Forward", "2026-10-07", sensi="rho")[0]
    assert _GRID[np.nonzero(delta)[0][0]] == "ON"  # spot
    assert _GRID[np.nonzero(rho)[0][0]] == "3M"  # bullet at the ~3M maturity


def test_no_sensi_type_falls_back_to_instrument_shape():
    # without a sensi_type the spot rule can't fire -> instrument shape (ladder here)
    assert np.count_nonzero(_ladder("Equity", "Equity Option", "2029-07-07")[0]) > 5


# ── the maturity cap (zero beyond maturity, nearest bucket) ────────────────────


def test_zero_beyond_maturity_snapped_to_nearest_bucket():
    # maturing in ~1M -> 1M kept, everything strictly beyond 1M is zero (your spec)
    s = _ladder("Rates", "IR Swap", "2026-08-05")[0]  # ~29 days -> nearest bucket 1M
    one_month = gf._tenor_to_days("1M")
    beyond = s[one_month < _GRID_DAYS]
    assert np.all(beyond == 0.0)
    assert s[_GRID.index("1M")] != 0.0  # the 1M bucket itself is live


def test_matured_trade_is_all_zero():
    s = _ladder("Credit", "CDS", "2026-06-01")[0]  # matured before the as-of
    assert np.all(s == 0.0)


def test_live_region_shrinks_as_asof_advances():
    # fixed maturity, later as-of -> fewer (or equal) live buckets (recall recomputes)
    args = dict(asset_class="Rates", instrument="IR Swap", maturity="2027-07-07")
    early = np.count_nonzero(_ladder(**args, asof="2026-07-07")[0])
    late = np.count_nonzero(_ladder(**args, asof="2027-01-07")[0])
    assert 0 < late < early


# ── magnitude scales by asset class ────────────────────────────────────────────


def test_magnitude_scales_by_asset_class():
    # same long maturity + ladder shape; Rates buckets are an order larger than Equity
    rates = _ladder(["Rates"] * 200, ["IR Swap"] * 200, ["2031-07-07"] * 200, seed=1)
    equity = _ladder(["Equity"] * 200, ["Equity Swap"] * 200, ["2031-07-07"] * 200, seed=1)
    assert np.abs(rates[rates != 0]).mean() > 3 * np.abs(equity[equity != 0]).mean()


# ── grid + determinism ─────────────────────────────────────────────────────────


def test_array_length_follows_grid_and_override():
    assert _ladder("Rates", "IR Swap", "2031-07-07").shape[1] == len(_GRID)
    custom = ["ON", "1M", "1Y", "10Y"]
    assert _ladder("Rates", "IR Swap", "2031-07-07", grid=custom).shape[1] == len(custom)


def test_deterministic():
    a = _ladder("Rates", "IR Swap", "2031-07-07", seed=7)
    b = _ladder("Rates", "IR Swap", "2031-07-07", seed=7)
    assert np.array_equal(a, b)


def test_sensi_type_vocab():
    out = gf.SensiType().call(_number_of_rows_per_batch=50, _rng=np.random.default_rng(0))
    assert set(out) <= set(fd.SENSI_TYPES)
    assert gf.SensiType().cardinality() == len(fd.SENSI_TYPES)


# ── inference wiring + ordering + end-to-end ───────────────────────────────────


def test_inference_maps_and_orders_ladder_after_maturity():
    cols = [
        {"name": "asset_class", "type": "string"},
        {"name": "instrument_type", "type": "string"},
        {"name": "trade_date", "type": "date"},
        {"name": "maturity_date", "type": "date"},
        {"name": "sensi_type", "type": "string"},
        {"name": "sensis", "type": "array(float)"},
    ]
    inf = heuristic_inferrer("s", cols)
    assert inf["sensis"]["semantic"] == "tenor_sensitivities"
    assert inf["sensis"]["parent"] == "maturity_date"  # ordered after maturity_date
    assert inf["sensi_type"]["semantic"] == "sensi_type"


def _sensi_ir():
    return {
        "name": "risk",
        "tables": [
            {
                "type": "fact",
                "name": "sensis_tbl",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "asset_class", "type": "string"},
                    {"name": "instrument_type", "type": "string"},
                    {"name": "tenor", "type": "string"},
                    {"name": "trade_date", "type": "date"},
                    {"name": "maturity_date", "type": "date"},
                    {"name": "sensi_type", "type": "string"},
                    {"name": "sensis", "type": "array(float)"},
                ],
            }
        ],
    }


def test_end_to_end_zero_beyond_maturity(tmp_path):
    cfg = Configs(
        input=_sensi_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=300,
        number_of_batches=1,
        start_date="2026-07-07",
        end_date="2026-07-07",
        semantic_inferrer="heuristic",
    ).config_template
    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("maturity_date") < order.index("sensis")

    Generator(data_generator_config_file=cfg).generate()
    df = pl.read_parquet(next((tmp_path / "out" / "sensis_tbl").rglob("*.parquet")))
    assert len(df["sensis"][0]) == len(_GRID)  # array aligned to the grid

    asof = np.datetime64("2026-07-07", "D")
    for row in df.iter_rows(named=True):
        rem = (np.datetime64(row["maturity_date"], "D") - asof).astype(int)
        mat_bucket = _GRID_DAYS[np.abs(_GRID_DAYS - rem).argmin()] if rem > 0 else -1
        s = np.array(row["sensis"])
        assert np.all(s[mat_bucket < _GRID_DAYS] == 0.0)  # no risk beyond maturity, every row
