"""Trade-attribute fields — Tier 1 (categoricals, Greeks, sensitivities) + Tier 2
(coherent dates, counterparty ticker/LEI, underlying).

Covers the new fixed-vocabulary categoricals, the coherent clearing house / underlying
(read asset_class), the coherent counterparty ticker/LEI (read counterparty, LEI carries
a valid ISO 7064 check digit), the tenor-consistent dates (maturity = trade_date + tenor,
settlement = trade_date + lag), and the inference wiring for all of them.
"""

import datetime

import numpy as np
import polars as pl

import datagenic.domains.market_finance._data as fd
import datagenic.domains.market_finance.generators as gf
from datagenic import Configs, Generator
from datagenic.semantics import SEMANTIC_GENERATORS, build_generator, heuristic_inferrer

# ── fixed-vocabulary categoricals ─────────────────────────────────────────────


def test_fixed_vocab_generators_draw_and_report_cardinality():
    for cls, vocab in [
        (gf.Side, fd.SIDES),
        (gf.OptionType, fd.OPTION_TYPES),
        (gf.ExerciseStyle, fd.EXERCISE_STYLES),
        (gf.PaymentFrequency, fd.PAYMENT_FREQUENCIES),
        (gf.DayCountConvention, fd.DAY_COUNT_CONVENTIONS),
        (gf.TradeStatus, fd.TRADE_STATUSES),
        (gf.Exchange, fd.EXCHANGES),
    ]:
        out = cls().call(_number_of_rows_per_batch=30, _rng=np.random.default_rng(0))
        assert set(out) <= set(vocab)
        assert cls().cardinality() == len(vocab)


def test_clearing_house_is_coherent_with_asset_class():
    table = pl.DataFrame({"asset_class": ["Rates"] * 50 + ["Equity"] * 50})
    out = gf.ClearingHouse().call(
        _number_of_rows_per_batch=100, _table=table, _rng=np.random.default_rng(0)
    )
    ac = table["asset_class"].to_numpy()
    assert all(
        ch in fd.CLEARING_HOUSES_BY_ASSET_CLASS["Rates"] for ch in np.array(out)[ac == "Rates"]
    )
    assert all(
        ch in fd.CLEARING_HOUSES_BY_ASSET_CLASS["Equity"] for ch in np.array(out)[ac == "Equity"]
    )


# ── Greeks / risk numerics / sensitivities priors ────────────────────────────


def test_greeks_and_risk_numerics_registered_flat():
    for sem in [
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "volatility",
        "strike",
        "spot",
        "var",
        "expected_shortfall",
    ]:
        module, _cls, base_args, parent = SEMANTIC_GENERATORS[sem]
        assert module == "datagenic.generators" and parent is None
        assert set(base_args) == {"mu", "sigma", "distribution"}
    # signed Greeks are normal; positive drivers are lognormal
    assert build_generator("delta")["args"]["distribution"] == "normal"
    assert build_generator("volatility")["args"]["distribution"] == "lognormal"


def test_sensitivities_is_a_flat_array_semantic():
    module, cls, base_args, parent = SEMANTIC_GENERATORS["sensitivities"]
    assert cls == "RandomArrayFloats" and parent is None
    assert base_args["size_of_arrays"] == 10


# ── coherent dates ────────────────────────────────────────────────────────────


def test_tenor_to_days():
    assert gf._tenor_to_days("ON") == 1
    assert gf._tenor_to_days("1W") == 7
    assert gf._tenor_to_days("3M") == 90
    assert gf._tenor_to_days("5Y") == 5 * 365
    assert gf._tenor_to_days("garbage") == 365  # default 1Y


def test_maturity_is_trade_date_plus_tenor():
    trade = gf.TradeDate().call(
        _number_of_rows_per_batch=4, _date="2024-06-30", _rng=np.random.default_rng(1)
    )
    table = pl.DataFrame({"trade_date": trade, "tenor": ["5Y", "3M", "ON", "1W"]})
    mat = gf.MaturityDate().call(_number_of_rows_per_batch=4, _date="2024-06-30", _table=table)
    expected_days = np.array([5 * 365, 90, 1, 7], dtype="timedelta64[D]")
    assert np.array_equal(
        np.asarray(mat, dtype="datetime64[D]"),
        np.asarray(trade, dtype="datetime64[D]") + expected_days,
    )


def test_settlement_is_trade_date_plus_lag():
    trade = gf.TradeDate().call(
        _number_of_rows_per_batch=3, _date="2024-06-30", _rng=np.random.default_rng(1)
    )
    table = pl.DataFrame({"trade_date": trade})
    setl = gf.SettlementDate().call(_number_of_rows_per_batch=3, _date="2024-06-30", _table=table)
    assert np.array_equal(
        np.asarray(setl, dtype="datetime64[D]"),
        np.asarray(trade, dtype="datetime64[D]") + np.timedelta64(2, "D"),
    )


def test_trade_date_is_within_lookback():
    out = gf.TradeDate().call(
        _number_of_rows_per_batch=200,
        _date="2024-06-30",
        lookback_days=365,
        _rng=np.random.default_rng(0),
    )
    asof = np.datetime64("2024-06-30", "D")
    out = np.asarray(out, dtype="datetime64[D]")
    assert (out <= asof).all() and (out > asof - np.timedelta64(365, "D")).all()


def test_trades_are_always_live_when_tenor_is_present():
    # trade-date age is bounded by the row's tenor, so every generated trade is live at
    # the as-of date — maturity >= as-of >= trade_date holds row-wise, across a mix of
    # short and long tenors, while maturity = trade_date + tenor stays exact.
    asof = "2026-07-07"
    tenors = ["ON", "1W", "1M", "3M", "6M", "1Y", "5Y", "10Y"] * 25  # 200 rows
    n = len(tenors)
    trade = gf.TradeDate().call(
        _number_of_rows_per_batch=n,
        _date=asof,
        _table=pl.DataFrame({"tenor": tenors}),
        _rng=np.random.default_rng(0),
    )
    table = pl.DataFrame({"trade_date": trade, "tenor": tenors})
    mat = gf.MaturityDate().call(_number_of_rows_per_batch=n, _date=asof, _table=table)

    asof_d = np.datetime64(asof, "D")
    trade_d = np.asarray(trade, dtype="datetime64[D]")
    mat_d = np.asarray(mat, dtype="datetime64[D]")
    # the full ordering the flat lookback couldn't guarantee
    assert (trade_d <= asof_d).all()
    assert (mat_d >= asof_d).all()
    # still no older than the default lookback, and maturity is exactly trade + tenor
    assert (trade_d > asof_d - np.timedelta64(365, "D")).all()
    expected_days = np.array([gf._tenor_to_days(t) for t in tenors], dtype="timedelta64[D]")
    assert np.array_equal(mat_d, trade_d + expected_days)


def test_trade_date_is_deterministic_with_tenor():
    kw = dict(
        _number_of_rows_per_batch=3,
        _date="2026-07-07",
        _table=pl.DataFrame({"tenor": ["3M", "5Y", "1M"]}),
    )
    a = gf.TradeDate().call(**kw, _rng=np.random.default_rng(7))
    b = gf.TradeDate().call(**kw, _rng=np.random.default_rng(7))
    assert np.array_equal(
        np.asarray(a, dtype="datetime64[D]"), np.asarray(b, dtype="datetime64[D]")
    )


# ── counterparty ticker / LEI (coherent + valid checksum) ─────────────────────


def test_ticker_and_lei_are_coherent_with_counterparty():
    table = pl.DataFrame({"counterparty": ["Apple", "Apple", "LVMH"]})
    tk = gf.CounterpartyTicker().call(
        _number_of_rows_per_batch=3, _table=table, _rng=np.random.default_rng(0)
    )
    lei = gf.CounterpartyLEI().call(
        _number_of_rows_per_batch=3, _table=table, _rng=np.random.default_rng(0)
    )
    assert tk[0] == tk[1] and tk[0] != tk[2]  # same name -> same ticker
    assert lei[0] == lei[1] and lei[0] != lei[2]  # coherent per name
    assert all(len(x) == 20 for x in lei)
    assert all(gf._lei_check_digits(x[:18]) == x[18:] for x in lei)  # valid ISO 7064 check


# ── underlying coherent with asset class ──────────────────────────────────────


def test_underlying_is_a_name_for_equity_credit_only():
    table = pl.DataFrame({"asset_class": ["Equity", "Credit", "Rates", "FX"]})
    out = gf.Underlying().call(
        _number_of_rows_per_batch=4, _table=table, _rng=np.random.default_rng(0)
    )
    assert out[0] in fd.COUNTERPARTIES and out[1] in fd.COUNTERPARTIES
    assert out[2] == "N/A" and out[3] == "N/A"


# ── inference wiring ──────────────────────────────────────────────────────────


def test_inference_maps_and_links_new_fields():
    columns = [
        {"name": "asset_class", "type": "string"},
        {"name": "counterparty", "type": "string"},
        {"name": "tenor", "type": "string"},
        {"name": "trade_date", "type": "date"},
        {"name": "side", "type": "string"},
        {"name": "clearing_house", "type": "string"},
        {"name": "underlying", "type": "string"},
        {"name": "cpty_ticker", "type": "string"},
        {"name": "lei", "type": "string"},
        {"name": "maturity_date", "type": "date"},
        {"name": "value_date", "type": "date"},
        {"name": "delta", "type": "float"},
        {"name": "sensitivities", "type": "array(float)"},
    ]
    inf = heuristic_inferrer("trades", columns)
    assert inf["side"]["semantic"] == "side"
    assert inf["clearing_house"] == {
        **inf["clearing_house"],
        "semantic": "clearing_house",
        "parent": "asset_class",
    }
    assert inf["underlying"]["parent"] == "asset_class"
    assert (
        inf["cpty_ticker"]["semantic"] == "counterparty_ticker"
        and inf["cpty_ticker"]["parent"] == "counterparty"
    )
    assert inf["lei"]["semantic"] == "counterparty_lei"
    assert inf["trade_date"]["parent"] == "tenor"  # bounds trade age -> always live
    assert inf["maturity_date"]["parent"] == "trade_date"
    assert (
        inf["value_date"]["semantic"] == "settlement_date"
        and inf["value_date"]["parent"] == "trade_date"
    )
    assert inf["delta"]["semantic"] == "delta"
    assert inf["sensitivities"]["semantic"] == "sensitivities"


# ── end-to-end: coherent trade blotter ────────────────────────────────────────


def _blotter_ir():
    # dependents listed before parents to exercise coherence reordering
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "maturity_date", "type": "date"},
                    {"name": "clearing_house", "type": "string"},
                    {"name": "asset_class", "type": "string"},
                    {"name": "tenor", "type": "string"},
                    {"name": "trade_date", "type": "date"},
                    {"name": "side", "type": "string"},
                ],
            }
        ],
    }


def test_end_to_end_blotter_is_coherent(tmp_path):
    cfg = Configs(
        input=_blotter_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=500,
        number_of_batches=1,
        # a single as-of snapshot: trade dates are frozen at inception, so "every trade is
        # live" is a point-in-time property — over later reporting dates a short-tenor
        # trade (immutable booking date) legitimately matures.
        start_date="2024-06-26",  # a Wednesday (single business-day as-of)
        end_date="2024-06-26",
        semantic_inferrer="heuristic",
    ).config_template
    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("asset_class") < order.index("clearing_house")
    # tenor -> trade_date -> maturity_date: the full chain the coherence sort resolves,
    # so trade-date age is bounded by tenor and every trade is live at the as-of date
    assert order.index("tenor") < order.index("trade_date") < order.index("maturity_date")

    Generator(data_generator_config_file=cfg).generate()
    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])

    assert set(df["side"].to_list()) <= set(fd.SIDES)
    # clearing house coherent with asset class
    assert all(
        ch in fd.CLEARING_HOUSES_BY_ASSET_CLASS[ac]
        for ac, ch in zip(df["asset_class"].to_list(), df["clearing_house"].to_list(), strict=True)
    )
    # maturity strictly after trade date (tenor >= ON = 1 day)
    assert all(
        m > t
        for t, m in zip(df["trade_date"].to_list(), df["maturity_date"].to_list(), strict=True)
        if isinstance(t, datetime.date) and isinstance(m, datetime.date)
    )
    # every trade is live at its as-of date: maturity >= as-of >= trade_date, per partition
    for f in files:
        asof = datetime.date.fromisoformat(f.parent.name)  # partition folder = as-of date
        part = pl.read_parquet(f)
        assert all(t <= asof for t in part["trade_date"].to_list())
        assert all(m >= asof for m in part["maturity_date"].to_list())
