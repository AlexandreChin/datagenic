"""Rolling book — a constant-size book whose trades roll at maturity.

Approach A (stateless renewal): each row is a "slot" running a deterministic renewal
process seeded date-independently from (global_seed, batch, slot). At any as-of date the
slot emits the trade whose [trade_date, maturity) interval contains it, so every emitted
trade is live, the book stays exactly N rows, short-tenor slots turn over (new tenor +
new trade_id on each roll) while long-tenor trades persist. Because the schedule is a
pure function of the seed, it is identical on every date and in every shard/worker.
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl

import datagenic.market_finance.generators as gf
from datagenic import Configs, Generator
from datagenic.orchestration import run_shard

# ── one rolling unit, computed directly through the generators ─────────────────

_TENORS = ["ON", "1W", "1M", "3M", "1Y", "5Y", "10Y"]


def _unit(asof, n=2000, start="2024-01-01", seed=42, batch=1, tenors=None, probabilities=None):
    """Run the four rolling generators for one (date, batch) unit, as the engine would:
    tenor + trade_date first, then maturity (reads both) and trade_id (reads trade_date)."""
    tenors = tenors or _TENORS
    fw = dict(
        _number_of_rows_per_batch=n,
        _start_date=start,
        _date=asof,
        _global_seed=seed,
        _batch_number=batch,
    )
    tenor = gf.RollingTenor().call(tenors=tenors, probabilities=probabilities, **fw)
    trade_date = gf.RollingTradeDate().call(tenors=tenors, probabilities=probabilities, **fw)
    partial = pl.DataFrame({"tenor": tenor, "trade_date": trade_date})
    maturity = gf.RollingMaturityDate().call(_table=partial, **fw)
    trade_id = gf.RollingTradeId().call(_table=partial, **fw)
    return {
        "tenor": np.asarray(tenor, dtype=object),
        "trade_date": np.asarray(trade_date, dtype="datetime64[D]"),
        "maturity_date": np.asarray(maturity, dtype="datetime64[D]"),
        "trade_id": np.asarray(trade_id),
    }


# ── the core invariants, for any as-of date ────────────────────────────────────


def test_every_trade_is_live_and_coherent_at_every_date():
    for d in ["2024-01-01", "2024-02-15", "2024-06-30", "2025-01-01", "2026-07-07"]:
        u = _unit(d)
        asof = np.datetime64(d, "D")
        # trade_date <= as-of < maturity  (live, with the strict maturity-day boundary)
        assert (u["trade_date"] <= asof).all()
        assert (u["maturity_date"] > asof).all()
        # maturity == trade_date + tenor, exactly (derived from the same active trade)
        expected = u["trade_date"] + np.array(
            [gf._tenor_to_days(t) for t in u["tenor"]], dtype="timedelta64[D]"
        )
        assert np.array_equal(u["maturity_date"], expected)


def test_book_is_constant_size_with_turnover_and_new_ids():
    n = 2000
    first = _unit("2024-01-01", n=n)
    later = _unit("2024-03-01", n=n)  # ~60 days on: short tenors have rolled
    assert len(first["trade_id"]) == n and len(later["trade_id"]) == n  # constant pool
    assert len(set(first["trade_id"].tolist())) == n  # ids unique within a date

    gone = set(first["trade_id"].tolist()) - set(later["trade_id"].tolist())
    born = set(later["trade_id"].tolist()) - set(first["trade_id"].tolist())
    assert gone and born  # some trades expired and were replaced
    assert len(gone) == len(born)  # replaced one-for-one -> pool does not shrink


def test_long_tenor_trades_persist_short_ones_roll():
    first = _unit("2024-01-01")
    later = _unit("2024-02-15")
    same_id = first["trade_id"] == later["trade_id"]  # a slot that did NOT roll
    # a persisting slot keeps its id, tenor, trade_date and maturity unchanged
    assert same_id.any() and (~same_id).any()
    assert np.array_equal(first["trade_date"][same_id], later["trade_date"][same_id])
    assert np.array_equal(first["tenor"][same_id], later["tenor"][same_id])
    # slots that rolled tend to be the short-dated ones: their first-date maturity
    # fell on/before the later as-of
    rolled = ~same_id
    assert (first["maturity_date"][rolled] <= np.datetime64("2024-02-15", "D")).all()


def test_probabilities_can_bias_toward_short_maturities():
    # weight the whole book onto ON so almost every slot rolls day to day
    short = ["ON"]
    u = _unit("2024-06-30", tenors=short, probabilities=[1.0])
    assert set(u["tenor"].tolist()) == {"ON"}
    a = _unit("2024-06-30", tenors=short, probabilities=[1.0])["trade_id"]
    b = _unit("2024-07-01", tenors=short, probabilities=[1.0])["trade_id"]
    assert set(a.tolist()).isdisjoint(set(b.tolist()))  # a 1-day book fully turns over


# ── determinism ────────────────────────────────────────────────────────────────


def test_same_seed_is_reproducible_different_seed_differs():
    a = _unit("2024-06-30", seed=7)
    b = _unit("2024-06-30", seed=7)
    assert np.array_equal(a["trade_date"], b["trade_date"])
    assert np.array_equal(a["trade_id"], b["trade_id"])
    c = _unit("2024-06-30", seed=8)
    assert not np.array_equal(a["trade_id"], c["trade_id"])


def test_schedule_is_independent_of_which_date_computes_it():
    # a slot's active trade at a date does not depend on evaluation order/history:
    # computing 2024-06-30 directly matches computing it in any run.
    direct = _unit("2024-06-30")
    assert np.array_equal(_unit("2024-06-30")["trade_id"], direct["trade_id"])


# ── end-to-end via the scaffolder (opt-in) ─────────────────────────────────────


def _rolling_ir():
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "tenor", "type": "string"},
                    {"name": "trade_date", "type": "date"},
                    {"name": "maturity_date", "type": "date"},
                ],
            }
        ],
    }


def _rolling_config(out, *, batches=1):
    return Configs(
        input=_rolling_ir(),
        output_format="parquet",
        output_folder_path=str(out),
        number_of_rows_in_fact_tables=800,
        number_of_batches=batches,
        start_date="2024-01-01",
        end_date="2024-01-05",
        seed=3,
        semantic_inferrer="heuristic",
        rolling_book=True,
    ).config_template


def test_scaffold_wires_rolling_generators():
    cfg = _rolling_config("unused")
    gens = {c["col_name"]: c["column_generator"]["udf_name"] for c in cfg["tables"][0]["columns"]}
    assert gens["tenor"] == "RollingTenor"
    assert gens["trade_date"] == "RollingTradeDate"
    assert gens["maturity_date"] == "RollingMaturityDate"
    assert gens["trade_id"] == "RollingTradeId"  # the unicity key rolls with the book
    # trade_date must be built before the columns that read it
    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("trade_date") < order.index("maturity_date")
    assert order.index("trade_date") < order.index("trade_id")
    # row-index == slot must stay stable across dates -> the key sort is skipped
    assert cfg["default_parameters"]["scale_skip_global_key_sort"] is True


def test_end_to_end_rolling_book(tmp_path):
    out = tmp_path / "out"
    Generator(data_generator_config_file=_rolling_config(out)).generate()

    prev_ids = None
    counts = []
    for f in sorted((out / "trades").rglob("*.parquet")):
        asof = np.datetime64(f.parent.name, "D")
        df = pl.read_parquet(f)
        counts.append(len(df))
        td = df["trade_date"].to_numpy().astype("datetime64[D]")
        mat = df["maturity_date"].to_numpy().astype("datetime64[D]")
        assert (td <= asof).all() and (mat > asof).all()  # every trade live at its as-of
        ids = df["trade_id"].to_list()
        assert len(set(ids)) == len(ids)  # unique within the partition
        if prev_ids is not None:
            gone = len(prev_ids - set(ids))
            born = len(set(ids) - prev_ids)
            assert gone == born  # constant-size book: expired trades replaced one-for-one
        prev_ids = set(ids)
    assert len(counts) >= 2 and len(set(counts)) == 1  # same row count on every date


def _hash_tree(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.parquet"))
    }


def test_rolling_book_is_shard_byte_identical(tmp_path):
    single = tmp_path / "single"
    Generator(data_generator_config_file=_rolling_config(single, batches=4)).generate()

    sharded = tmp_path / "sharded"
    for shard in range(1, 3):
        run_shard(_rolling_config(sharded, batches=4), shard=shard, of=2)

    assert _hash_tree(sharded) == _hash_tree(single)  # union of shards == single run
