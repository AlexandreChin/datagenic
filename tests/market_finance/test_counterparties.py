"""Counterparty pool + metadata (sector / country) from bundled real-index CSVs.

``market_finance._data`` loads real large-cap names and metadata from per-index CSVs under
``datagenic/data/counterparties/`` (CAC 40, DAX 40, EURO STOXX 50, SMI, FTSE 100,
S&P 500). ``CounterpartySector`` / ``CounterpartyCountry`` read the row's counterparty
from the partial table and map it via ``COUNTERPARTY_METADATA`` — so a counterparty's
sector/country are always coherent (Apple → Information Technology / US), the numeric
twin of the asset_class → instrument_type hierarchy. Inference wires them automatically.
"""

import importlib
import importlib.resources

import numpy as np
import polars as pl

import datagenic.market_finance._data as fd
import datagenic.market_finance.generators as gf
from datagenic import Configs, Generator
from datagenic.semantics import build_generator, heuristic_inferrer

_GICS = set(fd.GICS_SECTORS)


# ── pool + metadata ───────────────────────────────────────────────────────────


def test_metadata_is_large_real_and_well_formed():
    meta = fd.COUNTERPARTY_METADATA
    assert len(meta) > 400
    assert list(meta) == fd.COUNTERPARTIES
    assert len(set(fd.COUNTERPARTIES)) == len(fd.COUNTERPARTIES)
    for name, info in meta.items():
        assert info["sector"] in _GICS, (name, info["sector"])
        assert isinstance(info["country"], str) and len(info["country"]) == 2
        assert info["index"]


def test_known_members_map_correctly():
    meta = fd.COUNTERPARTY_METADATA
    assert meta["Apple"] == {"sector": "Information Technology", "country": "US", "index": "sp500"}
    assert meta["LVMH"]["sector"] == "Consumer Discretionary" and meta["LVMH"]["country"] == "FR"
    assert meta["Shell"]["index"] == "ftse100"
    assert meta["Nestlé"]["country"] == "CH"  # Swiss names live in the SMI file, not EURO STOXX


# ── coherent generators ─────────────────────────────────────────────────────


def test_sector_and_country_are_coherent_with_counterparty():
    table = pl.DataFrame({"counterparty": ["Apple", "JPMorgan Chase", "Shell", "Apple"]})
    sec = gf.CounterpartySector().call(
        _number_of_rows_per_batch=4, _table=table, _rng=np.random.default_rng(0)
    )
    ctry = gf.CounterpartyCountry().call(
        _number_of_rows_per_batch=4, _table=table, _rng=np.random.default_rng(0)
    )
    assert list(sec) == ["Information Technology", "Financials", "Energy", "Information Technology"]
    assert list(ctry) == ["US", "US", "GB", "US"]


def test_absent_counterparty_column_falls_back_to_flat_draw():
    out = gf.CounterpartySector().call(
        _number_of_rows_per_batch=5, _table=pl.DataFrame({}), _rng=np.random.default_rng(0)
    )
    assert len(out) == 5 and set(out) <= _GICS


def test_unknown_counterparty_draws_from_vocab():
    table = pl.DataFrame({"counterparty": ["Not A Real Name"] * 3})
    out = gf.CounterpartySector().call(
        _number_of_rows_per_batch=3, _table=table, _rng=np.random.default_rng(0)
    )
    assert set(out) <= _GICS  # unmapped -> a plausible sector, never an error


def test_coherent_generators_are_deterministic():
    table = pl.DataFrame({"counterparty": ["Apple", "Unknown Co", "Shell"]})
    kw = dict(_number_of_rows_per_batch=3, _table=table)
    a = gf.CounterpartySector().call(**kw, _rng=np.random.default_rng(1))
    b = gf.CounterpartySector().call(**kw, _rng=np.random.default_rng(1))
    assert list(a) == list(b)


# ── fallback when package data is missing ─────────────────────────────────────


def test_falls_back_to_fictional_metadata_when_files_missing(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("no package data")

    monkeypatch.setattr(importlib.resources, "files", _boom)
    meta = fd._load_counterparty_metadata()
    assert meta == fd._FALLBACK_COUNTERPARTY_METADATA
    assert all(info["sector"] == "Financials" for info in meta.values())


# ── inference wiring ──────────────────────────────────────────────────────────


def test_inference_maps_and_links_counterparty_attributes():
    columns = [
        {"name": "counterparty", "type": "string"},
        {"name": "cpty_sector", "type": "string"},
        {"name": "counterparty_country", "type": "string"},
        {"name": "sector", "type": "string"},  # bare 'sector' is still the GICS semantic
    ]
    inferred = heuristic_inferrer("trades", columns)
    assert inferred["cpty_sector"]["semantic"] == "counterparty_sector"
    assert inferred["cpty_sector"]["parent"] == "counterparty"
    assert inferred["counterparty_country"]["semantic"] == "counterparty_country"
    assert inferred["counterparty_country"]["parent"] == "counterparty"
    assert inferred["counterparty"]["semantic"] == "counterparty"  # not swallowed
    assert inferred["sector"]["semantic"] == "gics_sector"

    cg = build_generator("counterparty_sector", "counterparty")
    assert cg["udf_name"] == "CounterpartySector"
    assert cg["args"]["counterparty_column"] == "counterparty"


def _cpty_ir():
    # sector listed before counterparty to exercise coherence reordering
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "cpty_sector", "type": "string"},
                    {"name": "counterparty", "type": "string"},
                ],
            }
        ],
    }


def test_scaffold_orders_counterparty_before_its_sector():
    template = Configs(
        input=_cpty_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    table = template["tables"][0]
    order = [c["col_name"] for c in table["columns"]]
    assert order.index("counterparty") < order.index("cpty_sector")
    sec = {c["col_name"]: c for c in table["columns"]}["cpty_sector"]
    assert sec["column_generator"]["udf_name"] == "CounterpartySector"
    assert sec["_inferred"]["parent"] == "counterparty"


def test_end_to_end_sector_matches_counterparty(tmp_path):
    cfg = Configs(
        input=_cpty_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=500,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])
    meta = fd.COUNTERPARTY_METADATA
    # every row's sector is the one mapped from its counterparty
    assert all(
        meta[cp]["sector"] == sec
        for cp, sec in zip(df["counterparty"].to_list(), df["cpty_sector"].to_list(), strict=True)
    )
