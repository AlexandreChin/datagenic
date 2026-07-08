"""Cardinality-inferred reference-table sizing (opt-in ``Configs(reference_rows="auto")``).

A reference/dimension table's strict minimal grain is one row per distinct unicity-key
tuple: ``rows = ∏ cardinality(key)``. Bounded keys use their real domain; free keys use
the ``number_of_rows_in_non_fact_tables`` budget, split geometrically only when ≥2 keys
are free (the one case cardinality is equalized). Auto references become broadcast
dimensions (single_batch/single_date). Default (``reference_rows=None``) is unchanged.
"""

import warnings

import polars as pl

from datagenic import Configs, Generator


def _cfg(ir, **kw):
    base = dict(output_format="parquet", number_of_batches=1, semantic_inferrer="heuristic")
    base.update(kw)
    return Configs(input=ir, **base).config_template


def _table(cfg, name):
    return next(t for t in cfg["tables"] if t["table_name"] == name)


def _card(table, col):
    return next(c for c in table["columns"] if c["col_name"] == col)["column_generator"][
        "args"
    ].get("cardinality")


# ── opt-in gate ────────────────────────────────────────────────────────────────


def test_default_is_unchanged_flat_sizing():
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "products",
                "keys": ["sku"],
                "columns": [{"name": "sku", "type": "string"}],
            },
        ],
    }
    cfg = _cfg(ir, number_of_rows_in_non_fact_tables=100)  # reference_rows defaults to None
    products = _table(cfg, "products")
    assert products["number_of_rows"] == 100  # legacy flat count
    assert products["single_batch"] is False  # not a broadcast dim by default


# ── grain sizing per key kind ──────────────────────────────────────────────────


def test_bounded_key_sized_to_its_domain():
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "currencies",
                "keys": ["ccy"],
                "columns": [{"name": "ccy", "type": "string", "semantic": "currency"}],
            },
        ],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sole bounded key warns
        cfg = _cfg(ir, reference_rows="auto")
    cur = _table(cfg, "currencies")
    from datagenic.domains.market_finance import _data as fd

    assert cur["number_of_rows"] == len(fd.CURRENCIES)  # its real domain, not a flat 100
    assert cur["single_batch"] is True and cur["single_date"] is True  # broadcast dim


def test_free_single_key_uses_budget():
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "products",
                "keys": ["sku"],
                "columns": [{"name": "sku", "type": "string"}],
            },
        ],
    }
    cfg = _cfg(ir, reference_rows="auto", number_of_rows_in_non_fact_tables=120)
    assert _table(cfg, "products")["number_of_rows"] == 120  # free key -> the budget


def test_two_free_keys_are_equalized_geometrically():
    # THE one case cardinality is equalized: ≥2 free keys split the budget geometrically
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "grid",
                "keys": ["a", "b"],
                "columns": [{"name": "a", "type": "string"}, {"name": "b", "type": "string"}],
            },
        ],
    }
    cfg = _cfg(ir, reference_rows="auto", number_of_rows_in_non_fact_tables=100)
    grid = _table(cfg, "grid")
    assert _card(grid, "a") == _card(grid, "b") == 10  # equal — round(100 ** 1/2)
    assert grid["number_of_rows"] == 100  # 10 × 10
    assert (
        cfg["default_parameters"]["scale_unique_by_construction"] is True
    )  # composite -> enumerate


def test_bounded_and_free_keys_are_not_equalized():
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "mix",
                "keys": ["ccy", "id"],
                "columns": [
                    {"name": "ccy", "type": "string", "semantic": "currency"},
                    {"name": "id", "type": "string"},
                ],
            },
        ],
    }
    cfg = _cfg(ir, reference_rows="auto", number_of_rows_in_non_fact_tables=100)
    from datagenic.domains.market_finance import _data as fd

    mix = _table(cfg, "mix")
    assert _card(mix, "ccy") == len(fd.CURRENCIES)  # bounded keeps its domain
    assert _card(mix, "id") == 100  # the lone free key gets the full budget (not equalized)
    assert mix["number_of_rows"] == len(fd.CURRENCIES) * 100


def test_sole_bounded_key_warns():
    ir = {
        "name": "s",
        "tables": [
            {
                "type": "referential",
                "name": "currencies",
                "keys": ["ccy"],
                "columns": [{"name": "ccy", "type": "string", "semantic": "currency"}],
            },
        ],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _cfg(ir, reference_rows="auto")
    assert any("bounded vocabulary" in str(w.message) for w in caught)


# ── end-to-end: grain, broadcast, FK integrity, fact-decoupling ────────────────


def _shop_ir():
    return {
        "name": "shop",
        "tables": [
            {
                "type": "fact",
                "name": "orders",
                "keys": ["order_id"],
                "columns": [{"name": "order_id", "type": "int"}, {"name": "sku", "type": "string"}],
                "foreign_keys": [{"column": "sku", "references": "products.sku"}],
            },
            {
                "type": "referential",
                "name": "products",
                "keys": ["sku"],
                "columns": [{"name": "sku", "type": "string"}, {"name": "name", "type": "string"}],
            },
        ],
    }


def test_end_to_end_grain_broadcast_and_fk(tmp_path):
    cfg = _cfg(
        _shop_ir(),
        reference_rows="auto",
        number_of_rows_in_fact_tables=1000,
        number_of_rows_in_non_fact_tables=100,
        number_of_batches=2,
        output_folder_path=str(tmp_path / "out"),
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    Generator(data_generator_config_file=cfg).generate()

    prod_files = list((tmp_path / "out" / "products").rglob("*.parquet"))
    assert len(prod_files) == 1  # broadcast dimension: produced once across 2 batches × 2 dates
    products = pl.read_parquet(prod_files[0])
    assert products["sku"].n_unique() == 100  # free single key -> exact grain

    orders = pl.concat(
        [pl.read_parquet(f) for f in (tmp_path / "out" / "orders").rglob("*.parquet")]
    )
    assert set(orders["sku"].to_list()) <= set(products["sku"].to_list())  # FK integrity


def test_changing_fact_rows_leaves_reference_data_unchanged(tmp_path):
    # the decoupling promise: bump fact volume -> reference *data* is identical.
    # (raw bytes differ because the whole-config hash is embedded as parquet metadata,
    # so a fact-count change does not corrupt the dimension, only its checkpoint reuse.)
    def run(sub, fact_rows):
        cfg = _cfg(
            _shop_ir(),
            reference_rows="auto",
            number_of_rows_in_fact_tables=fact_rows,
            number_of_rows_in_non_fact_tables=100,
            number_of_batches=1,
            output_folder_path=str(tmp_path / sub),
            start_date="2024-01-01",
            end_date="2024-01-01",
        )
        Generator(data_generator_config_file=cfg).generate()
        f = next((tmp_path / sub / "products").rglob("*.parquet"))
        return pl.read_parquet(f)

    small = run("a", 1000)
    big = run("b", 50000)
    assert small.equals(big)  # dimension data is byte-for-byte independent of fact volume
