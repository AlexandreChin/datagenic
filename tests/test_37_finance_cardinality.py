"""Finance categorical generators honor the engine's auto-assigned cardinality.

A fixed-domain categorical used as a unicity key used to ignore the cardinality
the engine budgeted for it, so the keys' product undershot the row count and the
final dedup collapsed the table. The generators now declare their true distinct
count (``cardinality()``) and honor an assigned cardinality as a cap; the engine
consults that declaration to budget the other keys, and warns when the keys are
mathematically too small to reach the requested row count.
"""

import numpy as np
import polars as pl

import datagenic._finance_data as fd
import datagenic.generators_finance as gf
from datagenic import Configs, Generator

# ── declare + honor cardinality (unit) ────────────────────────────────────────


def test_cardinality_declares_vocab_size():
    assert gf.FinancialAssetClass().cardinality() == len(fd.ASSET_CLASSES)
    assert gf.Currency().cardinality() == len(fd.CURRENCIES)
    assert gf.Tenor().cardinality() == len(fd.TENORS)
    assert gf.CreditRating().cardinality(scale="Moody") == len(fd.MOODYS_RATINGS)


def test_cardinality_capped_by_assignment():
    g = gf.FinancialAssetClass()
    assert g.cardinality(cardinality=3) == 3
    assert g.cardinality(cardinality=100) == len(fd.ASSET_CLASSES)  # can't exceed the domain


def test_cardinality_respects_list_of_values():
    assert gf.Currency().cardinality(list_of_values=["USD", "EUR"]) == 2


def test_call_honors_cardinality_cap():
    out = gf.FinancialAssetClass().call(
        _number_of_rows_per_batch=300, cardinality=3, _rng=np.random.default_rng(0)
    )
    assert set(out) <= set(fd.ASSET_CLASSES[:3])  # head slice, deterministic
    assert len(set(out)) <= 3


def test_call_full_vocab_without_cardinality():
    out = gf.Currency().call(_number_of_rows_per_batch=500, _rng=np.random.default_rng(0))
    assert set(out) <= set(fd.CURRENCIES)


# ── engine: row count no longer collapses ─────────────────────────────────────


def _ir(keys):
    return {
        "name": "mr",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": keys,
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "asset_class", "type": "string"},
                    {"name": "notional", "type": "float"},
                ],
            }
        ],
    }


def test_composite_finance_key_preserves_row_count(tmp_path):
    """With a finance categorical in a composite key, the engine budgets the other
    key for the remaining rows instead of over-allocating to the small domain.

    Verified under scale_unique_by_construction (deterministic mixed-radix
    enumeration): the fix yields ~600 distinct tuples; the old behavior — treating
    asset_class as a free key assigned ~sqrt(600)=24 while it can only deliver 6 —
    capped the product at 6*24 = 144.
    """
    cfg = Configs(
        input=_ir(["trade_id", "asset_class"]),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=600,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    cfg["default_parameters"]["scale_unique_by_construction"] = True

    Generator(data_generator_config_file=cfg).generate()
    df = pl.read_parquet(next((tmp_path / "out" / "trades").rglob("*.parquet")))

    assert len(df) > 400  # fix ~590; pre-fix collapse was 144
    # coherence/domain still intact
    assert set(df["asset_class"].to_list()) <= set(fd.ASSET_CLASSES)


def test_unsatisfiable_keys_warn_and_cap(tmp_path, capsys):
    """A lone small-domain categorical key cannot reach the requested rows — the
    engine warns with the achievable ceiling rather than collapsing silently."""
    ir = {
        "name": "dim",
        "tables": [
            {
                "type": "referential",
                "name": "asset_classes",
                "keys": ["asset_class"],
                "columns": [{"name": "asset_class", "type": "string"}],
            }
        ],
    }
    cfg = Configs(
        input=ir,
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_non_fact_tables=100,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template

    Generator(data_generator_config_file=cfg).generate()
    out = capsys.readouterr().out
    assert "admit at most" in out
    df = pl.read_parquet(next((tmp_path / "out" / "asset_classes").rglob("*.parquet")))
    assert len(df) == len(fd.ASSET_CLASSES)  # capped at the domain


def test_finance_key_generation_is_deterministic(tmp_path):
    cfg = Configs(
        input=_ir(["trade_id", "asset_class"]),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=300,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template

    g = Generator(data_generator_config_file=cfg)
    g.generate(force_regenerate=True)
    first = pl.read_parquet(next((tmp_path / "out" / "trades").rglob("*.parquet"))).to_dicts()
    g.generate(force_regenerate=True)
    second = pl.read_parquet(next((tmp_path / "out" / "trades").rglob("*.parquet"))).to_dicts()
    assert first == second
