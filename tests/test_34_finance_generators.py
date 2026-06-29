"""Financial-markets domain generators + automatic semantic inference at scaffold time.

Covers:
  - the finance generators (coherent hierarchy, identifiers with valid check
    digits, determinism, per-batch key uniqueness),
  - the heuristic semantic inferrer,
  - Configs wiring inference into a scaffolded config (generator override,
    ``_inferred`` annotation, parent-before-dependent ordering),
  - end-to-end generation producing coherent rows,
  - the import-light boundary (finance/numpy not pulled by a bare import),
  - the ``review-config`` CLI.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import datagenic._finance_data as fd
import datagenic.generators_finance as gf
from datagenic import Configs, Generator
from datagenic.semantics import heuristic_inferrer

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Generators: determinism + coherence ──────────────────────────────────────


def test_asset_class_is_deterministic():
    a = gf.FinancialAssetClass().call(_number_of_rows_per_batch=50, _rng=np.random.default_rng(7))
    b = gf.FinancialAssetClass().call(_number_of_rows_per_batch=50, _rng=np.random.default_rng(7))
    assert list(a) == list(b)
    assert set(a) <= set(fd.ASSET_CLASSES)


def test_instrument_type_and_risk_factor_group_are_coherent():
    rng = np.random.default_rng(0)
    asset_class = gf.FinancialAssetClass().call(_number_of_rows_per_batch=200, _rng=rng)
    table = pl.DataFrame({"asset_class": asset_class})

    instr = gf.FinancialInstrumentType().call(_number_of_rows_per_batch=200, _rng=rng, _table=table)
    rfg = gf.FinancialRiskFactorGroup().call(_number_of_rows_per_batch=200, _rng=rng, _table=table)

    assert all(
        i in fd.INSTRUMENT_TYPES_BY_ASSET_CLASS[a] for a, i in zip(asset_class, instr, strict=True)
    )
    assert all(
        r in fd.RISK_FACTOR_GROUPS_BY_ASSET_CLASS[a] for a, r in zip(asset_class, rfg, strict=True)
    )


def test_hierarchy_generator_falls_back_without_parent():
    """No asset_class column in _table -> flat draw over all instrument types."""
    out = gf.FinancialInstrumentType().call(
        _number_of_rows_per_batch=30, _rng=np.random.default_rng(1), _table=pl.DataFrame({})
    )
    assert set(out) <= set(fd.ALL_INSTRUMENT_TYPES)


# ── Identifiers: valid check digits + per-batch uniqueness ────────────────────


@pytest.mark.parametrize(
    "cls,length,check_fn",
    [
        (gf.ISIN, 12, gf._isin_check_digit),
        (gf.CUSIP, 9, gf._cusip_check_digit),
        (gf.SEDOL, 7, gf._sedol_check_digit),
    ],
)
def test_identifier_check_digits_validate(cls, length, check_fn):
    codes = cls().call(
        _number_of_rows_per_batch=25,
        _batch_number=1,
        _is_unicity_key=True,
        _rng=np.random.default_rng(3),
    )
    assert all(len(c) == length for c in codes)
    assert all(c[:-1] + check_fn(c[:-1]) == c for c in codes)


def test_identifier_keys_disjoint_across_batches():
    b1 = gf.ISIN().call(
        _number_of_rows_per_batch=100,
        _batch_number=1,
        _is_unicity_key=True,
        _rng=np.random.default_rng(5),
    )
    b2 = gf.ISIN().call(
        _number_of_rows_per_batch=100,
        _batch_number=2,
        _is_unicity_key=True,
        _rng=np.random.default_rng(5),
    )
    assert set(b1).isdisjoint(set(b2))
    assert len(set(b1)) == 100  # unique within a batch


def test_composed_instrument_name():
    table = pl.DataFrame(
        {"ccy": ["EUR", "USD"], "instrument_type": ["IR Swap", "FX Forward"], "tenor": ["5Y", "1M"]}
    )
    names = gf.InstrumentName().call(_number_of_rows_per_batch=2, _table=table)
    assert list(names) == ["EUR IR Swap 5Y", "USD FX Forward 1M"]


# ── Heuristic inference ───────────────────────────────────────────────────────


def test_heuristic_inferrer_maps_and_links_parent():
    columns = [
        {"name": "trade_id", "type": "int"},
        {"name": "asset_class", "type": "string"},
        {"name": "instrument_type", "type": "string"},
        {"name": "risk_factor_group", "type": "string"},
        {"name": "ccy", "type": "string"},
        {"name": "tenor", "type": "string"},
        {"name": "isin", "type": "string"},
        {"name": "amount", "type": "float"},
    ]
    inferred = heuristic_inferrer("trades", columns)

    assert inferred["asset_class"]["semantic"] == "asset_class"
    assert inferred["instrument_type"]["semantic"] == "instrument_type"
    assert inferred["ccy"]["semantic"] == "currency"
    assert inferred["isin"]["semantic"] == "isin"
    # numeric columns are not inferred
    assert "trade_id" not in inferred and "amount" not in inferred
    # hierarchy children are linked to the asset_class column
    assert inferred["instrument_type"]["parent"] == "asset_class"
    assert inferred["risk_factor_group"]["parent"] == "asset_class"


# ── Scaffolding integration ───────────────────────────────────────────────────


def _finance_ir():
    # instrument_type intentionally listed BEFORE asset_class to exercise reordering
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    {"name": "instrument_type", "type": "string"},
                    {"name": "asset_class", "type": "string"},
                    {"name": "risk_factor_group", "type": "string"},
                    {"name": "ccy", "type": "string"},
                    {"name": "tenor", "type": "string"},
                    {"name": "isin", "type": "string"},
                ],
            }
        ],
    }


def test_scaffold_without_inference_is_unchanged():
    template = Configs(
        input=_finance_ir(), output_format="parquet", number_of_batches=1
    ).config_template
    cols = {c["col_name"]: c for c in template["tables"][0]["columns"]}
    assert cols["asset_class"]["column_generator"]["udf_module"] == "datagenic.generators"
    assert "_inferred" not in cols["asset_class"]


def test_scaffold_with_inference_wires_finance_generators():
    template = Configs(
        input=_finance_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    table = template["tables"][0]
    cols = {c["col_name"]: c for c in table["columns"]}

    ac_gen = cols["asset_class"]["column_generator"]
    assert ac_gen["udf_module"] == "datagenic.generators_finance"
    assert ac_gen["udf_name"] == "FinancialAssetClass"
    # default call() args are introspected, same as for the built-in generators
    assert set(ac_gen["args"]) == {"list_of_values", "probabilities"}
    assert cols["instrument_type"]["column_generator"]["udf_name"] == "FinancialInstrumentType"
    assert (
        cols["instrument_type"]["column_generator"]["args"]["asset_class_column"] == "asset_class"
    )
    assert cols["isin"]["column_generator"]["udf_name"] == "ISIN"
    # annotation present for manual review
    assert cols["instrument_type"]["_inferred"]["semantic"] == "instrument_type"

    # parent (asset_class) ordered before its dependents
    order = [c["col_name"] for c in table["columns"]]
    assert order.index("asset_class") < order.index("instrument_type")
    assert order.index("asset_class") < order.index("risk_factor_group")


def test_explicit_ir_semantic_overrides_inference():
    ir = {
        "name": "m",
        "tables": [
            {
                "type": "referential",
                "name": "dim",
                "keys": ["id"],
                "columns": [
                    {"name": "id", "type": "int"},
                    # name would heuristically be nothing; force a semantic explicitly
                    {"name": "label", "type": "string", "semantic": "currency"},
                ],
            }
        ],
    }
    template = Configs(input=ir, output_format="parquet", number_of_batches=1).config_template
    cols = {c["col_name"]: c for c in template["tables"][0]["columns"]}
    assert cols["label"]["column_generator"]["udf_name"] == "Currency"
    assert cols["label"]["_inferred"]["by"] == "explicit"


def test_end_to_end_generation_is_coherent(tmp_path):
    cfg = Configs(
        input=_finance_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=300,
        number_of_batches=1,
        semantic_inferrer="heuristic",
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])

    pairs = set(zip(df["asset_class"].to_list(), df["instrument_type"].to_list(), strict=True))
    assert all(i in fd.INSTRUMENT_TYPES_BY_ASSET_CLASS[a] for a, i in pairs)
    rf_pairs = set(zip(df["asset_class"].to_list(), df["risk_factor_group"].to_list(), strict=True))
    assert all(r in fd.RISK_FACTOR_GROUPS_BY_ASSET_CLASS[a] for a, r in rf_pairs)
    # ISINs are valid
    assert all(c[:-1] + gf._isin_check_digit(c[:-1]) == c for c in df["isin"].to_list())


# ── Import boundary + CLI ─────────────────────────────────────────────────────


def test_bare_import_does_not_pull_finance_or_numpy():
    code = (
        "import datagenic, sys; "
        "print(any(m in sys.modules for m in "
        "('datagenic.generators_finance', 'datagenic._finance_data', 'numpy')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_cli_infer_semantics_and_review_config(tmp_path):
    ir = json.dumps(_finance_ir())
    out = tmp_path / "config.json"
    gen = subprocess.run(
        [
            sys.executable,
            "-m",
            "datagenic",
            "generate-data-config",
            ir,
            "-o",
            str(out),
            "--output-folder",
            str(tmp_path / "data"),
            "--infer-semantics",
            "heuristic",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert gen.returncode == 0, gen.stderr
    cfg = json.loads(out.read_text())
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}
    assert cols["asset_class"]["column_generator"]["udf_name"] == "FinancialAssetClass"

    review = subprocess.run(
        [sys.executable, "-m", "datagenic", "review-config", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert review.returncode == 0, review.stderr
    assert "asset_class -> asset_class" in review.stdout
    assert "instrument_type -> instrument_type" in review.stdout
