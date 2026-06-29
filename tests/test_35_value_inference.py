"""Phase 2 — value-based semantic inference (``--infer-semantics values``).

The ``values`` inferrer classifies a column from its actual sample values
(currency/tenor/taxonomy-vocabulary membership; ISIN/CUSIP/SEDOL regex +
checksum), and falls back to the name heuristic for columns with no sample
evidence. ``Configs`` accepts ``samples`` (a dict of frames/paths or a directory)
to feed it.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

import datagenic.generators_finance as gf
from datagenic import Configs
from datagenic.semantics import values_inferrer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_isins(n, seed=0):
    return list(
        gf.ISIN().call(
            _number_of_rows_per_batch=n,
            _batch_number=1,
            _is_unicity_key=True,
            _rng=np.random.default_rng(seed),
        )
    )


# ── values_inferrer directly ──────────────────────────────────────────────────


def test_values_classify_by_content_not_name():
    """A column whose NAME gives no hint is still classified from its values."""
    columns = [
        {"name": "field_a", "type": "string"},
        {"name": "field_b", "type": "string"},
        {"name": "field_c", "type": "string"},
    ]
    samples = {
        "field_a": ["EUR", "USD", "GBP", "JPY", "EUR"],
        "field_b": _valid_isins(10),
        "field_c": ["1Y", "5Y", "10Y", "3M"],
    }
    inferred = values_inferrer("t", columns, samples=samples)
    assert inferred["field_a"]["semantic"] == "currency"
    assert inferred["field_a"]["by"] == "values"
    assert inferred["field_b"]["semantic"] == "isin"
    assert inferred["field_c"]["semantic"] == "tenor"
    # value evidence is high-confidence
    assert inferred["field_a"]["confidence"] >= 0.9


def test_isin_classification_requires_valid_checksum():
    valid = _valid_isins(12)
    # corrupt the check digit so the regex still matches but the checksum fails
    corrupt = [s[:-1] + str((int(s[-1]) + 1) % 10) for s in valid]

    good = values_inferrer("t", [{"name": "x", "type": "string"}], samples={"x": valid})
    bad = values_inferrer("t", [{"name": "x", "type": "string"}], samples={"x": corrupt})
    assert good["x"]["semantic"] == "isin"
    assert "x" not in bad  # bad checksums -> not recognised as ISIN (no name hint either)


def test_values_links_hierarchy_parent_and_is_coherent_target():
    columns = [
        {"name": "ac", "type": "string"},
        {"name": "it", "type": "string"},
    ]
    samples = {
        "ac": ["Rates", "Credit", "Equity", "FX"],
        "it": ["IR Swap", "CDS", "Equity Option", "FX Forward"],
    }
    inferred = values_inferrer("t", columns, samples=samples)
    assert inferred["ac"]["semantic"] == "asset_class"
    assert inferred["it"]["semantic"] == "instrument_type"
    assert inferred["it"]["parent"] == "ac"  # linked to the asset_class column


def test_values_falls_back_to_name_heuristic_without_samples():
    columns = [
        {"name": "ccy", "type": "string"},  # name hint, no sample
        {"name": "isin", "type": "string"},
    ]
    inferred = values_inferrer("t", columns, samples={})
    assert inferred["ccy"]["semantic"] == "currency"
    assert inferred["ccy"]["by"] == "heuristic"
    assert inferred["isin"]["semantic"] == "isin"


# ── Configs integration ───────────────────────────────────────────────────────


def _finance_ir():
    return {
        "name": "market_risk",
        "tables": [
            {
                "type": "fact",
                "name": "trades",
                "keys": ["trade_id"],
                "columns": [
                    {"name": "trade_id", "type": "int"},
                    # deliberately unhelpful names so only values can classify them
                    {"name": "dim1", "type": "string"},
                    {"name": "dim2", "type": "string"},
                    {"name": "dim3", "type": "string"},
                ],
            }
        ],
    }


def _trades_sample(n=40):
    rng = np.random.default_rng(1)
    asset_class = gf.FinancialAssetClass().call(_number_of_rows_per_batch=n, _rng=rng)
    table = pl.DataFrame({"asset_class": asset_class})
    instr = gf.FinancialInstrumentType().call(_number_of_rows_per_batch=n, _rng=rng, _table=table)
    return pl.DataFrame(
        {
            "trade_id": np.arange(1, n + 1),
            "dim1": asset_class,
            "dim2": instr,
            "dim3": gf.Currency().call(_number_of_rows_per_batch=n, _rng=rng),
        }
    )


def test_configs_with_frame_samples_classifies_by_value():
    cfg = Configs(
        input=_finance_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="values",
        samples={"trades": _trades_sample()},
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}

    assert cols["dim1"]["column_generator"]["udf_name"] == "FinancialAssetClass"
    assert cols["dim2"]["column_generator"]["udf_name"] == "FinancialInstrumentType"
    assert cols["dim3"]["column_generator"]["udf_name"] == "Currency"
    assert cols["dim1"]["_inferred"]["by"] == "values"
    # hierarchy child wired + ordered after its parent
    assert cols["dim2"]["column_generator"]["args"]["asset_class_column"] == "dim1"
    order = [c["col_name"] for c in cfg["tables"][0]["columns"]]
    assert order.index("dim1") < order.index("dim2")


def test_configs_with_directory_samples(tmp_path):
    _trades_sample().write_parquet(tmp_path / "trades.parquet")
    cfg = Configs(
        input=_finance_ir(),
        output_format="parquet",
        number_of_batches=1,
        semantic_inferrer="values",
        samples=str(tmp_path),
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}
    assert cols["dim1"]["column_generator"]["udf_name"] == "FinancialAssetClass"
    assert cols["dim3"]["column_generator"]["udf_name"] == "Currency"


def test_cli_infer_values_with_samples_dir(tmp_path):
    _trades_sample().write_parquet(tmp_path / "trades.parquet")
    out = tmp_path / "config.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datagenic",
            "generate-data-config",
            json.dumps(_finance_ir()),
            "-o",
            str(out),
            "--output-folder",
            str(tmp_path / "data"),
            "--infer-semantics",
            "values",
            "--samples",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    cfg = json.loads(out.read_text())
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}
    assert cols["dim1"]["column_generator"]["udf_name"] == "FinancialAssetClass"
