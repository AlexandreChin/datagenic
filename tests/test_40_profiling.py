"""Feature 2 — profile a sample of files into a generation config (no IR authored).

``build_config`` reads sample frames, synthesises an IR (tables, keys, FK
candidates) and fits generator args (cardinality, mu/sigma, value frequencies,
null rate) from the data, then scaffolds via ``Configs``. The IR is internal; the
user supplies only files.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from datagenic import Generator
from datagenic.profiling import build_config, profile_frames

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frames():
    rng = np.random.default_rng(0)
    customers = pl.DataFrame(
        {
            "id": [f"c{i}" for i in range(60)],  # unique -> key
            "region": ["EU"] * 30 + ["US"] * 20 + ["APAC"] * 10,  # low-card categorical
        }
    )
    orders = pl.DataFrame(
        {
            "order_id": list(range(300)),  # unique -> key
            "customer_id": [f"c{i % 60}" for i in range(300)],  # ⊆ customers.id -> FK
            "amount": rng.normal(100.0, 15.0, 300),  # float -> mu/sigma
            "note": [None if i % 5 == 0 else "x" for i in range(300)],  # 20% null
        }
    )
    return {"customers": customers, "orders": orders}


# ── profile_frames ────────────────────────────────────────────────────────────


def test_profile_infers_keys_fks_and_types():
    ir, overrides = profile_frames(_frames())
    by = {t["name"]: t for t in ir["tables"]}

    assert by["customers"]["keys"] == ["id"]
    assert by["customers"]["type"] == "referential"
    assert by["orders"]["keys"] == ["order_id"]
    assert by["orders"]["type"] == "fact"
    assert {"column": "customer_id", "references": "customers.id"} in by["orders"]["foreign_keys"]

    # per-table row counts captured
    assert overrides["customers"]["rows"] == 60
    assert overrides["orders"]["rows"] == 300


def test_profile_fits_generator_args():
    _ir, overrides = profile_frames(_frames())

    region = overrides["customers"]["cols"]["region"]["args"]
    assert set(region["list_of_values"]) == {"EU", "US", "APAC"}
    assert abs(sum(region["probabilities"]) - 1.0) < 1e-9  # numpy-safe

    amount = overrides["orders"]["cols"]["amount"]["args"]
    assert 90 < amount["mu"] < 110
    assert amount["sigma"] > 0

    assert 0.15 < overrides["orders"]["cols"]["note"]["null_rate"] < 0.25

    # the FK column's args were dropped (KeyEnforcer supplies its values)
    assert "customer_id" not in overrides["orders"]["cols"]


def test_nullable_fk_becomes_fk_null_rate():
    customers = pl.DataFrame({"id": [f"c{i}" for i in range(10)]})
    orders = pl.DataFrame(
        {
            "order_id": list(range(40)),
            "customer_id": [None if i % 4 == 0 else f"c{i % 10}" for i in range(40)],
        }
    )
    ir, _ = profile_frames({"customers": customers, "orders": orders})
    fk = next(t for t in ir["tables"] if t["name"] == "orders")["foreign_keys"][0]
    assert fk["column"] == "customer_id"
    assert fk.get("null_rate", 0) > 0  # nulls in an FK column -> nullable FK hint


# ── build_config: end-to-end round trip ───────────────────────────────────────


def test_build_config_round_trips(tmp_path):
    config, ir = build_config(
        _frames(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_batches=1,
    )
    # config is usable as-is — no IR was authored by the caller
    assert ir["name"] == "profiled_model"
    Generator(data_generator_config_file=config).generate()

    customers = pl.read_parquet(next((tmp_path / "out" / "customers").rglob("*.parquet")))
    orders = pl.read_parquet(next((tmp_path / "out" / "orders").rglob("*.parquet")))

    # row volume preserved from the sample
    assert len(customers) == 60
    # categorical fidelity + FK integrity + fitted distribution
    assert set(customers["region"].to_list()) <= {"EU", "US", "APAC"}
    assert set(orders["customer_id"].drop_nulls().to_list()) <= set(customers["id"].to_list())
    assert 90 < orders["amount"].mean() < 110
    assert 0.1 < orders["note"].null_count() / len(orders) < 0.3


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_profile_from_directory(tmp_path):
    frames = _frames()
    samples = tmp_path / "samples"
    samples.mkdir()
    for name, df in frames.items():
        df.write_parquet(samples / f"{name}.parquet")

    out = tmp_path / "config.json"
    ir_out = tmp_path / "ir.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "datagenic",
            "profile",
            str(samples),
            "-o",
            str(out),
            "--emit-ir",
            str(ir_out),
            "--output-folder",
            str(tmp_path / "gen"),
            "--batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads(out.read_text())
    assert {t["table_name"] for t in config["tables"]} == {"customers", "orders"}
    assert ir_out.exists()
    # the emitted config generates
    Generator(data_generator_config_file=config).generate()
