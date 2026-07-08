"""Cardinality-aware relationship enforcement (KeyEnforcer fanout + nullable FK).

Two opt-in options on a single-column FK rule: ``null_rate`` (optional FK) and
``fanout`` (per-parent child-count, target-count semantics). Both draw from an RNG
seeded off the postprocessor's _seed, so they are deterministic and batch-local —
the table stays shardable (KeyEnforcer INPUT_SCOPE='batch').
"""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datagenic import Configs, Generator
from datagenic.postprocessors import KeyEnforcer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parent(n, prefix="C"):
    return pl.DataFrame({"id": [f"{prefix}{i}" for i in range(n)]})


def _child(n):
    return pl.DataFrame({"order_id": list(range(n))})


# ── nullable FK ───────────────────────────────────────────────────────────────


def test_nullable_fk_nulls_a_fraction_and_keeps_pool_membership():
    pool = _parent(50)
    rules = [{"col": "customer_id", "from": "customers.id", "null_rate": 0.3}]
    out = KeyEnforcer().call(
        _child(200), {"customers": pool}, rules, _unicity_keys=["order_id"], _seed=7
    )
    frac = out["customer_id"].null_count() / len(out)
    assert 0.2 < frac < 0.4
    non_null = set(out["customer_id"].drop_nulls().to_list())
    assert non_null <= set(pool["id"].to_list())  # no orphans among the non-null values


# ── fan-out (target counts) ───────────────────────────────────────────────────


def test_fanout_shapes_per_parent_counts():
    # pool 100, rows 200 -> total target ~300 > 200 -> truncate path -> count <= max
    pool = _parent(100)
    rules = [{"col": "customer_id", "from": "customers.id", "fanout": {"min": 1, "max": 5}}]
    out = KeyEnforcer().call(
        _child(200), {"customers": pool}, rules, _unicity_keys=["order_id"], _seed=3
    )
    assert len(out) == 200
    assert set(out["customer_id"].to_list()) <= set(pool["id"].to_list())
    vc = out["customer_id"].value_counts()
    count_col = vc.columns[-1]
    assert vc[count_col].max() <= 5  # no parent exceeds the target max
    assert vc[count_col].n_unique() > 1  # multiplicity actually varies


def test_relationship_rules_are_deterministic():
    pool = _parent(80)
    rules = [
        {
            "col": "customer_id",
            "from": "customers.id",
            "null_rate": 0.25,
            "fanout": {"min": 1, "max": 4},
        }
    ]
    ke = KeyEnforcer()
    a = ke.call(_child(300), {"customers": pool}, rules, _unicity_keys=["order_id"], _seed=11)
    b = ke.call(_child(300), {"customers": pool}, rules, _unicity_keys=["order_id"], _seed=11)
    assert a["customer_id"].to_list() == b["customer_id"].to_list()


# ── fan-out distribution families (fanout["dist"]) ────────────────────────────


def _counts(pool, n, fanout, seed=0):
    out = KeyEnforcer._fanout_assign(
        [f"C{i}" for i in range(pool)], n, fanout, np.random.default_rng(seed)
    )
    assert len(out) == n
    assert set(out) <= {f"C{i}" for i in range(pool)}
    return Counter(out)


def test_fanout_default_dist_matches_explicit_uniform():
    """No ``dist`` key must behave exactly like ``dist="uniform"`` (backward compat)."""
    pool = [f"C{i}" for i in range(100)]
    a = KeyEnforcer._fanout_assign(pool, 200, {"min": 1, "max": 5}, np.random.default_rng(3))
    b = KeyEnforcer._fanout_assign(
        pool, 200, {"dist": "uniform", "min": 1, "max": 5}, np.random.default_rng(3)
    )
    assert a == b


def test_fanout_zipf_is_heavy_tailed():
    # zipf concentrates references on a few parents and leaves a long tail unreferenced,
    # unlike the flat uniform draw (whose counts cluster tightly around n/pool).
    zipf = _counts(200, 2000, {"dist": "zipf", "s": 2.0}, seed=2)
    uniform = _counts(200, 2000, {"dist": "uniform", "min": 1, "max": 5}, seed=2)
    mean = 2000 / 200
    assert max(zipf.values()) > 6 * mean  # a whale far above the flat mean
    assert max(zipf.values()) > 3 * max(uniform.values())  # far heavier tail than uniform
    # concentration: a handful of parents own most of the references (the "whale" shape)
    top5 = sum(c for _, c in zipf.most_common(5))
    assert top5 > 0.5 * 2000


def test_fanout_zipf_does_not_over_allocate():
    """A heavy-tailed draw must not build a multiset far larger than ``n`` (each
    per-parent count is capped at ``n``)."""
    counts = _counts(50, 500, {"dist": "zipf", "s": 1.2}, seed=5)
    assert max(counts.values()) <= 500


def test_fanout_zipf_is_deterministic():
    a = _counts(60, 300, {"dist": "zipf", "s": 2.0}, seed=7)
    b = _counts(60, 300, {"dist": "zipf", "s": 2.0}, seed=7)
    assert a == b


def test_fanout_unknown_dist_raises():
    with pytest.raises(ValueError, match="fanout"):
        KeyEnforcer._fanout_assign(["C0", "C1"], 4, {"dist": "bogus"}, np.random.default_rng(0))


def test_fanout_and_null_ignored_on_a_unicity_key():
    """A key column must not be nulled or fanned-out (would break dedup)."""
    pool = _parent(40)
    rules = [{"col": "k", "from": "customers.id", "null_rate": 0.5, "fanout": {"min": 1, "max": 5}}]
    out = KeyEnforcer().call(_child(100), {"customers": pool}, rules, _unicity_keys=["k"], _seed=1)
    assert out["k"].null_count() == 0  # never nulled
    assert out["k"].n_unique() == len(out)  # deduped to distinct keys, not fanned out


# ── scaffolding wiring ────────────────────────────────────────────────────────


def _ir():
    return {
        "name": "m",
        "tables": [
            {
                "type": "referential",
                "name": "customers",
                "keys": ["id"],
                "columns": [{"name": "id", "type": "string"}],
            },
            {
                "type": "fact",
                "name": "orders",
                "keys": ["order_id"],
                "columns": [
                    {"name": "order_id", "type": "int"},
                    {"name": "customer_id", "type": "string"},
                ],
                "foreign_keys": [
                    {
                        "column": "customer_id",
                        "references": "customers.id",
                        "null_rate": 0.2,
                        "fanout": {"dist": "uniform", "min": 1, "max": 5},
                    }
                ],
            },
        ],
    }


def test_ir_fk_hints_become_keyenforcer_rule_options():
    cfg = Configs(input=_ir(), number_of_batches=1).config_template
    orders = next(t for t in cfg["tables"] if t["table_name"] == "orders")
    rule = next(
        r for r in orders["post_processor"]["udf_args"]["rules"] if r["col"] == "customer_id"
    )
    assert rule["null_rate"] == 0.2
    assert rule["fanout"] == {"dist": "uniform", "min": 1, "max": 5}


def test_end_to_end_nullable_fk_has_no_orphans(tmp_path):
    cfg = Configs(
        input=_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=500,
        number_of_rows_in_non_fact_tables=100,
        number_of_batches=1,
    ).config_template
    Generator(data_generator_config_file=cfg).generate()

    customers = pl.read_parquet(next((tmp_path / "out" / "customers").rglob("*.parquet")))
    orders = pl.read_parquet(next((tmp_path / "out" / "orders").rglob("*.parquet")))
    pool = set(customers["id"].to_list())
    non_null = orders["customer_id"].drop_nulls()

    assert set(non_null.to_list()) <= pool  # every non-null FK is a real parent
    frac = orders["customer_id"].null_count() / len(orders)
    assert 0.1 < frac < 0.3  # ~0.2


def test_nullable_fk_fanout_config_is_distributable(tmp_path):
    """The new rules are batch-local, so check-distributable must still pass."""
    cfg = Configs(
        input=_ir(),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_fact_tables=200,
        number_of_rows_in_non_fact_tables=50,
        number_of_batches=2,
    ).config_template
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "datagenic", "check-distributable", str(cfg_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
