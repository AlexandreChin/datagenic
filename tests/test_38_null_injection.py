"""Null / missing-value injection.

A column may declare an opt-in ``null_spec`` (scaffolded from the IR hints
``null_rate`` / ``null_when``). The engine nulls a fraction of the column's values
from a per-(table, column, date, batch) RNG — deterministic, shard-safe, and never
applied to a unicity key (rejected at validate()).
"""

import polars as pl
import pytest

from datagenic import Configs, Generator


def _ir(columns, keys=("id",), name="t", table_type="referential"):
    return {
        "name": "m",
        "tables": [{"type": table_type, "name": name, "keys": list(keys), "columns": columns}],
    }


def _scaffold(columns, tmp_path, rows=1000, **kw):
    cfg = Configs(
        input=_ir(columns, **kw),
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_rows_in_non_fact_tables=rows,
        number_of_rows_in_fact_tables=rows,
        number_of_batches=1,
    ).config_template
    return cfg


def _read(tmp_path, table="t"):
    return pl.read_parquet(next((tmp_path / "out" / table).rglob("*.parquet")))


# ── scaffolding wiring ────────────────────────────────────────────────────────


def test_ir_null_rate_becomes_null_spec():
    cfg = Configs(
        input=_ir(
            [{"name": "id", "type": "int"}, {"name": "amount", "type": "float", "null_rate": 0.2}]
        ),
        number_of_batches=1,
    ).config_template
    cols = {c["col_name"]: c for c in cfg["tables"][0]["columns"]}
    assert cols["amount"]["null_spec"] == {"rate": 0.2}
    assert "null_spec" not in cols["id"]  # untouched


def test_no_null_hint_leaves_no_null_spec():
    cfg = Configs(input=_ir([{"name": "id", "type": "int"}]), number_of_batches=1).config_template
    assert "null_spec" not in cfg["tables"][0]["columns"][0]


# ── MCAR (rate) ───────────────────────────────────────────────────────────────


def test_mcar_rate_nulls_about_the_right_fraction(tmp_path):
    cfg = _scaffold(
        [{"name": "id", "type": "int"}, {"name": "amount", "type": "float", "null_rate": 0.3}],
        tmp_path,
        rows=1000,
    )
    Generator(data_generator_config_file=cfg).generate()
    df = _read(tmp_path)

    frac = df["amount"].null_count() / len(df)
    assert 0.22 < frac < 0.38  # ~0.3, generous tolerance
    assert df["id"].null_count() == 0  # key never nulled


def test_fully_null_column_keeps_dtype(tmp_path):
    cfg = _scaffold(
        [{"name": "id", "type": "int"}, {"name": "amount", "type": "float", "null_rate": 1.0}],
        tmp_path,
        rows=200,
    )
    Generator(data_generator_config_file=cfg).generate()  # schema validation must pass
    df = _read(tmp_path)
    assert df["amount"].null_count() == len(df)
    assert df["amount"].dtype == pl.Float64  # dtype preserved despite all-null


# ── MAR (when) ────────────────────────────────────────────────────────────────


def test_mar_when_nulls_only_matching_rows(tmp_path):
    # note is nulled (rate defaults to 1.0) exactly where id > 100
    cfg = _scaffold(
        [
            {"name": "id", "type": "int"},
            {"name": "note", "type": "string", "null_when": "_table['id'] > 100"},
        ],
        tmp_path,
        rows=300,
    )
    Generator(data_generator_config_file=cfg).generate()
    df = _read(tmp_path)

    assert df.filter(pl.col("id") <= 100)["note"].null_count() == 0
    high = df.filter(pl.col("id") > 100)
    assert high["note"].null_count() == len(high)  # all matching rows null


# ── determinism ───────────────────────────────────────────────────────────────


def test_null_positions_are_deterministic(tmp_path):
    cfg = _scaffold(
        [{"name": "id", "type": "int"}, {"name": "amount", "type": "float", "null_rate": 0.4}],
        tmp_path,
        rows=500,
    )
    g = Generator(data_generator_config_file=cfg)
    g.generate(force_regenerate=True)
    first = _read(tmp_path)["amount"].is_null().to_list()
    g.generate(force_regenerate=True)
    second = _read(tmp_path)["amount"].is_null().to_list()
    assert first == second


def test_null_mask_does_not_perturb_other_columns(tmp_path):
    """Adding null_spec to one column must not change another column's values
    (the mask uses its own per-column RNG)."""
    base = _scaffold(
        [{"name": "id", "type": "int"}, {"name": "amount", "type": "float"}],
        tmp_path / "a",
        rows=300,
    )
    Generator(data_generator_config_file=base).generate()
    amount_no_null = pl.read_parquet(next((tmp_path / "a" / "out" / "t").rglob("*.parquet")))[
        "amount"
    ].to_list()

    withnull = _scaffold(
        [
            {"name": "id", "type": "int"},
            {"name": "amount", "type": "float"},
            {"name": "other", "type": "float", "null_rate": 0.5},
        ],
        tmp_path / "b",
        rows=300,
    )
    Generator(data_generator_config_file=withnull).generate()
    amount_with_null = pl.read_parquet(next((tmp_path / "b" / "out" / "t").rglob("*.parquet")))[
        "amount"
    ].to_list()

    assert amount_no_null == amount_with_null  # 'amount' unchanged by 'other's nulls


# ── validation ────────────────────────────────────────────────────────────────


def test_null_spec_on_unicity_key_is_rejected(tmp_path):
    cfg = _scaffold([{"name": "id", "type": "int"}, {"name": "x", "type": "float"}], tmp_path)
    # force a null_spec onto the key column
    for col in cfg["tables"][0]["columns"]:
        if col["col_name"] == "id":
            col["null_spec"] = {"rate": 0.5}
    with pytest.raises(ValueError, match="unicity key"):
        Generator(data_generator_config_file=cfg).generate()


def test_bad_null_rate_is_rejected(tmp_path):
    cfg = _scaffold([{"name": "id", "type": "int"}, {"name": "x", "type": "float"}], tmp_path)
    for col in cfg["tables"][0]["columns"]:
        if col["col_name"] == "x":
            col["null_spec"] = {"rate": 1.5}
    with pytest.raises(ValueError, match=r"rate must be a number in \[0, 1\]"):
        Generator(data_generator_config_file=cfg).generate()
