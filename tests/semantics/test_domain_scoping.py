"""Module selection + scaffold transforms: a caller scopes scaffolding to chosen domain
packs (``Configs(domains=...)``) and opts into their post-inference transforms. The
registry stamps provenance on every contribution and filters reads by an active-module
scope; ``None`` = all discovered, an empty set = core only.
"""

import pytest

from datagenic import Configs
from datagenic.semantics_registry import CORE, ScaffoldContext, SemanticRegistry

# ── registry-level: provenance + scoped reads ─────────────────────────────────


def test_scoped_reads_filter_by_active_module():
    r = SemanticRegistry()
    r.load()
    # patterns: core `identifier` is always visible; finance `asset_class` only in scope
    core_only = {s for _, s, _ in r.compiled("string", active=set())}
    with_fin = {s for _, s, _ in r.compiled("string", active={"market_finance"})}
    all_active = {s for _, s, _ in r.compiled("string", active=None)}
    assert "identifier" in core_only and "asset_class" not in core_only
    assert "asset_class" in with_fin
    assert with_fin == all_active  # only one pack today

    # generators + links honour the same scope
    assert r.generator_for("identifier", active=set()) is not None
    assert r.generator_for("asset_class", active=set()) is None
    assert r.generator_for("asset_class", active={"market_finance"}) is not None
    assert r.links_for(active=set()) == {}
    assert "asset_class" in r.links_for(active={"market_finance"})


def test_core_generics_carry_the_core_sentinel():
    r = SemanticRegistry()
    r.load()
    assert r._gen_domain["identifier"] is CORE
    assert r._gen_domain["asset_class"] == "market_finance"
    assert r.discovered_domains() == {"market_finance"}


# ── registry-level: scaffold transforms ───────────────────────────────────────


def test_scaffold_transform_lookup_and_scope():
    r = SemanticRegistry()
    r.load()
    assert callable(r.scaffold_transform("rolling_book", active={"market_finance"}))
    assert callable(r.scaffold_transform("rolling_book", active=None))
    assert r.transforms_for_domain("market_finance") == {"rolling_book"}


def test_scaffold_transform_errors_out_of_scope_or_unknown():
    r = SemanticRegistry()
    r.load()
    with pytest.raises(ValueError, match="rolling_book"):
        r.scaffold_transform("rolling_book", active=set())  # finance not active
    with pytest.raises(ValueError, match="unknown scaffold transform"):
        r.scaffold_transform("does_not_exist", active=None)


def test_scaffold_context_defaults():
    ctx = ScaffoldContext(ir={"tables": []}, semantics={})
    assert ctx.default_parameters == {}


# ── Configs surface: scoping ──────────────────────────────────────────────────


def _fact_ir(columns, keys):
    return {
        "name": "m",
        "tables": [{"type": "fact", "name": "t", "keys": keys, "columns": columns}],
    }


def _scaffold(ir, tmp_path, **extra):
    return Configs(
        input=ir,
        output_format="parquet",
        output_folder_path=str(tmp_path),
        number_of_rows_in_fact_tables=10,
        number_of_batches=1,
        start_date="2024-01-01",
        end_date="2024-01-02",
        seed=1,
        semantic_inferrer="heuristic",
        **extra,
    ).config_template


def _gen(cfg, col):
    return next(
        c["column_generator"]["udf_name"]
        for c in cfg["tables"][0]["columns"]
        if c["col_name"] == col
    )


def test_selecting_market_finance_infers_finance_semantics(tmp_path):
    ir = _fact_ir([{"name": "asset_class", "type": "string"}], keys=[])
    cfg = _scaffold(ir, tmp_path, domains={"market_finance": {}})
    assert _gen(cfg, "asset_class") == "FinancialAssetClass"


def test_core_only_scope_gets_no_finance_semantics(tmp_path):
    ir = _fact_ir([{"name": "asset_class", "type": "string"}], keys=[])
    cfg = _scaffold(ir, tmp_path, domains={})  # explicit empty = core only
    assert _gen(cfg, "asset_class") != "FinancialAssetClass"


def test_omitting_domains_matches_selecting_the_only_pack(tmp_path):
    ir = _fact_ir([{"name": "asset_class", "type": "string"}], keys=[])
    default = _scaffold(ir, tmp_path, domains=None)
    selected = _scaffold(ir, tmp_path, domains={"market_finance": {}})
    assert _gen(default, "asset_class") == _gen(selected, "asset_class") == "FinancialAssetClass"


# ── Configs surface: validation errors ────────────────────────────────────────


def test_unknown_domain_raises(tmp_path):
    ir = _fact_ir([{"name": "x", "type": "string"}], keys=[])
    with pytest.raises(ValueError, match="unknown domain 'credit_risk'"):
        _scaffold(ir, tmp_path, domains={"credit_risk": {}})


def test_unknown_per_domain_key_raises(tmp_path):
    ir = _fact_ir([{"name": "x", "type": "string"}], keys=[])
    with pytest.raises(ValueError, match="only 'transforms' is supported"):
        _scaffold(ir, tmp_path, domains={"market_finance": {"transformz": {}}})


def test_transform_not_owned_by_domain_raises(tmp_path):
    ir = _fact_ir([{"name": "x", "type": "string"}], keys=[])
    with pytest.raises(ValueError, match="no transform 'vintage_curves'"):
        _scaffold(ir, tmp_path, domains={"market_finance": {"transforms": {"vintage_curves": {}}}})


def test_domains_must_be_a_dict(tmp_path):
    ir = _fact_ir([{"name": "x", "type": "string"}], keys=[])
    with pytest.raises(TypeError, match="domains must be a dict"):
        _scaffold(ir, tmp_path, domains=["market_finance"])
