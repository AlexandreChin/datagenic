"""Domain registry: domain packs self-register their semantics; the core names none.

`datagenic.semantics` drives inference off `datagenic.semantics_registry.REGISTRY`,
which is populated by discovering `datagenic.domains.*` packs (each exposing a
`register(registry)` hook) plus a few domain-neutral core generics. So the core
inference module hardcodes no domain, and dropping a new domain folder under
`datagenic/domains/` is all it takes to extend inference.
"""

from pathlib import Path

import datagenic.semantics as sem
from datagenic.semantics_registry import REGISTRY, SemanticRegistry

_SEMANTICS_SRC = Path(sem.__file__).read_text()


def test_semantics_core_names_no_domain():
    # the decoupling guarantee: the core inference module mentions no specific domain.
    assert "market_finance" not in _SEMANTICS_SRC


def test_registry_autodiscovers_market_finance():
    r = SemanticRegistry()
    r.load()
    # finance semantics registered purely via discovery — nothing hardcoded in core
    assert "asset_class" in r.generators
    assert r.generators["isin"][0] == "datagenic.domains.market_finance.generators"
    # name patterns present, gated by type
    assert any(s == "asset_class" for _, s, _ in r.patterns["string"])
    assert any(s == "notional" for _, s, _ in r.patterns["float"])
    assert any(s == "mtf" for _, s, _ in r.patterns["array_float"])
    assert any(s == "trade_date" for _, s, _ in r.patterns["date"])
    # parent-linking rules populated
    assert "instrument_type" in r.links["asset_class"]
    # numeric fit-kinds recorded (so the fitter dispatches without importing priors)
    assert r.numeric_kinds["notional"] == "scalar_conditional"
    assert r.numeric_kinds["pnl_series"] == "array_conditional"


def test_core_generics_present_without_any_domain():
    r = SemanticRegistry()
    r.load()
    assert r.generators["identifier"][1] == "UniqueIncrementalStrings"
    assert r.generators["identifier_int"][1] == "UniqueIncrementalIntegers"
    assert any(s == "identifier" for _, s, _ in r.patterns["string"])
    assert any(s == "identifier_int" for _, s, _ in r.patterns["int"])


def test_a_new_domain_only_needs_a_register_hook():
    # a fresh domain adds entries through the same register(registry) API; no core edit.
    r = SemanticRegistry()

    def register(reg):
        reg.add_generator(
            "widget_id", "datagenic.generators", "UniqueIncrementalStrings", {"prefix": "W_"}
        )
        reg.add_pattern("string", r"widget[_ ]?id", "widget_id", 0.9)

    register(r)
    compiled = r.compiled("string")
    assert any(rx.search("widgetId") and s == "widget_id" for rx, s, _ in compiled)


def test_importing_semantics_eagerly_loads_the_shared_registry():
    # importing datagenic.semantics populates the shared REGISTRY via discovery, so the
    # public SEMANTIC_GENERATORS map (used by build_generator and tests) is ready.
    assert REGISTRY._loaded
    assert "asset_class" in sem.SEMANTIC_GENERATORS
    assert sem.SEMANTIC_GENERATORS is REGISTRY.generators
