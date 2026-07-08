# Domain-agnostic scaffolding: module selection + registered scaffold transforms

**Date:** 2026-07-08
**Status:** Design — approved surface, pending spec review
**Repos touched:** `datagenic` (core change), `opensee-core` (demo notebook only)

## Problem

`Configs` (`datagenic/scaffolding.py`) is meant to be domain-agnostic. Domain
knowledge is supposed to plug in *only* by dropping a folder under
`datagenic/domains/*` with a `register(registry)` hook — the registry docstring
states the contract: "Adding a new domain is dropping a folder with a
`register(registry)` hook; no change to [the core]."

`rolling_book` breaks that contract. Three pieces of `market_finance` vocabulary
live in the generic scaffolder:

- `Configs.__init__` params `rolling_book: bool` and `rolling_scenarios: int`
  (`scaffolding.py:351-352`).
- `_apply_rolling_book()` (`scaffolding.py:854`) hardcodes finance semantics:
  `trade_date`, `tenor`, `maturity_date`, `rolling_trade_id`, `rolling_scenario`,
  and the `"rolling_book"` provenance tag.
- The `scale_skip_global_key_sort` default-parameter toggle, gated on
  `rolling_book` (`scaffolding.py:426`).

A second, related gap: the registry auto-loads **every** domain under
`datagenic/domains/*` into one global `REGISTRY` unconditionally. There is no way
for a caller to say "scaffold this model using only the `market_finance` module."
As more business modules are added, their name patterns / generators would all
compete during inference with no way to scope them.

`reference_rows` (`scaffolding.py:353`) is genuinely generic (row count from key
cardinalities, no domain vocabulary) and stays as-is. This design is only about
the rolling-book trio and module scoping.

## Goals

1. Remove all `market_finance` vocabulary from `datagenic/scaffolding.py`.
2. Let a caller select which business modules are active for a scaffold, defaulting
   to today's behavior (all discovered) when unspecified.
3. Let a domain pack contribute **named scaffold transforms** (post-inference
   rewrites of the inferred semantics), owned and validated per module.
4. No change to the byte-identical generation invariant: the config produced by the
   new surface must be identical to the one the old `rolling_book=True` produced.

## Non-goals

- No CLI surface for transforms (none exists today; can be added later separately).
- No change to `reference_rows`, the inferrer contract, or generation.
- No new domain modules — `market_finance` remains the only pack.

## Public surface

A single new `Configs` parameter, `domains`, a **dict** keyed by domain-pack folder
name. Each value is a per-domain options dict whose only recognised key today is
`transforms` (a dict of `transform_name → options`).

```python
# notebook Step 2 — rolling book, default options
Configs(
    input=ir,
    semantic_inferrer="heuristic",
    domains={"market_finance": {"transforms": {"rolling_book": {}}}},
    output_format="parquet",
    ...
)

# old rolling_scenarios=3 -> a transform option
domains={"market_finance": {"transforms": {"rolling_book": {"scenarios": 3}}}}

# scope inference to one module, no transform
domains={"market_finance": {}}

# omit `domains` entirely -> all discovered modules active, no transforms (today's behavior)
Configs(input=ir, semantic_inferrer="heuristic", ...)

# future: several modules, each owning its transforms
domains={
    "market_finance": {"transforms": {"rolling_book": {"scenarios": 3}}},
    "credit_risk":    {"transforms": {"vintage_curves": {}}},
}
```

### Surface rules

- **Shape is dict-only.** One accepted shape, one parse path. Activating a module
  with no transforms is `{"market_finance": {}}`. A list form (`["market_finance"]`)
  is deliberately *not* accepted.
- **Default (omitted `domains`):** `None` → all discovered packs active, no
  transforms. Fully backward-compatible; existing callers, tests, and the notebook's
  other usages are unaffected.
- **Explicit empty (`domains={}`) is distinct from omitted.** `{}` means "I
  selected zero modules" → core generics only, no domain semantics. `None`
  (omitted) means "all discovered". The two must not be conflated.
- **Ownership is structural.** A transform can only appear inside the dict of the
  module that registered it. This makes "enable a transform whose module isn't
  active" unrepresentable — an entire error class is removed vs. a flat
  `scaffold_transforms=` param.
- **`rolling_scenarios` → `scenarios`.** The old top-level int becomes an option of
  the `rolling_book` transform.

### Validation / errors

- Unknown domain key (not a discovered pack) → `ValueError` listing discovered packs.
- Unknown key inside a domain's options dict (anything other than `transforms`) →
  `ValueError`.
- Transform name not registered by *that* domain → `ValueError` listing that
  domain's available transforms.

## Design

### 1. Registry: provenance, scoped reads, transforms (`semantics_registry.py`)

**Provenance.** `load()` stamps the current domain name around each pack's
`register()` call; `_register_core_generics` entries are stamped as always-on
(provenance sentinel, e.g. `None`/`"__core__"`). Each `add_*` records the
contributing domain. Provenance is stored in **parallel maps** so the existing
`generators` tuple shape — and the `SEMANTIC_GENERATORS` alias in `semantics.py` —
stays byte-for-byte unchanged.

```python
def load(self):
    ...
    _register_core_generics(self)          # provenance = core sentinel
    for mod in pkgutil.iter_modules(_domains.__path__):
        if mod.name.startswith("_"):
            continue
        self._current_domain = mod.name    # stamp
        importlib.import_module(f"datagenic.domains.{mod.name}").register(self)
    self._current_domain = None
```

**Scoped reads.** Read methods gain `active: set[str] | None` (`None` = all). An
entry is visible when its provenance is core, or is in `active`:

- `compiled(kind, active=None)`
- `links_for(active=None)`
- `value_classifiers(active=None)`
- a scoped generator lookup for `build_generator`
- `numeric_kinds` scoping as needed by `semantics_agent`

`active=None` reproduces today's behavior exactly (no filtering).

**New contribution type.**

```python
def add_scaffold_transform(self, name, fn):
    self.scaffold_transforms[name] = (fn, self._current_domain)

def scaffold_transform(self, name, active=None):
    """Return the transform fn, or raise if unknown / not owned by an active module."""
```

`discovered_domains()` helper returns the set of loaded pack names for validation.

### 2. Thread the active set through inference (`semantics.py`)

`resolve_inferrer(spec, active=None)` binds the inferrer closure to `active`, which
flows into the registry read calls it makes: `_match_by_type`, `_apply_links`, the
value classifiers, and `build_generator(...)`. **No global mutation** — the active
set is a per-call argument, so two `Configs` instances in one process can hold
different sets safely.

The module-level free functions (`_match_by_type`, `_apply_links`, `build_generator`)
gain an `active` parameter threaded from the inferrer. This threading is the largest
mechanical part of the change.

### 3. `Configs` (`scaffolding.py`)

- **Remove** `rolling_book` and `rolling_scenarios` from `__init__`.
- **Add** `domains: dict | None = None`.
- Parse/validate `domains` into `self._active: set[str] | None` and a
  `self._transforms: dict[str, dict]` (flattened `{transform_name: options}` across
  the active modules). `None` domains → `_active = None`, `_transforms = {}`.
- Pass `self._active` everywhere inference is invoked (`resolve_inferrer(...,
  active=self._active)` and each `build_generator` call in `_apply_semantic`).
- After `_compute_semantics()`, run each requested transform:

```python
ctx = ScaffoldContext(ir=self._ir, semantics=self._semantics,
                      default_parameters={})
for name, options in self._transforms.items():
    fn = REGISTRY.scaffold_transform(name, active=self._active)
    fn(ctx, **options)
# ctx.default_parameters overrides fold into config_template default_parameters
```

- `ScaffoldContext` is a small dataclass: `ir` (read), `semantics` (mutable — the
  transform rewrites records), `default_parameters` (mutable — the transform writes
  overrides like `scale_skip_global_key_sort`). The `scale_skip_global_key_sort`
  special-case at `scaffolding.py:426` is **deleted** from the core; the transform
  sets it on `ctx.default_parameters`.

After this, `scaffolding.py` contains zero finance vocabulary.

### 4. `market_finance` domain

- New module `datagenic/domains/market_finance/_transforms.py` with
  `apply_rolling_book(ctx, scenarios=1)` — the body of the old `_apply_rolling_book`,
  operating on `ctx.ir` / `ctx.semantics`, and setting
  `ctx.default_parameters["scale_skip_global_key_sort"] = True`. `scenarios` replaces
  the old `self.rolling_scenarios`.
- `_semantics.py` `register()` adds:
  `registry.add_scaffold_transform("rolling_book", apply_rolling_book)` (imported from
  `_transforms`; keep the import inside `register` to preserve numpy-free discovery if
  `_transforms` pulls anything heavy — the current body is pure dict manipulation, so
  a module-level import is fine).

## Migration inventory

| File | Change |
| --- | --- |
| `datagenic/datagenic/semantics_registry.py` | provenance, scoped reads, `add_scaffold_transform` / `scaffold_transform` / `discovered_domains` |
| `datagenic/datagenic/semantics.py` | thread `active` through inferrer + free functions |
| `datagenic/datagenic/scaffolding.py` | drop `rolling_book`/`rolling_scenarios`, add `domains`, `ScaffoldContext`, transform runner; delete the `scale_skip_global_key_sort` special-case |
| `datagenic/datagenic/domains/market_finance/_transforms.py` | new — `apply_rolling_book` |
| `datagenic/datagenic/domains/market_finance/_semantics.py` | register the transform |
| `datagenic/tests/market_finance/test_rolling_trades.py` | update all `rolling_book=`/`rolling_scenarios=` call sites to the new surface |
| `datagenic/README.md` | update the `rolling_book=True` example |
| `datagenic/datagenic/domains/market_finance/README.md` | update the `rolling_book=True` example + prose |
| `opensee-core/demo/finance_generate_to_push.ipynb` | Step 2 cell + the markdown that explains `rolling_book` |

No CLI flag exists for `rolling_book` today, so no CLI change.

## Testing

- **Registry units:** provenance stamping; scoped reads (`active=None` = all;
  `active={"market_finance"}` = core + finance; `active=set()` = core only);
  `add_scaffold_transform` / `scaffold_transform` lookup + error paths.
- **Parity (critical):** the config produced by
  `domains={"market_finance": {"transforms": {"rolling_book": {}}}}` is byte-identical
  to the one the old `rolling_book=True` produced. Lean on the existing
  `test_rolling_book_is_shard_byte_identical` and add a config-level parity assertion.
- **Scoping:** `domains={"market_finance": {}}` equals today's default (only one pack
  exists); a core-only scope (`domains={}` mapping to empty active, or an explicit
  empty selection) yields no finance semantics.
- **Errors:** unknown domain key; unknown per-domain key; transform not owned by an
  active module — each raises `ValueError` with a helpful message.
- **End-to-end:** `test_end_to_end_rolling_book` and `test_composite_key_rolling_book`
  pass unchanged in behavior after the call-site migration.

## Open risks

- Threading `active` through the inference free-functions (§2) is the bulk of the
  mechanical work and the main place a regression could hide; the `active=None`
  default must exactly reproduce current behavior. Parity + scoping tests guard this.
- `discovered_domains()` must reflect the same discovery `load()` uses, so validation
  can't drift from what's actually loadable.
