"""Semantic inference for scaffolding: guess each column's domain meaning so the
scaffolder can wire a realistic generator automatically, without the user
hand-tagging the IR.

This module is **domain-agnostic**: it names no specific domain. All domain content
(semantic → generator mappings, name patterns, coherence links, value classifiers)
is contributed by domain packs under ``datagenic/domains/*`` through
``datagenic.semantics_registry`` — discovered and loaded once at import here. Adding a
new domain is dropping a folder with a ``register(registry)`` hook; no change to this
module.

This runs at scaffold time only (``Configs``), so it is *not* subject to the
byte-identical generation invariant — the inference produces a config that the user
then reviews and may hand-edit; generation from the frozen config stays deterministic
regardless.

Inferrer contract::

    fn(table_name: str, columns: list[dict], samples=None) -> dict[col_name, record]

where ``columns`` is a list of ``{"name", "type"}`` and each record is::

    {"semantic": str, "confidence": float, "parent": str | None,
     "alternatives": list[str], "by": str}

``resolve_inferrer`` maps a CLI/string spec to a callable. Built-ins: ``heuristic``
(name-based, dependency-free), ``values`` (sample-value based, with a name-heuristic
fallback) and ``agent`` (opt-in LLM via ``datagenic.semantics_agent``; needs the
anthropic SDK, lazy-imported only when invoked).
"""

import functools
import importlib

from datagenic.semantics_registry import REGISTRY

# Populate the registry once (core generics + discovered domain packs). Registration
# records only strings/patterns and each pack's pure data, so this stays numpy-free —
# domain generator classes are imported later, only when build_generator builds one.
REGISTRY.load()

# Public alias: semantic -> (udf_module, udf_class, base_args, parent_arg | None). Same
# dict object the registry owns, kept for the historical import path used by scaffolding.
SEMANTIC_GENERATORS = REGISTRY.generators


def _is_string_type(col_type: str | None) -> bool:
    """Categorical/identifier semantics apply to string-typed columns only."""
    return bool(col_type) and "string" in col_type.lower()


def _is_float_array_type(col_type: str | None) -> bool:
    """Array-quantity semantics apply to array(float) columns — width-agnostic, so
    array(float32) / array(float64) count too (the hand-authored IR uses the sized
    variants)."""
    t = col_type.lower() if col_type else ""
    return "array" in t and "float" in t


def _is_float_type(col_type: str | None) -> bool:
    """Scalar numeric semantics apply to *scalar* float columns only — not ints
    (keys/counts) and not array(float) (which get the array-quantity semantics)."""
    t = col_type.lower() if col_type else ""
    return "float" in t and "array" not in t


def _is_date_type(col_type: str | None) -> bool:
    """Date semantics apply to date / datetime columns."""
    return bool(col_type) and "date" in col_type.lower()


def _is_int_type(col_type: str | None) -> bool:
    """Integer identifier semantics apply to *scalar* int columns only — not floats
    and not array(int)."""
    t = col_type.lower() if col_type else ""
    return "int" in t and "float" not in t and "array" not in t


def _match_from(compiled, name: str) -> tuple[str | None, float]:
    for pattern, semantic, confidence in compiled:
        if pattern.search(name):
            return semantic, confidence
    return None, 0.0


def _match_by_type(name: str, col_type: str | None, active=None) -> tuple[str | None, float]:
    """Name-based match dispatched on column type: string -> categorical/id,
    array(float) -> array quantity, scalar float -> scalar quantity, date/datetime ->
    date semantics, scalar int -> integer identifier. The type (never the name) decides
    which pattern set is consulted, so a name like "pnl" resolves to the scalar or the
    array semantic by whether the column is a float or an array(float), and "productId"
    resolves to `identifier` (string) or `identifier_int` (int). Patterns come from the
    domain registry, scoped to the ``active`` modules. Other types are never matched."""
    if _is_string_type(col_type):
        return _match_from(REGISTRY.compiled("string", active), name)
    if _is_float_array_type(col_type):
        return _match_from(REGISTRY.compiled("array_float", active), name)
    if _is_float_type(col_type):
        return _match_from(REGISTRY.compiled("float", active), name)
    if _is_date_type(col_type):
        return _match_from(REGISTRY.compiled("date", active), name)
    if _is_int_type(col_type):
        return _match_from(REGISTRY.compiled("int", active), name)
    return None, 0.0


def heuristic_inferrer(table_name, columns, samples=None, active=None):
    """Infer column semantics from column names (and types) alone.

    Dependency-free and deterministic. ``samples`` is accepted for interface
    parity with richer inferrers but unused here. ``active`` scopes which domain
    modules' patterns/links participate (``None`` = all discovered).
    """
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        semantic, confidence = _match_by_type(str(name), col.get("type"), active)
        if semantic is None:
            continue
        result[name] = {
            "semantic": semantic,
            "confidence": confidence,
            "parent": None,
            "alternatives": [],
            "by": "heuristic",
        }

    return _link_hierarchy_parents(result, active)


def _link_hierarchy_parents(result: dict, active=None) -> dict:
    """Point each coherent child record at the column it must read (in place), so the
    generator stays consistent with that parent — using the parent→children links the
    domain packs registered (e.g. the asset_class hierarchy + conditional numerics,
    counterparty attributes, trade-date-derived dates), scoped to the ``active`` modules.
    A parent the inferrer already chose (e.g. the agent conditioning a value on a
    different column) is kept."""
    for parent_semantic, children in REGISTRY.links_for(active).items():
        parent_col = next(
            (n for n, rec in result.items() if rec["semantic"] == parent_semantic), None
        )
        if parent_col is None:
            continue
        for rec in result.values():
            if rec["semantic"] in children and not rec.get("parent"):
                rec["parent"] = parent_col
    return result


# ── Value-based inference (Phase 2) ──────────────────────────────────────────

# Minimum fraction of sample values that must match for a semantic to be assigned.
_VALUE_THRESHOLD = 0.9

# Built lazily from the registry's classifier factories on first use, so importing this
# module (and value-free inference) pulls in no numpy-backed validators. Cached per
# scope (the active-module set) so a scoped run only classifies with its modules.
_VALUE_CLASSIFIERS: dict = {}


def _classify_values(values, active=None) -> tuple[str | None, float]:
    """Classify a column from its sample values. Returns the most specific
    semantic whose match fraction clears the threshold, with that fraction as the
    confidence; ``(None, 0.0)`` if nothing matches."""
    vals = [str(v) for v in values if v is not None and str(v) != ""]
    if not vals:
        return None, 0.0
    key = None if active is None else frozenset(active)
    if key not in _VALUE_CLASSIFIERS:
        _VALUE_CLASSIFIERS[key] = REGISTRY.value_classifiers(active)

    best_semantic, best_fraction = None, 0.0
    for semantic, predicate in _VALUE_CLASSIFIERS[key]:  # specific first; strict > keeps it on ties
        fraction = sum(predicate(v) for v in vals) / len(vals)
        if fraction >= _VALUE_THRESHOLD and fraction > best_fraction:
            best_semantic, best_fraction = semantic, fraction
    if best_semantic is None:
        return None, 0.0
    return best_semantic, round(min(0.99, best_fraction), 2)


# ── Numeric distribution fitting (Phase 4) ────────────────────────────────────

# Minimum per-group sample count before a conditional case is fitted (thinner
# groups fall back to the pooled default).
_MIN_GROUP_SAMPLES = 30
# mean/median ratio above which an all-positive column is treated as lognormal.
# For a lognormal this equals exp(sigma**2/2) (≈1.13 at sigma=0.5, ≈1.65 at 1.0);
# a symmetric distribution sits at ≈1.0.
_LOGNORMAL_SKEW = 1.2


def fit_distribution(values) -> dict | None:
    """Fit a ``{distribution, mu, sigma}`` spec to numeric sample values.

    Picks **lognormal** (``mu`` = median in real units, ``sigma`` = log-space
    spread) when the values are all-positive and right-skewed (mean pulled above
    the median), else **normal** (``mu`` = mean, ``sigma`` = std). The
    parameterisation matches ``generators._draw_floats`` exactly, so
    generate→profile→regenerate round-trips. ``None`` when nothing to fit.
    """
    import numpy as np

    x = np.asarray([v for v in values if v is not None], dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    median = float(np.median(x))
    right_skewed = median > 0 and float(x.mean()) / median >= _LOGNORMAL_SKEW
    if bool((x > 0).all()) and x.size >= 2 and right_skewed:
        logs = np.log(x)
        return {
            "distribution": "lognormal",
            "mu": round(float(np.exp(logs.mean())), 6),
            "sigma": round(float(logs.std()), 6),
        }
    return {
        "distribution": "normal",
        "mu": round(float(x.mean()), 6),
        "sigma": round(float(x.std()) if x.size > 1 else 0.0, 6),
    }


def _fit_conditional(child_values, parent_values) -> dict | None:
    """Fit ``{cases, default}`` for ``ConditionalFloats`` from row-aligned child /
    parent samples: a per-category distribution for each parent value with enough
    samples, plus a pooled ``default`` covering thin/unseen categories."""
    n = min(len(child_values), len(parent_values))
    groups: dict[str, list] = {}
    for child, parent in zip(child_values[:n], parent_values[:n], strict=True):
        if child is None or parent is None:
            continue
        groups.setdefault(str(parent), []).append(child)

    default = fit_distribution(child_values)
    if default is None:
        return None
    cases = {}
    for value, vals in groups.items():
        if len(vals) >= _MIN_GROUP_SAMPLES:
            spec = fit_distribution(vals)
            if spec is not None:
                cases[value] = spec
    return {"cases": cases, "default": default}


def _flatten_arrays(arrays) -> list:
    """All elements of a list-of-arrays sample, skipping null rows/elements."""
    return [v for arr in arrays if arr is not None for v in arr if v is not None]


def _array_length(arrays) -> int | None:
    """Representative (median) array length across the sample rows."""
    import statistics

    lengths = [len(arr) for arr in arrays if arr is not None]
    return round(statistics.median(lengths)) if lengths else None


def fit_array_distribution(arrays) -> dict | None:
    """Fit ``{distribution, mu, sigma, size_of_arrays}`` for an array-quantity
    column: the element family/params (flattening all rows) plus the array length.
    ``None`` when there is nothing to fit."""
    spec = fit_distribution(_flatten_arrays(arrays))
    if spec is None:
        return None
    size = _array_length(arrays)
    if size:
        spec = {**spec, "size_of_arrays": size}
    return spec


def _fit_conditional_array(child_arrays, parent_values) -> dict | None:
    """Fit ``{cases, default, size_of_arrays}`` for ``ConditionalArrayFloats`` from
    row-aligned array/parent samples: per-category element params (enough elements),
    pooled ``default``, and a shared array length."""
    n = min(len(child_arrays), len(parent_values))
    groups: dict[str, list] = {}
    for arr, parent in zip(child_arrays[:n], parent_values[:n], strict=True):
        if arr is None or parent is None:
            continue
        groups.setdefault(str(parent), []).append(arr)

    default = fit_distribution(_flatten_arrays(child_arrays))
    if default is None:
        return None
    cases = {}
    for value, arrs in groups.items():
        flat = _flatten_arrays(arrs)
        if len(flat) >= _MIN_GROUP_SAMPLES:
            spec = fit_distribution(flat)
            if spec is not None:
                cases[value] = spec
    fitted = {"cases": cases, "default": default}
    size = _array_length(child_arrays)
    if size:
        fitted["size_of_arrays"] = size
    return fitted


def _fit_numeric(result: dict, samples: dict) -> dict:
    """Attach fitted distribution args (in place) to numeric-semantic records using the
    sample values, dispatching on the semantic's registered numeric fit-kind — conditional
    (per parent category) for conditional semantics, flat otherwise. Overrides the curated
    priors at ``build_generator``."""
    for col, rec in result.items():
        values = samples.get(col)
        if not values:
            continue
        kind = REGISTRY.numeric_kinds.get(rec["semantic"])
        if kind == "scalar_conditional":
            parent_values = samples.get(rec.get("parent"))
            fitted = _fit_conditional(values, parent_values) if parent_values else None
            if fitted is None:  # no usable parent -> still fit a data-driven default
                spec = fit_distribution(values)
                fitted = {"default": spec} if spec else None
            if fitted:
                rec["args"] = fitted
        elif kind == "array_conditional":
            parent_values = samples.get(rec.get("parent"))
            fitted = _fit_conditional_array(values, parent_values) if parent_values else None
            if fitted is None:  # no usable parent -> pooled element fit as default
                spec = fit_distribution(_flatten_arrays(values))
                fitted = {"default": spec} if spec else None
            if fitted:
                rec["args"] = fitted
        elif kind == "scalar":
            spec = fit_distribution(values)
            if spec:
                rec["args"] = spec
        elif kind == "array":
            fitted = fit_array_distribution(values)
            if fitted:
                rec["args"] = fitted
    return result


def values_inferrer(table_name, columns, samples=None, active=None):
    """Infer semantics from real sample values, falling back to the name
    heuristic for columns with no sample evidence (so one pass covers everything).
    Numeric quantities additionally have their distribution **fitted** from the
    samples (family + params, per parent category when conditional).

    ``samples`` is ``{col_name: [values]}`` (as normalised by ``Configs``); ``active``
    scopes which domain modules participate (``None`` = all discovered).
    """
    samples = samples or {}
    result: dict[str, dict] = {}
    for col in columns:
        name = col.get("name", "")
        semantic, confidence, by = None, 0.0, "values"

        if name in samples:
            semantic, confidence = _classify_values(samples[name], active)
        if semantic is None:
            semantic, confidence = _match_by_type(str(name), col.get("type"), active)
            by = "heuristic"
        if semantic is None:
            continue

        result[name] = {
            "semantic": semantic,
            "confidence": confidence,
            "parent": None,
            "alternatives": [],
            "by": by,
        }

    _link_hierarchy_parents(result, active)  # sets parents so conditional fits can group
    return _fit_numeric(result, samples)


def _agent_inferrer(table_name, columns, samples=None, active=None):
    """Lazy delegator to the LLM-backed inferrer. Imported on first call only, so
    neither this module nor ``scaffolding`` pulls in the anthropic SDK at import.
    ``active`` is accepted for interface parity; the agent proposes semantics directly
    rather than matching registered patterns, so module scoping does not constrain it."""
    from datagenic.semantics_agent import agent_inferrer

    return agent_inferrer(table_name, columns, samples=samples)


_BUILTINS = {
    "heuristic": heuristic_inferrer,
    "values": values_inferrer,
    "agent": _agent_inferrer,
}


def resolve_inferrer(spec, active=None):
    """Resolve an inferrer spec to a callable.

    ``spec`` may be a callable (returned as-is), ``None`` (returns ``None`` = no
    inference), a built-in name (``heuristic`` / ``values`` / ``agent``), or a
    ``module:function`` path (resolved like the scaffolder's ``--adaptor``).

    ``active`` (the selected-module scope, ``None`` = all) is bound into the built-in
    inferrers so inference only consults those modules' patterns/links/classifiers.
    User-supplied inferrers (a callable or ``module:function``) propose semantics
    directly, so scoping is not applied to them.
    """
    if spec is None or callable(spec):
        return spec
    if spec in _BUILTINS:
        return functools.partial(_BUILTINS[spec], active=active)
    if ":" in spec:
        module_name, _, func_name = spec.partition(":")
        return getattr(importlib.import_module(module_name), func_name)
    raise ValueError(
        f"unknown semantic inferrer {spec!r}; expected a built-in "
        f"({', '.join(_BUILTINS)}) or a 'module:function' path"
    )


def build_generator(
    semantic: str, parent: str | None = None, overrides: dict | None = None, active=None
) -> dict:
    """Build the ``column_generator`` dict for a recognised semantic.

    Introspects the target generator's ``call`` defaults (same as ``Columns``
    does for the built-in generators) and wires the parent column into the
    coherence arg when applicable. ``overrides`` (e.g. distribution params fitted
    from a sample) take precedence over the curated priors. ``active`` scopes the
    lookup to the selected modules (``None`` = all). Raises ``KeyError`` for a
    semantic unknown to (or out of scope for) the active modules.
    """
    spec = REGISTRY.generator_for(semantic, active)
    if spec is None:
        raise KeyError(semantic)
    module_name, class_name, base_args, parent_arg = spec
    from datagenic.iokit import func

    gen_cls = getattr(importlib.import_module(module_name), class_name)
    args = func.get_default_args(gen_cls().call) | dict(base_args)
    if parent_arg and parent:
        args[parent_arg] = parent
    if overrides:
        args.update(overrides)
    return {"udf_module": module_name, "udf_name": class_name, "args": args}
