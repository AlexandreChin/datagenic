"""Domain-pack registry for scaffold-time semantic inference.

Core infrastructure — imports **no** domain code and no numpy/polars. Domain packs
under ``datagenic/domains/*`` each expose a ``register(registry)`` hook that adds
their contributions:

- ``add_generator`` — a semantic → column-generator spec (module, class, base args,
  the parent-coherence arg, and an optional numeric fit-kind).
- ``add_pattern`` — an ordered, type-gated name pattern (``string`` / ``float`` /
  ``array_float`` / ``date`` / ``int``) mapping a regex to a semantic + confidence.
- ``add_link`` — parent-semantic → child-semantics coherence links.
- ``add_value_classifiers`` — a *factory* (called lazily) yielding value-based
  ``(semantic, predicate)`` classifiers, so heavy imports (numpy validators) stay
  deferred until value inference actually runs.
- ``add_scaffold_transform`` — a named post-inference transform the caller can opt
  into (``Configs(domains={"pack": {"transforms": {...}}})``); it rewrites the
  inferred semantics after inference (e.g. the finance rolling-book remap).

``datagenic.semantics`` drives inference entirely off the populated registry, so it
names no specific domain and a new domain folder is discovered automatically.
Discovery is done once, on first ``load()`` (which ``datagenic.semantics`` triggers
at import) — registration records only strings/patterns and imports each pack's pure
data, so it stays numpy-free.

**Provenance & scoping.** ``load()`` stamps the contributing domain onto every entry
(core generics carry the ``CORE`` sentinel and are always visible). The read API
accepts an ``active: set[str] | None`` scope so a caller can restrict inference to a
chosen subset of modules: ``None`` = all discovered packs, an empty set = core only,
``{"market_finance"}`` = core + that pack. Scoping is a per-call argument — the shared
global registry is never mutated — so two ``Configs`` in one process can hold
different scopes safely.
"""

import importlib
import pkgutil
import re
from dataclasses import dataclass, field


@dataclass
class ScaffoldContext:
    """What a scaffold transform receives. ``ir`` is the neutral model (read-only for
    the transform's purposes); ``semantics`` is the inferred ``{(table, column): record}``
    map, which the transform rewrites in place; ``default_parameters`` is a fresh dict the
    transform writes config-level overrides into (folded into the config's
    ``default_parameters`` by ``Configs``). Kept here — a dependency-light core module —
    so domain packs can import it without a cycle through ``scaffolding``."""

    ir: dict
    semantics: dict
    default_parameters: dict = field(default_factory=dict)


# type-kinds a name pattern can be gated to (mirror the column-type dispatch in
# datagenic.semantics._match_by_type).
PATTERN_KINDS = ("string", "float", "array_float", "date", "int")

# numeric fit-kinds — how the sample-fitter (datagenic.semantics._fit_numeric) should
# fit a numeric semantic from data.
NUMERIC_KINDS = ("scalar", "scalar_conditional", "array", "array_conditional")

# Provenance sentinel for core (domain-neutral) contributions: always visible under
# any scope. Distinct from a real domain name so it can never collide with a pack.
CORE = None


def _visible(domain, active) -> bool:
    """A contribution is in scope when it is core, when no scope is set (``active`` is
    ``None`` = all), or when its domain is one of the selected modules."""
    return domain is CORE or active is None or domain in active


class SemanticRegistry:
    """Mutable collection of everything the inferrers need, contributed by domain packs."""

    def __init__(self) -> None:
        # semantic -> (udf_module, udf_class, base_args, parent_arg_name | None)
        self.generators: dict[str, tuple[str, str, dict, str | None]] = {}
        # pattern-kind -> ordered [(regex_str, semantic, confidence)] (first match wins)
        self.patterns: dict[str, list[tuple[str, str, float]]] = {k: [] for k in PATTERN_KINDS}
        # parent_semantic -> [child_semantic, …] coherence links
        self.links: dict[str, list[str]] = {}
        # semantic -> numeric fit-kind (only for numeric/array-quantity semantics)
        self.numeric_kinds: dict[str, str] = {}
        # lazy factories: each returns [(semantic, predicate)] for value-based inference
        self._classifier_factories: list[tuple] = []
        # name -> (transform_fn, owning_domain) — post-inference scaffold transforms
        self._scaffold_transforms: dict[str, tuple] = {}

        # ── provenance (parallel to the public maps above; keeps their shapes intact) ──
        self._gen_domain: dict[str, object] = {}  # semantic -> domain
        self._pattern_domain: dict[str, list] = {k: [] for k in PATTERN_KINDS}  # aligned by index
        self._link_domain: dict[tuple[str, str], object] = {}  # (parent, child) -> domain
        self._domains: set[str] = set()  # non-core packs seen during load()
        self._current_domain: object = CORE  # stamped around each pack's register()

        self._compiled: dict[str, list] = {}
        self._loaded = False

    # ── registration API (called by domain register() hooks) ──────────────────
    def add_generator(
        self, semantic, module, cls, base_args=None, parent_arg=None, numeric_kind=None
    ):
        self.generators[semantic] = (module, cls, dict(base_args or {}), parent_arg)
        self._gen_domain[semantic] = self._current_domain
        if numeric_kind is not None:
            if numeric_kind not in NUMERIC_KINDS:
                raise ValueError(f"unknown numeric_kind {numeric_kind!r}; expected {NUMERIC_KINDS}")
            self.numeric_kinds[semantic] = numeric_kind

    def add_pattern(self, kind, regex, semantic, confidence):
        if kind not in self.patterns:
            raise ValueError(f"unknown pattern kind {kind!r}; expected one of {PATTERN_KINDS}")
        self.patterns[kind].append((regex, semantic, confidence))
        self._pattern_domain[kind].append(self._current_domain)
        self._compiled.pop(kind, None)  # invalidate the compiled cache for this kind

    def add_link(self, parent_semantic, children):
        bucket = self.links.setdefault(parent_semantic, [])
        for child in children:
            if child not in bucket:
                bucket.append(child)
            self._link_domain.setdefault((parent_semantic, child), self._current_domain)

    def add_value_classifiers(self, factory):
        self._classifier_factories.append((factory, self._current_domain))

    def add_scaffold_transform(self, name, fn):
        """Register a named post-inference transform owned by the current domain."""
        self._scaffold_transforms[name] = (fn, self._current_domain)

    # ── read API (used by datagenic.semantics) ────────────────────────────────
    def compiled(self, kind, active=None):
        """Compiled (regex, semantic, confidence) triples for a pattern kind, scoped to
        ``active`` (``None`` = all). The full compile is cached per kind; scoping filters
        the cache, so it stays cheap and order-preserving (first match wins)."""
        if kind not in self._compiled:
            self._compiled[kind] = [
                (re.compile(p, re.IGNORECASE), sem, conf, dom)
                for (p, sem, conf), dom in zip(
                    self.patterns[kind], self._pattern_domain[kind], strict=True
                )
            ]
        return [
            (rx, sem, conf) for rx, sem, conf, dom in self._compiled[kind] if _visible(dom, active)
        ]

    def links_for(self, active=None):
        """``parent_semantic -> [children]`` coherence links, scoped to ``active``."""
        out: dict[str, list[str]] = {}
        for parent, children in self.links.items():
            kept = [c for c in children if _visible(self._link_domain.get((parent, c)), active)]
            if kept:
                out[parent] = kept
        return out

    def value_classifiers(self, active=None):
        """Flatten the lazy classifier factories (specific-first, in registration order),
        scoped to ``active``."""
        out: list = []
        for factory, dom in self._classifier_factories:
            if _visible(dom, active):
                out.extend(factory())
        return out

    def generator_for(self, semantic, active=None):
        """The generator spec for ``semantic`` if it is in scope, else ``None``. Used by
        ``build_generator`` so a scoped scaffold never wires an out-of-scope generator."""
        if semantic in self.generators and _visible(self._gen_domain.get(semantic), active):
            return self.generators[semantic]
        return None

    def scaffold_transform(self, name, active=None):
        """Return the transform fn for ``name`` if registered by an in-scope module.

        Raises ``ValueError`` if the name is unknown, or is owned by a module not in
        ``active`` — the two cases share a message listing the transforms available in
        the current scope, since to the caller both mean "not available here"."""
        entry = self._scaffold_transforms.get(name)
        if entry is not None and _visible(entry[1], active):
            return entry[0]
        available = sorted(
            n for n, (_, dom) in self._scaffold_transforms.items() if _visible(dom, active)
        )
        raise ValueError(
            f"unknown scaffold transform {name!r}; available in the active modules: "
            f"{available or '(none)'}"
        )

    def transforms_for_domain(self, domain):
        """The transform names registered by ``domain`` (for per-domain validation)."""
        return {n for n, (_, dom) in self._scaffold_transforms.items() if dom == domain}

    def discovered_domains(self):
        """The set of non-core domain packs loaded by ``load()`` (for scope validation)."""
        return set(self._domains)

    # ── discovery ──────────────────────────────────────────────────────────────
    def load(self):
        """Populate once: core generics, then every discovered ``datagenic.domains.*`` pack.

        Idempotent. Registration imports each pack's pure data and records strings, so
        this stays numpy-free; domain generator classes are imported later, only when
        ``datagenic.semantics.build_generator`` instantiates one. Each pack's entries are
        stamped with its name (core generics carry ``CORE``) for scope filtering.
        """
        if self._loaded:
            return self
        self._loaded = True
        self._current_domain = CORE
        _register_core_generics(self)
        import datagenic.domains as _domains

        for mod in pkgutil.iter_modules(_domains.__path__):
            if mod.name.startswith("_"):
                continue
            pack = importlib.import_module(f"datagenic.domains.{mod.name}")
            hook = getattr(pack, "register", None)
            if callable(hook):
                self._current_domain = mod.name
                self._domains.add(mod.name)
                hook(self)
        self._current_domain = CORE
        return self


def _register_core_generics(registry):
    """Domain-neutral semantics available regardless of any domain pack: generic unique
    identifiers, keyed by column type (string vs int). Registered first, so a domain's
    patterns layer after them."""
    registry.add_generator(
        "identifier", "datagenic.generators", "UniqueIncrementalStrings", {"prefix": "ID_"}
    )
    registry.add_generator("identifier_int", "datagenic.generators", "UniqueIncrementalIntegers")
    _ID = r"product[_ ]?id|prod[_ ]?id"
    registry.add_pattern("string", _ID, "identifier", 0.8)
    registry.add_pattern("int", _ID, "identifier_int", 0.8)


# The shared registry that datagenic.semantics loads at import.
REGISTRY = SemanticRegistry()
