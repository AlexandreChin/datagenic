"""Domain packs — self-contained bundles of domain-specific column generators,
vocabularies, and semantic priors that the domain-agnostic scaffolder wires in via
semantic inference. ``market_finance`` is the first; add further domains here.

Kept import-light on purpose (like the top-level package): importing this namespace
must not pull in numpy/polars or any domain module. Packs are imported lazily by
``datagenic.semantics.build_generator`` through ``importlib.import_module``.
"""
