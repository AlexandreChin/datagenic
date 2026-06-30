"""datagenic — generic synthetic-data generation.

Kept import-light on purpose: importing ``datagenic`` must NOT pull in any
platform/integration code, any cloud SDK, or even the generation engine
(polars/pandas), so the package stays a standalone, reusable leaf.

The public generation API is re-exported here lazily (PEP 562): ``from datagenic
import Generator`` (or ``Configs``/``Columns``/``GraphDependencies``) imports the
owning module on first access, so a bare ``import datagenic`` stays cheap.
Their homes:
  - ``Configs`` / ``Columns``  → ``datagenic.scaffolding``
  - ``Generator``             → ``datagenic.engine``
  - ``GraphDependencies``     → ``datagenic.dependency_graph``
  - ``build_config``          → ``datagenic.profiling`` (scaffold a config from sample files)
"""

_LAZY = {
    "Configs": "datagenic.scaffolding",
    "Columns": "datagenic.scaffolding",
    "Generator": "datagenic.engine",
    "GraphDependencies": "datagenic.dependency_graph",
    "build_config": "datagenic.profiling",
}

__all__ = list(_LAZY)


def __getattr__(name):
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'datagenic' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted(__all__)
