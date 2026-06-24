"""Generator/postprocessor UDF contracts — runtime-checkable Protocols checked in
``Generator.validate()``. Split out of the former ``data_generator`` module
(package split Part 6).
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StatelessColumnGenerator(Protocol):
    """A column generator implementing only call() (same output every date)."""

    def call(self, _number_of_rows_per_batch: int, **kwargs): ...


@runtime_checkable
class ColumnGenerator(StatelessColumnGenerator, Protocol):
    """A column generator implementing call() and recall() (time-propagated)."""

    def recall(self, input_array, **kwargs): ...


@runtime_checkable
class StatelessPostProcessor(Protocol):
    """A postprocessor implementing only call() (run on every date).

    ``INPUT_SCOPE`` (optional class attribute, default ``"batch"``) declares what a
    postprocessor depends on, so the distribution-safety gate can refuse to shard a
    config its author has marked unshardable:
      - ``"batch"``     — depends only on its own batch's data (safe to shard).
      - ``"reference"`` — also reads small reference/dimension tables every worker
                          regenerates identically (safe to shard).
      - ``"global"``    — depends on the whole table / cross-batch state (NOT
                          shardable; rejected by ``validate(distributed=True)``).
    Reading another table via ``udf_args._input_tables`` is always batch-local (the
    framework injects only the current batch's slice), so it does not by itself make
    a postprocessor unshardable — declare ``"global"`` only for true whole-table or
    cross-batch dependencies.
    """

    def call(self, _table, _input_tables: dict, **kwargs): ...


@runtime_checkable
class PostProcessor(StatelessPostProcessor, Protocol):
    """A postprocessor implementing call() (date 1) and recall() (later dates)."""

    def recall(self, _table, _input_tables: dict, **kwargs): ...
