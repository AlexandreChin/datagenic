"""Generation-side distribution: shard / parallelise a run, and check shardability.

Pure-generation work-split — these only ever *generate* (no ingestion), so they
belong in `datagenic`:
- ``shard_batch_range`` / ``run_shard`` — disjoint batch slices for master-free
  distribution (Scalability Step 6 / Part 4).
- ``run_parallel`` — single-machine multicore generation (Step 8 / Part 5).
- ``check_distributable`` — the distribution-safety gate (Step 7 / Part 10).

The generate→consume→reclaim orchestrator that drains to an ingestion consumer
(`ChunkedRun`) is platform/integration code and lives outside this package.
"""

import copy
import hashlib
import math
import tempfile
from pathlib import Path

import numpy

from datagenic import Generator


def _hash_tree(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.parquet"))
    }


def check_distributable(config):
    """Verdict on whether a config can be sharded safely (Step 7 / Part 10 gate).

    Layers, run on the config as given (use a small/representative config — the
    dependency *structure* is independent of row count, so a sample proves the
    full-scale case): (a)+(d) the static gate via ``validate(distributed=True)``;
    (b) a determinism probe — generate twice, perturbing the *global* numpy RNG in
    between, and require byte-identical output (a postprocessor using
    ``np.random``/``random``/``uuid``/``time`` instead of the injected ``_rng`` would
    diverge); (c) a sample independence proof — the union of two shards must equal a
    single run byte-for-byte. Returns ``{"ok": bool, "failures": [str, ...]}``. All
    runs use a temp directory; the caller's output location is never touched.
    """
    resolved = copy.deepcopy(Generator(config)._source_config)
    failures = []

    try:
        Generator(data_generator_config_file=copy.deepcopy(resolved)).validate(distributed=True)
    except ValueError as exc:
        failures.append(f"static gate: {exc}")

    def _generate_into(subdir, *, global_seed=None):
        cfg = copy.deepcopy(resolved)
        cfg["default_parameters"]["output_folder_path"] = str(subdir)
        # Force a known *global* RNG state so the probe never depends on ambient
        # state left by an earlier run.
        if global_seed is not None:
            numpy.random.seed(global_seed)
        Generator(data_generator_config_file=cfg).generate()
        return _hash_tree(subdir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # (b) determinism probe: two runs under DIFFERENT global RNG states. A
        # config that uses only the injected _rng is immune (identical output); one
        # whose postprocessor touches np.random/random/uuid/time diverges.
        first = _generate_into(tmp / "det0", global_seed=111_111_111)
        second = _generate_into(tmp / "det1", global_seed=222_222_222)
        if first != second:
            failures.append(
                "determinism probe: output changed with the global numpy RNG in a different "
                "state — a postprocessor likely uses np.random/random/uuid/time instead of the "
                "injected _rng, so workers would diverge."
            )

        # (c) sample independence: union of shards == single run
        single = _generate_into(tmp / "single")
        sharded_dir = tmp / "sharded"
        for shard in (1, 2):
            cfg = copy.deepcopy(resolved)
            cfg["default_parameters"]["output_folder_path"] = str(sharded_dir)
            run_shard(cfg, shard=shard, of=2)
        if _hash_tree(sharded_dir) != single:
            failures.append(
                "sample independence: union of 2 shards is not byte-identical to a single run "
                "— a postprocessor breaks batch independence."
            )

    return {"ok": not failures, "failures": failures}


def shard_batch_range(number_of_batches, shard, of):
    """Disjoint batch slice ``[lo, hi)`` for worker ``shard`` of ``of`` (Step 6).

    Shards are **1-based**: worker ``i`` (``1 … N``) owns batches
    ``[⌈(i-1)·B/N⌉, ⌈i·B/N⌉)``; the framework computes the range so the developer
    never does batch math. The slices tile ``[0, B)`` exactly — no gaps, no overlaps
    — and ``shard_batch_range`` returns a ``[lo, hi]`` pair to drop into
    ``default_parameters["batch_range"]`` (which the generator reads as 1-based
    batches ``lo+1 … hi``).
    """
    if of < 1:
        raise ValueError(f"--of must be >= 1, got {of}")
    if not (1 <= shard <= of):
        raise ValueError(f"--shard must be in [1, {of}], got {shard}")
    lo = math.ceil((shard - 1) * number_of_batches / of)
    hi = math.ceil(shard * number_of_batches / of)
    return [lo, hi]


def run_shard(config, shard=1, of=1, tables=None):
    """Generate one shard's batch slice of a run (master-free distribution).

    Shards are 1-based (``shard`` in ``1 … of``). ``of=1`` (the default) is today's
    behaviour untouched — the config runs as-is with no ``batch_range`` injected, so
    a non-distributed run is byte-identical. For ``of>1`` the shard's ``batch_range``
    is applied to a *copy* of the resolved config (the canonical config/file is never
    mutated); seeds are unchanged, so the union of all shards equals a single-machine
    run byte-for-byte. Workers write disjoint ``table/date/batch`` files into the one
    shared output location.

    ``tables`` runs only a *job* — the given root table(s) plus their transitive
    upstream closure (orthogonal to sharding, which slices batches).
    """
    if of == 1 and shard == 1:
        Generator(data_generator_config_file=config).generate(tables=tables)
        return

    resolved = copy.deepcopy(Generator(config)._source_config)
    params = resolved["default_parameters"]
    number_of_batches = params["number_of_batches"]
    params["batch_range"] = shard_batch_range(number_of_batches, shard, of)
    # Each shard gets its own diagnostic log/summary so concurrent workers writing
    # into the one shared output dir don't race on the same files (the summary's
    # atomic tmp+replace would otherwise collide). These are not data files.
    prefix = params.get("tables_prefix", "")
    suffix = f"shard{shard}-of{of}"
    params["generation_log_path"] = (
        f"{params['output_folder_path']}/{prefix}generation.{suffix}.log"
    )
    params["generation_summary_path"] = (
        f"{params['output_folder_path']}/{prefix}generation_summary.{suffix}.json"
    )
    Generator(data_generator_config_file=resolved).generate(tables=tables)


def _generate_shard_worker(task):
    """Pool worker: generate one shard. Module-level so it is picklable."""
    config, shard, of, tables = task
    run_shard(config, shard=shard, of=of, tables=tables)


def run_parallel(config, workers, tables=None):
    """Generate a run across ``workers`` local processes (Step 8 / Part 5).

    The on-one-machine counterpart to ``run_shard``: the batch space is split into
    ``workers`` disjoint slices (the same partition as ``--shard/--of``) and each is
    generated in its own process via a ``ProcessPoolExecutor``. No coordination is
    needed — per-unit seeded RNG makes every slice independent and partitioned
    output means workers write disjoint files — so parallel output is byte-identical
    to a sequential run. ``workers=1`` runs sequentially (today's behaviour).

    This parallelises *generation* only. A downstream consumer holding state across
    chunks (e.g. one spanning ingestion transaction) is a separate switch — see
    ``ChunkedRun``.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers == 1:
        run_shard(config, shard=1, of=1, tables=tables)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Never use the "fork" start method: by the time we get here numpy/polars/pyarrow
    # have spawned worker threads, and forking a multithreaded process leaves those
    # threads' locks held forever in the child -> deadlock. (Default on Python <=3.13;
    # 3.14 switched to "forkserver" for exactly this reason.) Force a clean start
    # method ourselves so behaviour is identical on every supported interpreter.
    methods = mp.get_all_start_methods()
    ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")

    tasks = [(config, shard, workers, tables) for shard in range(1, workers + 1)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        # list() forces evaluation so a worker exception propagates to the caller.
        list(pool.map(_generate_shard_worker, tasks))
