# Testing Reference

## Running the suite

```bash
uv sync                              # install deps incl. dev + cloud groups
pytest tests/                        # full suite (numbered tests/test_1..test_32)
pytest tests/test_2_safety_net.py::test_same_seed_is_byte_identical   # one test
pytest -m "not integration"          # skip tests needing cloud creds / network
ruff check .                         # lint
ruff format --check .                # format check (line-length 100, double quotes)
mypy datagenic                       # type-check (non-strict; targets py3.12)
```

`tests/test_*` are numbered by feature/step. The `integration` marker covers tests that need
network, credentials, or cloud storage — CI runs `-m "not integration"`.

## The golden baseline (determinism guard)

`tests/test_2_safety_net.py::test_golden_manifest_matches` asserts regenerated output against
`tests/golden/baseline_manifest.json`. The fixtures feeding both the capture script and the test
live in `tests/golden/golden_fixtures.py` (single source of truth). If you **intentionally**
change generation output, regenerate:

```bash
python tests/golden/capture_baseline.py
```

`hash_tree` hashes each parquet's **logical content** (schema + values), not raw bytes, so the
baseline survives pyarrow/Python upgrades — a failure you did *not* intend is a real determinism
regression; investigate, don't just refresh. (Note: the data is byte-stable across library
versions, but the locked toolchain pins this; the lockfile resolves a single numpy across all
supported Pythons.)

## What the suite already covers (don't re-prove by hand)

- **Determinism** — same seed/config → identical output; different seed → different output;
  repeated `generate()` and reusing one dict across `Generator` instances are identical;
  `generate()` does not mutate the caller's config.
- **Routing** — `call` on the first date, `recall` after; `single_date` runs once; `single_batch`
  produced once and broadcast; `dump_results: false` writes no parquet.
- **Data handoff** — `_table` carries generator output; `_input_tables` carries exactly the
  declared tables; referencing an ungenerated table raises (`_Ungenerated`).
- **Distribution** — `--parallel` and the union of shards are byte-identical to a single run;
  `check_distributable` flags non-shardable configs.
- **Checkpointing** — re-run resumes from content-hashed parquets; `extend` adds dates.
- **Validation / protocols / error wrapping / output schema / structured logging / config
  inheritance (`extends`) / dry-run.**

## Writing tests

- Use `tmp_path` for output; compare logical content, not raw parquet bytes, when asserting
  across environments.
- Guard temporal assertions: skip the first date when there is no previous data.
- Read sizes/params from the config's source of truth (the column's `args` or
  `default_parameters`), not a hard-coded constant.
- For randomness inside a post-processor/generator under test, assert it uses the injected
  `_rng` (the determinism + `check_distributable` tests will catch global-RNG usage).

The dev/sample UDFs the tests reference live in `postprocessors_udfs.generic_udfs` (resolved via
`pythonpath = .` in `pytest.ini`).
