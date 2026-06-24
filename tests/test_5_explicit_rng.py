"""Phase 3 — Explicit RNG instances (no global numpy state).

Generators and postprocessors draw from a framework-injected
``np.random.default_rng`` per table/date/batch, not numpy's global state. The key
new guarantee is *isolation*: a stray ``np.random`` draw anywhere can no longer
shift another unit's output. (Determinism and seed-sensitivity are covered by
test_2_safety_net.py.)
"""

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden_fixtures as gf  # noqa: E402

from datagenic import Generator, generators


def test_generator_output_depends_only_on_injected_rng():
    """A generator's output is unchanged by intervening global np.random calls."""
    a = generators.BrownianFloats().call(
        _number_of_rows_per_batch=100, _rng=np.random.default_rng(42)
    )
    np.random.seed(7)
    np.random.uniform(size=5000)  # perturb the global numpy state
    b = generators.BrownianFloats().call(
        _number_of_rows_per_batch=100, _rng=np.random.default_rng(42)
    )
    assert np.array_equal(a, b)


def _hash_tree(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.parquet"))
    }


def test_generation_isolated_from_global_state(tmp_path):
    """Full generation output is immune to global np.random perturbation between runs."""
    cfg = gf._rng_config()
    a, b = tmp_path / "a", tmp_path / "b"

    cfg_a = copy.deepcopy(cfg)
    cfg_a["default_parameters"]["output_folder_path"] = str(a)
    Generator(data_generator_config_file=cfg_a).generate()

    # heavily perturb global numpy state between the two runs
    np.random.seed(999)
    np.random.normal(size=100000)

    cfg_b = copy.deepcopy(cfg)
    cfg_b["default_parameters"]["output_folder_path"] = str(b)
    Generator(data_generator_config_file=cfg_b).generate()

    assert _hash_tree(a) == _hash_tree(b)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
