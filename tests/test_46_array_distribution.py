"""Distribution-aware array float generators (Phase 1 of array-quantity values).

`RandomArrayFloats` / `BrownianArrayFloats` gain `distribution` ("normal" default,
byte-identical, or "lognormal") and `path` ("iid" default, or "walk" — a per-row
cumulative series suited to a PnL/returns time series). The building blocks for
realistic array-typed quantities.
"""

import numpy as np
import pytest

from datagenic.generators import BrownianArrayFloats, RandomArrayFloats

# ── normal + iid is unchanged (golden-safe reshape identity) ──────────────────


def test_normal_iid_is_byte_identical_random():
    got = RandomArrayFloats().call(
        size_of_arrays=5,
        _number_of_rows_per_batch=4,
        mu=1000,
        sigma=100,
        _rng=np.random.default_rng(7),
    )
    expected = np.random.default_rng(7).normal(1000, 100, size=(4, 5)).tolist()
    assert got == expected


def test_normal_iid_is_byte_identical_brownian():
    got = BrownianArrayFloats().call(
        size_of_arrays=6,
        _number_of_rows_per_batch=3,
        mu=1000,
        sigma=100,
        _rng=np.random.default_rng(9),
    )
    expected = np.random.default_rng(9).normal(1000, 100, size=(3, 6)).tolist()
    assert got == expected


def test_explicit_normal_equals_default():
    kw = dict(size_of_arrays=4, _number_of_rows_per_batch=5, mu=50, sigma=5)
    a = RandomArrayFloats().call(**kw, _rng=np.random.default_rng(1))
    b = RandomArrayFloats().call(
        **kw, distribution="normal", path="iid", _rng=np.random.default_rng(1)
    )
    assert a == b


# ── lognormal ─────────────────────────────────────────────────────────────────


def test_lognormal_elements_positive_and_centered():
    out = np.array(
        RandomArrayFloats().call(
            size_of_arrays=250,
            _number_of_rows_per_batch=200,
            mu=1e5,
            sigma=0.8,
            distribution="lognormal",
            _rng=np.random.default_rng(0),
        )
    )
    assert out.shape == (200, 250)
    assert (out > 0).all()
    assert 0.85e5 < float(np.median(out)) < 1.15e5  # median ≈ mu (real units)


def test_lognormal_is_deterministic():
    kw = dict(
        size_of_arrays=10, _number_of_rows_per_batch=20, mu=5e5, sigma=1.0, distribution="lognormal"
    )
    a = RandomArrayFloats().call(**kw, _rng=np.random.default_rng(3))
    b = RandomArrayFloats().call(**kw, _rng=np.random.default_rng(3))
    assert a == b


# ── path: iid vs walk ─────────────────────────────────────────────────────────


def test_walk_is_cumulative_over_iid_increments():
    kw = dict(size_of_arrays=6, _number_of_rows_per_batch=3, mu=0, sigma=1)
    iid = np.array(RandomArrayFloats().call(**kw, _rng=np.random.default_rng(3)))
    walk = np.array(RandomArrayFloats().call(**kw, path="walk", _rng=np.random.default_rng(3)))
    # same seed -> same increments; walk is their running sum along the array axis
    assert np.allclose(walk, np.cumsum(iid, axis=1))
    # a walk's first element equals the first increment; later ones diverge
    assert np.allclose(walk[:, 0], iid[:, 0])
    assert not np.allclose(walk[:, -1], iid[:, -1])


# ── error handling ────────────────────────────────────────────────────────────


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        RandomArrayFloats().call(
            size_of_arrays=3,
            _number_of_rows_per_batch=2,
            distribution="weird",
            _rng=np.random.default_rng(0),
        )


def test_unknown_path_raises():
    with pytest.raises(ValueError, match="unknown path"):
        RandomArrayFloats().call(
            size_of_arrays=3,
            _number_of_rows_per_batch=2,
            path="zigzag",
            _rng=np.random.default_rng(0),
        )


# ── BrownianArrayFloats: recall still evolves across dates (normal shocks) ────


def test_brownian_recall_evolves_same_shape():
    g = BrownianArrayFloats()
    base = np.array(
        g.call(
            size_of_arrays=8,
            _number_of_rows_per_batch=10,
            mu=100,
            sigma=10,
            _rng=np.random.default_rng(0),
        )
    )
    nxt = np.array(
        g.recall(
            input_array=base,
            size_of_arrays=8,
            mu=100,
            sigma=10,
            _number_of_rows_per_batch=10,
            _rng=np.random.default_rng(1),
        )
    )
    assert nxt.shape == base.shape
    assert not np.array_equal(nxt, base)  # values moved
