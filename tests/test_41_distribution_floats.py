"""Distribution-aware float generators (Phase 1 of conditional numeric values).

`RandomFloats` / `BrownianFloats` gain a `distribution` arg: "normal" (default, the
historical N(mu, sigma) — byte-identical) or "lognormal" (positive, right-skewed; mu is
the median in real units, sigma the log-space spread). The building block for realistic
magnitudes like notionals/prices.
"""

import numpy as np
import pytest

from datagenic.generators import BrownianFloats, RandomFloats

# ── normal path is unchanged (golden-safe) ────────────────────────────────────


def test_normal_default_is_byte_identical():
    got = RandomFloats().call(
        _number_of_rows_per_batch=500, mu=1000, sigma=100, _rng=np.random.default_rng(7)
    )
    expected = np.random.default_rng(7).normal(loc=1000, scale=100, size=500)
    assert np.array_equal(got, expected)


def test_explicit_normal_equals_default():
    a = RandomFloats().call(
        _number_of_rows_per_batch=300, mu=50, sigma=5, _rng=np.random.default_rng(1)
    )
    b = RandomFloats().call(
        _number_of_rows_per_batch=300,
        mu=50,
        sigma=5,
        distribution="normal",
        _rng=np.random.default_rng(1),
    )
    assert np.array_equal(a, b)


# ── lognormal ─────────────────────────────────────────────────────────────────


def test_lognormal_is_positive_and_centered_on_median():
    out = RandomFloats().call(
        _number_of_rows_per_batch=20000,
        mu=1_000_000,
        sigma=0.8,
        distribution="lognormal",
        _rng=np.random.default_rng(0),
    )
    assert (out > 0).all()  # positive
    assert 0.9e6 < float(np.median(out)) < 1.1e6  # median ≈ mu (real units)
    assert float(out.mean()) > float(np.median(out))  # right-skewed


def test_lognormal_is_deterministic():
    kw = dict(_number_of_rows_per_batch=1000, mu=5e6, sigma=1.0, distribution="lognormal")
    a = RandomFloats().call(**kw, _rng=np.random.default_rng(3))
    b = RandomFloats().call(**kw, _rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_lognormal_requires_positive_mu():
    with pytest.raises(ValueError, match=r"median.*> 0"):
        RandomFloats().call(
            _number_of_rows_per_batch=10,
            mu=0,
            sigma=1.0,
            distribution="lognormal",
            _rng=np.random.default_rng(0),
        )


def test_unknown_distribution_raises():
    with pytest.raises(ValueError, match="unknown distribution"):
        RandomFloats().call(
            _number_of_rows_per_batch=10, distribution="weird", _rng=np.random.default_rng(0)
        )


# ── BrownianFloats: distribution on call, normal shocks on recall ─────────────


def test_brownian_lognormal_call_then_recall():
    g = BrownianFloats()
    base = g.call(
        _number_of_rows_per_batch=2000,
        mu=1e6,
        sigma=0.7,
        distribution="lognormal",
        _rng=np.random.default_rng(0),
    )
    assert (base > 0).all()
    assert 0.85e6 < float(np.median(base)) < 1.15e6

    # recall propagates the series with (normal) shocks: same length, values move
    nxt = g.recall(input_array=base, drift=0, noise=100, _rng=np.random.default_rng(1))
    assert len(nxt) == len(base)
    assert not np.array_equal(np.asarray(nxt), base)


def test_brownian_normal_default_byte_identical():
    got = BrownianFloats().call(
        _number_of_rows_per_batch=400, mu=1000, sigma=100, _rng=np.random.default_rng(9)
    )
    expected = np.random.default_rng(9).normal(loc=1000, scale=100, size=400)
    assert np.array_equal(got, expected)
