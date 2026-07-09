"""Fitting numeric distributions from real sample data (Phase 4).

The authoritative accuracy path: instead of curated book-agnostic priors, the
``empirical`` inferrer and ``profile``-from-sample fit each numeric column's
distribution from the sample — family (lognormal vs normal) + params, and, for a
conditional quantity, a per-parent-category ``cases`` map. Parameterised to match
``generators._draw_floats`` (mu = median in real units, sigma = log-space spread),
so generate → profile → regenerate round-trips.
"""

import numpy as np
import polars as pl

from datagenic import Generator
from datagenic.profiling import build_config
from datagenic.semantics import build_generator, empirical_inferrer, fit_distribution


def _lognormal(rng, median, sigma, n):
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=n)


# ── fit_distribution: family detection + parameter recovery ───────────────────


def test_fits_lognormal_for_positive_right_skewed():
    rng = np.random.default_rng(0)
    spec = fit_distribution(_lognormal(rng, 1e6, 1.0, 20000).tolist())
    assert spec["distribution"] == "lognormal"
    assert 0.8e6 < spec["mu"] < 1.2e6  # mu recovers the median (real units)
    assert 0.9 < spec["sigma"] < 1.1  # sigma recovers the log-space spread


def test_fits_normal_for_symmetric():
    rng = np.random.default_rng(0)
    spec = fit_distribution(rng.normal(100, 5, 20000).tolist())
    assert spec["distribution"] == "normal"
    assert 98 < spec["mu"] < 102
    assert 4.5 < spec["sigma"] < 5.5


def test_fits_normal_when_values_can_be_negative():
    rng = np.random.default_rng(0)
    spec = fit_distribution(rng.normal(0.0, 1.0, 5000).tolist())
    assert spec["distribution"] == "normal"


def test_fit_ignores_nulls_and_handles_empty():
    assert fit_distribution([None, None]) is None
    assert fit_distribution([]) is None
    spec = fit_distribution([None, 100.0, 100.0, None])
    assert spec is not None


def test_lognormal_fit_round_trips_through_draw_floats():
    from datagenic.generators import _draw_floats

    rng = np.random.default_rng(1)
    sample = _lognormal(rng, 5e5, 0.9, 20000)
    spec = fit_distribution(sample.tolist())
    regen = _draw_floats(np.random.default_rng(2), 20000, spec["mu"], spec["sigma"], "lognormal")
    # the regenerated sample recovers the same median scale
    assert 0.85 < float(np.median(regen)) / float(np.median(sample)) < 1.15


# ── conditional per-group fit via the empirical inferrer ─────────────────────────


def _conditional_samples(rng, n=8000):
    # interleaved (not block-sorted) so any head-capped sample still sees both classes
    ac = rng.permutation(np.array(["Equity"] * (n // 2) + ["Rates"] * (n // 2)))
    notional = np.where(
        ac == "Equity",
        _lognormal(rng, 1e5, 0.8, n),
        _lognormal(rng, 2.5e7, 1.0, n),
    )
    return {"asset_class": ac.tolist(), "notional": notional.tolist()}


def test_empirical_inferrer_fits_conditional_cases_from_sample():
    samples = _conditional_samples(np.random.default_rng(0))
    columns = [{"name": "asset_class", "type": "string"}, {"name": "notional", "type": "float"}]
    result = empirical_inferrer("trades", columns, samples=samples)

    args = result["notional"]["args"]
    assert result["notional"]["parent"] == "asset_class"
    # per-class medians recovered straight from the sample, ~2 orders apart
    assert 0.6e5 < args["cases"]["Equity"]["mu"] < 1.6e5
    assert 1.5e7 < args["cases"]["Rates"]["mu"] < 4.0e7
    assert args["cases"]["Equity"]["distribution"] == "lognormal"
    assert "default" in args


def test_thin_group_falls_back_to_default():
    rng = np.random.default_rng(0)
    # both classes present in the sample, but Rates has too few rows to fit
    ac = np.array(["Equity"] * 1000 + ["Rates"] * 5)
    notional = np.concatenate([_lognormal(rng, 1e5, 0.8, 1000), _lognormal(rng, 2.5e7, 1.0, 5)])
    order = rng.permutation(len(ac))
    samples = {"asset_class": ac[order].tolist(), "notional": notional[order].tolist()}
    columns = [{"name": "asset_class", "type": "string"}, {"name": "notional", "type": "float"}]
    args = empirical_inferrer("trades", columns, samples=samples)["notional"]["args"]
    assert "Equity" in args["cases"]
    assert "Rates" not in args["cases"]  # below the min-sample threshold


def test_build_generator_applies_fitted_overrides():
    overrides = {"cases": {"Equity": {"distribution": "lognormal", "mu": 1e5, "sigma": 0.8}}}
    cg = build_generator("notional", "asset_class", overrides)
    assert cg["udf_name"] == "ConditionalFloats"
    assert cg["args"]["cases"] == overrides["cases"]  # fitted cases replace the curated prior
    assert cg["args"]["by_column"] == "asset_class"


# ── profiling: family-aware flat fit ──────────────────────────────────────────


def test_profiling_flat_fit_picks_lognormal(tmp_path):
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({"metric": _lognormal(rng, 5e5, 1.0, 5000)})  # no known name -> flat
    config, _ir = build_config(
        {"t": frame}, output_folder_path=str(tmp_path / "out"), number_of_batches=1
    )
    metric = {c["col_name"]: c for c in config["tables"][0]["columns"]}["metric"]
    args = metric["column_generator"]["args"]
    assert args["distribution"] == "lognormal"
    assert 0.8e6 > args["mu"] > 3e5  # median ≈ 5e5


# ── end-to-end: profile-from-sample + values inference → fitted conditional ───


def test_end_to_end_profile_from_sample_fits_conditional(tmp_path):
    rng = np.random.default_rng(0)
    samples = _conditional_samples(rng, n=8000)
    frame = pl.DataFrame(
        {
            "trade_id": np.arange(8000),
            "asset_class": samples["asset_class"],
            "notional": samples["notional"],
        }
    )
    config, _ir = build_config(
        {"trades": frame},
        semantic_inferrer="empirical",
        output_format="parquet",
        output_folder_path=str(tmp_path / "out"),
        number_of_batches=1,
    )
    notional = {c["col_name"]: c for c in config["tables"][0]["columns"]}["notional"]
    cg = notional["column_generator"]
    # inference fitted the conditional generator; profiling did not clobber it
    assert cg["udf_name"] == "ConditionalFloats"
    assert cg["args"]["by_column"] == "asset_class"
    assert 0.6e5 < cg["args"]["cases"]["Equity"]["mu"] < 1.6e5

    Generator(data_generator_config_file=config).generate()
    files = list((tmp_path / "out" / "trades").rglob("*.parquet"))
    assert files
    df = pl.read_parquet(files[0])
    medians = dict(
        zip(
            *df.group_by("asset_class")
            .agg(pl.col("notional").median().alias("m"))
            .to_dict(as_series=False)
            .values(),
            strict=True,
        )
    )
    # generated data reproduces the sample's per-class scale separation
    assert medians["Rates"] / medians["Equity"] > 20
