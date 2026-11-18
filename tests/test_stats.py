"""Inference helpers, checked on series with known dependence structure."""

import numpy as np
import pytest
from scipy import stats

from implied_stock_distributions.stats import (
    block_bootstrap_mean,
    diebold_mariano,
    newey_west_variance,
    pit_calibration,
    skill_score,
)


def test_newey_west_matches_the_sample_variance_at_zero_lag():
    generator = np.random.default_rng(0)
    values = generator.normal(size=5000)
    assert newey_west_variance(values, lag=0) == pytest.approx(
        values.var(), rel=1e-12
    )


def test_newey_west_inflates_the_variance_of_a_correlated_series():
    """Overlapping forecasts look more informative than they are."""
    generator = np.random.default_rng(1)
    innovations = generator.normal(size=6000)
    # A moving average of 21 draws mimics 21 overlapping daily forecasts.
    overlapping = np.convolve(innovations, np.ones(21) / 21, mode="valid")

    naive = overlapping.var()
    corrected = newey_west_variance(overlapping, lag=21)

    assert corrected > 3 * naive


def test_diebold_mariano_finds_no_difference_between_equal_forecasts():
    generator = np.random.default_rng(2)
    common = generator.gamma(2.0, size=1500)

    result = diebold_mariano(
        common + generator.normal(0, 0.01, 1500),
        common + generator.normal(0, 0.01, 1500),
        lag=21,
    )
    assert result["p_value"] > 0.05


def test_diebold_mariano_finds_a_real_difference():
    generator = np.random.default_rng(3)
    common = generator.gamma(2.0, size=1500)

    result = diebold_mariano(common + 0.5, common, lag=21, names=("worse", "better"))
    assert result["p_value"] < 1e-6
    assert result["mean_difference"] == pytest.approx(0.5, rel=1e-6)
    assert result["better"] == "better"


def test_diebold_mariano_reports_the_variance_inflation():
    generator = np.random.default_rng(4)
    innovations = generator.normal(size=3000)
    overlapping = np.convolve(innovations, np.ones(21) / 21, mode="valid")

    result = diebold_mariano(overlapping, np.zeros_like(overlapping), lag=21)
    assert result["variance_inflation"] > 2.0


def test_block_bootstrap_covers_the_true_mean():
    generator = np.random.default_rng(6)
    blocks = np.repeat(np.arange(120), 21)
    block_effect = generator.normal(0, 1.0, 120)
    values = block_effect[blocks] + generator.normal(0, 0.2, blocks.size)

    result = block_bootstrap_mean(values, blocks, replicates=500)
    assert result["lower"] < values.mean() < result["upper"]
    assert result["n_blocks"] == 120


def test_block_bootstrap_is_wider_than_an_iid_interval():
    generator = np.random.default_rng(7)
    blocks = np.repeat(np.arange(80), 21)
    values = generator.normal(0, 1.0, 80)[blocks]

    result = block_bootstrap_mean(values, blocks, replicates=500)
    iid_error = values.std(ddof=1) / np.sqrt(values.size)
    assert result["standard_error"] > 2 * iid_error


def test_pit_calibration_passes_uniform_values():
    # Evenly spaced PITs: exactly uniform, so every number sits on its null.
    n = 1000
    result = pit_calibration((np.arange(n) + 0.5) / n)

    assert result["mean"] == pytest.approx(0.5, abs=1e-12)
    assert result["variance_ratio"] == pytest.approx(1.0, abs=1e-5)
    assert result["share_below_10pct"] == pytest.approx(0.1, abs=1e-12)
    for name in ("mean", "variance_ratio", "share_below_10pct"):
        assert result[f"{name}_p_value"] > 0.05, name
    assert (
        result["share_below_10pct_lower"] < 0.1 < result["share_below_10pct_upper"]
    )


def test_pit_calibration_reads_a_risk_neutral_forecast():
    """The shape a risk-neutral density should show against outcomes.

    Outcomes land high (equity premium), cluster toward the middle relative
    to the forecast's width (variance premium) and rarely in the bottom
    decile (crash premium). Each number must point the right way.
    """
    generator = np.random.default_rng(9)
    values = generator.beta(2.4, 1.8, size=3000)

    result = pit_calibration(values)
    assert result["mean"] > 0.5 and result["mean_p_value"] < 1e-6
    assert result["variance_ratio"] < 1 and result["variance_ratio_p_value"] < 1e-6
    assert result["share_below_10pct_upper"] < 0.1
    assert result["share_below_10pct_p_value"] < 1e-6


def test_pit_calibration_separates_location_from_width():
    """A forecast that is only too narrow moves the variance, not the mean."""
    generator = np.random.default_rng(10)
    outcomes = generator.normal(size=4000)
    values = stats.norm.cdf(outcomes / 0.8)      # forecast sd 0.8, truth 1

    result = pit_calibration(values)
    assert result["mean_p_value"] > 0.01
    assert result["variance_ratio"] > 1 and result["variance_ratio_p_value"] < 1e-6


def test_skill_score_signs():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)


@pytest.mark.parametrize("in_tail", [0, 3, 17, 54])
def test_lowest_decile_interval_is_clopper_pearson(in_tail):
    n = 54
    generator = np.random.default_rng(in_tail)
    values = np.concatenate([
        generator.uniform(0.0, 0.1, in_tail),
        generator.uniform(0.1, 1.0, n - in_tail),
    ])

    result = pit_calibration(values)
    exact = stats.binomtest(in_tail, n, 0.1).proportion_ci(0.95, method="exact")

    assert result["share_below_10pct"] == pytest.approx(in_tail / n)
    assert result["share_below_10pct_lower"] == pytest.approx(exact.low, abs=1e-12)
    assert result["share_below_10pct_upper"] == pytest.approx(exact.high, abs=1e-12)


def test_identical_pits_give_no_spread_test():
    result = pit_calibration(np.full(20, 0.5))
    assert np.isnan(result["mean_p_value"])
    assert np.isnan(result["variance_ratio_p_value"])
