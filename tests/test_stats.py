"""Inference helpers, checked on series with known dependence structure."""

import numpy as np
import pytest

from implied_stock_distributions.stats import (
    block_bootstrap_mean,
    diebold_mariano,
    newey_west_variance,
    pit_uniformity,
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


def test_pit_uniformity_passes_uniform_values():
    generator = np.random.default_rng(8)
    result = pit_uniformity(generator.uniform(size=3000))
    assert result["ks_p_value"] > 0.05
    assert result["mean"] == pytest.approx(0.5, abs=0.03)


def test_pit_uniformity_detects_too_much_left_tail_mass():
    """The shape a risk-neutral density should show against outcomes."""
    generator = np.random.default_rng(9)
    values = generator.beta(2.0, 1.2, size=3000)

    result = pit_uniformity(values)
    assert result["ks_p_value"] < 1e-6
    assert result["share_below_10pct"] < 0.1


def test_skill_score_signs():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 1.0) == pytest.approx(0.0)
    assert skill_score(2.0, 1.0) == pytest.approx(-1.0)
