"""Tests for the section 2 and section 3 fixes.

Each test names the issue it pins down, so that a regression says which
argument in the notebook has stopped being true rather than only which
function broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from implied_stock_distributions import benchmarks, evaluation, stats as isd_stats
from implied_stock_distributions import validation
from implied_stock_distributions.fitting import (
    build_call_spline_program,
    fit_shape_constrained_call_spline,
    select_call_equivalent_quotes,
)
from implied_stock_distributions.forwards import estimate_forward_and_rate
from implied_stock_distributions.smoothing import create_density_from_chain
from implied_stock_distributions.synthetic import (
    MixtureModel,
    SyntheticMarket,
    build_chain,
)

FORWARD = 4000.0
VOLATILITY = 0.16
TAU = 30 / 365


# ==========================================================================
# 2.1 -- the CRPS quadrature
# ==========================================================================

def _lognormal_tabulation(grid):
    scale = VOLATILITY * np.sqrt(TAU)
    mean_log = np.log(FORWARD) - 0.5 * scale**2
    cdf = stats.lognorm.cdf(grid, s=scale, scale=np.exp(mean_log))
    return (cdf - cdf[0]) / (cdf[-1] - cdf[0]), mean_log, scale


def test_crps_matches_the_closed_form():
    """The tabulated score reproduces the exact lognormal CRPS."""
    grid = np.linspace(1e-4 * FORWARD, 2.5 * FORWARD, 4000)
    cdf, mean_log, scale = _lognormal_tabulation(grid)

    for outcome in (3000.0, 3800.0, FORWARD, 4300.0, 5000.0):
        exact = float(benchmarks.lognormal_crps(mean_log, scale, outcome))
        got = evaluation.crps_from_cdf(grid, cdf, outcome)
        assert abs(got - exact) < 0.02, (outcome, got, exact)


def test_crps_matches_the_gaussian_closed_form():
    """The same, for the Gaussian form used elsewhere in the project."""
    mean, sd = 100.0, 12.0
    grid = np.linspace(mean - 10 * sd, mean + 10 * sd, 6000)
    cdf = stats.norm.cdf(grid, loc=mean, scale=sd)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    for outcome in (70.0, 95.0, 100.0, 118.0, 140.0):
        exact = float(benchmarks.gaussian_crps(mean, sd, outcome))
        got = evaluation.crps_from_cdf(grid, cdf, outcome)
        assert abs(got - exact) < 0.01, (outcome, got, exact)


def test_crps_is_invariant_to_the_grid():
    """The headline issue: the score must not reward a denser tabulation.

    Under the trapezoid rule the same distribution scored on a plain
    linspace and on a linspace merged with traded strikes differed by
    enough per observation to swamp the effect the project is measuring.
    """
    plain = np.linspace(1e-4 * FORWARD, 2.5 * FORWARD, 1000)
    merged = np.unique(np.concatenate([plain, np.arange(2000.0, 5200.0, 5.0)]))

    outcomes = np.linspace(3000.0, 5000.0, 41)

    plain_cdf, _, _ = _lognormal_tabulation(plain)
    merged_cdf, _, _ = _lognormal_tabulation(merged)

    scores = np.array([
        evaluation.crps_from_cdf(merged, merged_cdf, y) for y in outcomes
    ])
    differences = np.array([
        evaluation.crps_from_cdf(plain, plain_cdf, y) for y in outcomes
    ]) - scores

    # What is left is the difference between the two piecewise-linear
    # *interpolants*, not quadrature error, and it is two parts in ten
    # thousand of the score. Under the trapezoid rule the same comparison
    # carried a per-observation spread of well over one index point.
    assert np.max(np.abs(differences / scores)) < 1e-3


def test_crps_handles_outcomes_outside_the_support():
    """Below and above the grid the integrand is one; check both ends."""
    grid = np.linspace(3000.0, 5000.0, 2001)
    cdf = np.linspace(0.0, 1.0, 2001)

    inside = evaluation.crps_from_cdf(grid, cdf, 3000.0)
    below = evaluation.crps_from_cdf(grid, cdf, 2900.0)
    above = evaluation.crps_from_cdf(grid, cdf, 5100.0)

    assert below == pytest.approx(inside + 100.0, abs=1e-6)
    assert above == pytest.approx(
        evaluation.crps_from_cdf(grid, cdf, 5000.0) + 100.0, abs=1e-6
    )


def test_crps_of_a_point_mass_is_the_absolute_error():
    """A degenerate forecast scores the absolute error, exactly."""
    grid = np.array([0.0, 99.999999, 100.0, 200.0])
    cdf = np.array([0.0, 0.0, 1.0, 1.0])
    assert evaluation.crps_from_cdf(grid, cdf, 130.0) == pytest.approx(30.0, abs=1e-4)


def test_crps_and_pit_describe_one_distribution():
    """Both read the same piecewise-linear interpolant.

    A CDF that is exactly linear on its support has an analytic CRPS, and
    the PIT of the outcome must be the value that formula uses.
    """
    grid = np.linspace(0.0, 1.0, 501)
    cdf = grid.copy()
    outcome = 0.37

    # For U(0,1): CRPS = y^2 - y + 1/3 at an outcome inside the support.
    assert evaluation.crps_from_cdf(grid, cdf, outcome) == pytest.approx(
        outcome**2 - outcome + 1 / 3, abs=1e-6
    )
    assert evaluation.probability_integral_transform(
        grid, cdf, outcome
    ) == pytest.approx(outcome, abs=1e-9)


# ==========================================================================
# 2.2 -- location versus shape
# ==========================================================================

def test_recentring_puts_the_mean_on_the_forward():
    grid = np.linspace(1.0, 8000.0, 4000)
    scale = 0.2
    mean_log = np.log(4200.0) - 0.5 * scale**2
    density = stats.lognorm.pdf(grid, s=scale, scale=np.exp(mean_log))

    before = evaluation.distribution_moments(grid, density)["mean"]
    assert before == pytest.approx(4200.0, rel=1e-3)

    for method in ("scale", "shift"):
        new_grid, new_density = evaluation.recentre_on_forward(
            grid, density, FORWARD, method=method
        )
        after = evaluation.distribution_moments(new_grid, new_density)["mean"]
        assert after == pytest.approx(FORWARD, rel=1e-6), method


def test_recentring_by_scaling_preserves_shape_in_log_space():
    """A multiplicative recentre moves location only, in log terms."""
    grid = np.linspace(1.0, 8000.0, 4000)
    scale = 0.2
    density = stats.lognorm.pdf(grid, s=scale, scale=4000.0)

    before = evaluation.distribution_moments(grid, density)
    new_grid, new_density = evaluation.recentre_on_forward(grid, density, FORWARD)
    after = evaluation.distribution_moments(new_grid, new_density)

    assert after["log_standard_deviation"] == pytest.approx(
        before["log_standard_deviation"], rel=1e-6
    )
    assert after["log_skewness"] == pytest.approx(before["log_skewness"], rel=1e-4)


def test_iv_baseline_reports_its_own_displacement():
    """``mean_over_forward`` must be present, so 2.2 can be measured."""
    generator = np.random.default_rng(3)
    chain, _ = build_chain(
        pd.Timestamp("2018-03-15"), pd.Timestamp("2018-04-20"), 30,
        2700.0, 0.15, SyntheticMarket(), generator,
    )
    estimates, _ = estimate_forward_and_rate(chain)
    result = create_density_from_chain(
        chain, forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
    )

    assert "mean_over_forward" in result
    assert np.isfinite(result["mean_over_forward"])
    assert result["n_smile_knots"] >= 4


# ==========================================================================
# 2.3 -- the pooled HAC lag is in rows, not days
# ==========================================================================

def test_pooled_lag_counts_rows_not_days():
    frame = pd.DataFrame({
        "data_date": np.repeat(pd.date_range("2020-01-01", periods=40), 3),
        "target_dte_days": np.tile([30, 60, 90], 40),
    })

    assert isd_stats.rows_per_observation_date(frame) == pytest.approx(3.0)
    # 63 trading days of overlap at three rows per day.
    assert isd_stats.pooled_hac_lag(frame) == 189


def test_pooled_lag_follows_the_horizons_present():
    frame = pd.DataFrame({
        "data_date": np.repeat(pd.date_range("2020-01-01", periods=10), 2),
        "target_dte_days": np.tile([30, 60], 10),
    })
    assert isd_stats.pooled_hac_lag(frame) == 84  # 42 days x 2 rows


def test_diebold_mariano_warns_when_the_lag_swamps_the_sample():
    generator = np.random.default_rng(0)
    a = generator.normal(size=40)
    b = generator.normal(size=40)

    with pytest.warns(UserWarning, match="not smaller than the sample size"):
        isd_stats.diebold_mariano(a, b, lag=60)


# ==========================================================================
# 2.4 -- one observation per expiration, chosen without bias
# ==========================================================================

def _panel():
    rows = []
    for month in range(1, 13):
        expiration = pd.Timestamp(2020, month, 20)
        for horizon in (30, 60, 90):
            for offset in range(horizon - 8, horizon + 9, 2):
                data_date = expiration - pd.Timedelta(days=offset)
                rows.append({
                    "data_date": data_date,
                    "expiration_date": expiration,
                    "target_dte_days": horizon,
                    "actual_dte": offset,
                    "pit_constrained": 0.5,
                })
    return pd.DataFrame(rows)


def test_one_row_per_expiration_not_per_expiration_and_horizon():
    panel = _panel()
    reduced = isd_stats.one_per_expiration(panel)

    assert len(reduced) == panel["expiration_date"].nunique() == 12
    assert not reduced["expiration_date"].duplicated().any()


def test_nearest_to_target_is_not_biased_toward_short_maturities():
    """``.last()`` picked the shortest DTE in every bucket; this must not."""
    panel = _panel()

    last = (
        panel.sort_values("data_date")
        .groupby(["expiration_date", "target_dte_days"], as_index=False)
        .last()
    )
    nearest = isd_stats.nearest_to_target(panel)

    for horizon in (30, 60, 90):
        biased = last.loc[last["target_dte_days"].eq(horizon), "actual_dte"]
        neutral = nearest.loc[nearest["target_dte_days"].eq(horizon), "actual_dte"]
        assert biased.mean() < horizon - 5      # systematically short
        assert neutral.mean() == pytest.approx(horizon, abs=1e-9)


def test_effective_sample_size_reports_expirations():
    panel = _panel()
    size = isd_stats.effective_sample_size(panel)
    assert size["distinct_expirations"] == 12
    assert size["rows"] > 12


def test_pit_p_values_are_suppressed_when_parameters_were_fitted():
    generator = np.random.default_rng(1)
    values = generator.uniform(size=200)

    honest = isd_stats.pit_uniformity(values)
    in_sample = isd_stats.pit_uniformity(values, parameters_estimated=1)

    assert np.isfinite(honest["ks_p_value"])
    assert honest["p_values_valid"]
    assert np.isnan(in_sample["ks_p_value"])
    assert np.isnan(in_sample["chi_square_p_value"])
    assert not in_sample["p_values_valid"]
    # The statistic itself is unchanged; only its p-value is withheld.
    assert in_sample["ks_statistic"] == pytest.approx(honest["ks_statistic"])


# ==========================================================================
# 2.5 -- the paired standard error
# ==========================================================================

def _cached_predictions(n_expirations=24, seed=5):
    generator = np.random.default_rng(seed)
    grid = np.linspace(1.0, 2.5 * FORWARD, 1200)
    scale = VOLATILITY * np.sqrt(TAU)
    cached = []

    for index in range(n_expirations):
        forward = FORWARD * float(np.exp(generator.normal(0, 0.05)))
        mean_log = np.log(forward) - 0.5 * scale**2
        density = stats.lognorm.pdf(grid, s=scale, scale=np.exp(mean_log))
        density = density / np.trapezoid(density, grid)
        cached.append({
            "strike_grid": grid,
            "fQ": density,
            "forward": forward,
            "expiration_date": pd.Timestamp(2020, 1, 1) + pd.DateOffset(months=index),
            # Outcomes drawn from a tilt at gamma = 2.
            "realized_value": float(np.exp(
                mean_log + 2.0 * scale**2 + scale * generator.standard_normal()
            )),
        })
    return cached


def test_paired_interval_is_far_tighter_than_the_level_interval():
    """The whole point of 2.5: chain-to-chain variation must cancel."""
    cached = _cached_predictions()
    gamma_grid = np.linspace(0.0, 5.0, 11)
    matrix = evaluation.relative_crps_curve(gamma_grid, cached)
    blocks = [observation["expiration_date"] for observation in cached]

    interval = evaluation.gamma_plausible_interval(
        gamma_grid, matrix, blocks, replicates=200
    )

    best = int(np.argmin(interval["curve"]))
    n = matrix.shape[1]

    # Compare like with like: both standard errors computed the same way.
    # What the interval's width depends on is the paired error NEAR the
    # minimum, since that is where its endpoints are decided.
    level_error = matrix[best].std(ddof=1) / np.sqrt(n)
    adjacent = best + 2 if best + 2 < len(gamma_grid) else best - 2
    paired_error = (matrix[adjacent] - matrix[best]).std(ddof=1) / np.sqrt(n)

    assert paired_error < level_error / 5, (paired_error, level_error)

    # And the bootstrap version, which is what the interval actually uses,
    # is smaller than the level error too.
    assert interval["standard_error"][adjacent] < level_error

    assert interval["plausible_lower"] <= interval["gamma_hat"]
    assert interval["plausible_upper"] >= interval["gamma_hat"]


def test_paired_difference_is_zero_at_the_minimum_and_positive_elsewhere():
    cached = _cached_predictions()
    gamma_grid = np.linspace(0.0, 5.0, 11)
    matrix = evaluation.relative_crps_curve(gamma_grid, cached)
    blocks = [observation["expiration_date"] for observation in cached]

    interval = evaluation.gamma_plausible_interval(
        gamma_grid, matrix, blocks, replicates=100
    )

    assert np.min(interval["mean_difference"]) == pytest.approx(0.0, abs=1e-15)
    assert np.all(interval["mean_difference"] >= -1e-15)


# ==========================================================================
# 2.6 -- what the tilt does
# ==========================================================================

def test_tilting_a_lognormal_is_an_exponential_tilt_in_log_space():
    """``s^gamma LN(m, v)`` is ``LN(m + gamma v, v)``, numerically."""
    grid = np.linspace(1.0, 6.0 * FORWARD, 20000)
    scale = 0.25
    mean_log = np.log(FORWARD) - 0.5 * scale**2
    density = stats.lognorm.pdf(grid, s=scale, scale=np.exp(mean_log))

    gamma = 3.0
    tilted = evaluation.physical_density_from_rn_grid(grid, density, gamma)

    expected_mean_log, expected_variance = evaluation.lognormal_tilt_parameters(
        mean_log, scale**2, gamma
    )
    moments = evaluation.distribution_moments(grid, tilted)

    assert moments["log_mean"] == pytest.approx(expected_mean_log, rel=1e-4)
    assert moments["log_standard_deviation"] == pytest.approx(
        np.sqrt(expected_variance), rel=1e-3
    )


def test_implied_equity_risk_premium_is_gamma_sigma_squared():
    assert evaluation.implied_equity_risk_premium(3, 0.18) == pytest.approx(0.0972)
    assert evaluation.implied_equity_risk_premium(5, 0.22) == pytest.approx(0.242)


def test_the_gamma_signal_in_crps_is_about_one_percent():
    """2.6's structural claim: CRPS is a low-power instrument here."""
    rows = dict(evaluation.crps_location_sensitivity(0.16, TAU))

    assert rows[1.0] == pytest.approx(0.0011, abs=2e-4)
    assert rows[4.0] == pytest.approx(0.0168, abs=1e-3)
    # Monotone, and never more than a couple of per cent of the score.
    values = [rows[g] for g in (1.0, 2.0, 3.0, 4.0)]
    assert values == sorted(values)
    assert max(values) < 0.03


def test_the_tilt_narrows_and_deskews_a_skewed_density():
    """One knob, two problems -- the argument for a second parameter."""
    model = MixtureModel(forward=FORWARD, time_to_expiry=TAU)
    grid = np.linspace(1.0, 2.5 * FORWARD, 6000)
    density = model.density(grid)

    rows = evaluation.tilt_moment_summary(grid, density, gammas=(3.0,))
    base, tilted = rows[0], rows[1]

    assert tilted["mean_change"] > 0                       # lifts the mean
    assert tilted["log_sd_change"] < 0                     # and narrows it
    assert tilted["log_skewness"] > base["log_skewness"]   # and cuts left skew


# ==========================================================================
# 2.7 -- cross-validation must not change the function space
# ==========================================================================

def test_holding_data_out_does_not_shrink_the_basis():
    x = np.linspace(0.5, 1.4, 60)
    y = np.maximum(1.0 - x, 0.0) + 0.02
    scale = np.full_like(x, 1e-3)

    full = build_call_spline_program(x, y, scale, x_upper=2.5)

    keep = np.arange(len(x)) % 5 != 0
    fixed_knots = build_call_spline_program(
        x[keep], y[keep], scale[keep], x_upper=2.5, knot_x=x
    )
    following_knots = build_call_spline_program(
        x[keep], y[keep], scale[keep], x_upper=2.5
    )

    assert fixed_knots["n_coefficients"] == full["n_coefficients"]
    assert following_knots["n_coefficients"] < full["n_coefficients"]
    np.testing.assert_allclose(fixed_knots["knots"], full["knots"])


def test_fit_accepts_an_explicit_knot_set():
    generator = np.random.default_rng(11)
    chain, _ = build_chain(
        pd.Timestamp("2018-03-15"), pd.Timestamp("2018-04-20"), 30,
        2700.0, 0.15, SyntheticMarket(), generator,
    )
    estimates, _ = estimate_forward_and_rate(chain)
    quotes = select_call_equivalent_quotes(
        chain, estimates["forward"], estimates["discount_factor"]
    )
    all_strikes = quotes["strike_price"].to_numpy(dtype=float)
    training = quotes.iloc[::2]

    fit = fit_shape_constrained_call_spline(
        chain, forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
        quotes=training, knot_strikes=all_strikes, roughness_penalty=100.0,
    )
    full = fit_shape_constrained_call_spline(
        chain, forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
        quotes=quotes, roughness_penalty=100.0,
    )

    assert fit["n_knots"] == full["n_knots"]
    assert fit["n_quotes"] < full["n_quotes"]


# ==========================================================================
# 3 -- the synthetic fixture and the external parsers
# ==========================================================================

def test_mixture_prices_are_consistent_with_its_own_density():
    """Breeden-Litzenberger must recover the generating density exactly."""
    model = MixtureModel(forward=FORWARD, time_to_expiry=TAU)
    rate = 0.03
    strikes = np.linspace(0.6 * FORWARD, 1.4 * FORWARD, 4001)

    prices = model.call_price(strikes, rate)
    step = strikes[1] - strikes[0]
    second = (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / step**2
    recovered = np.exp(rate * TAU) * second

    np.testing.assert_allclose(
        recovered, model.density(strikes[1:-1]), atol=2e-8
    )


def test_mixture_mean_is_the_forward():
    model = MixtureModel(forward=FORWARD, time_to_expiry=TAU)
    grid = np.linspace(1.0, 6.0 * FORWARD, 40000)
    moments = evaluation.distribution_moments(grid, model.density(grid))
    assert moments["mean"] == pytest.approx(FORWARD, rel=1e-5)


def test_the_tilted_mixture_is_the_tilted_density():
    """``MixtureModel.tilted`` must agree with the grid-based tilt."""
    model = MixtureModel(forward=FORWARD, time_to_expiry=TAU)
    grid = np.linspace(1.0, 4.0 * FORWARD, 20000)

    gamma = 2.5
    by_grid = evaluation.physical_density_from_rn_grid(
        grid, model.density(grid), gamma
    )
    by_model = model.tilted(gamma).density(grid)
    by_model = by_model / np.trapezoid(by_model, grid)

    np.testing.assert_allclose(by_grid, by_model, atol=1e-9)


def test_synthetic_chain_supports_the_pipeline():
    generator = np.random.default_rng(21)
    market = SyntheticMarket()
    chain, model = build_chain(
        pd.Timestamp("2018-03-15"), pd.Timestamp("2018-04-20"), 30,
        2700.0, 0.15, market, generator,
    )

    estimates, _ = estimate_forward_and_rate(chain)
    # The parity regression should recover the generating forward closely.
    assert estimates["forward"] == pytest.approx(model.forward, rel=2e-4)
    assert estimates["passes_quality"]


def test_risk_neutral_skewness_of_a_lognormal_matches_the_closed_form():
    """For LN(m, s^2) the log-return skewness is zero."""
    grid = np.linspace(1.0, 6.0 * FORWARD, 40000)
    scale = 0.2
    density = stats.lognorm.pdf(grid, s=scale, scale=FORWARD)
    assert validation.risk_neutral_skewness(grid, density) == pytest.approx(
        0.0, abs=1e-3
    )


def test_skew_index_conversion():
    assert validation.skew_index_equivalent(-1.5) == pytest.approx(115.0)
    assert validation.skew_index_equivalent(0.0) == pytest.approx(100.0)


def test_external_parsers_read_the_published_headers(tmp_path, monkeypatch):
    """The published CSV layouts, as of September 2026."""
    fred = tmp_path / "fred.csv"
    fred.write_text(
        "observation_date,DGS3MO\n2020-01-02,1.54\n2020-01-03,.\n2020-01-06,1.56\n"
    )
    monkeypatch.setattr(
        validation, "FRED_CSV_URL", str(fred).replace("{", "{{") + "{series}"[:0] or str(fred)
    )
    frame = validation.fetch_fred_series("DGS3MO")
    assert len(frame) == 2                      # the '.' row is dropped
    assert frame["value"].iloc[0] == pytest.approx(0.0154)

    cboe = tmp_path / "skew.csv"
    cboe.write_text("DATE,SKEW\n01/02/1990,126.09\n01/03/1990,123.34\n")
    monkeypatch.setattr(validation, "CBOE_INDEX_HISTORY_URL", str(cboe))
    skew = validation.fetch_cboe_index("SKEW")
    assert list(skew.columns) == ["date", "close"]
    assert skew["close"].iloc[0] == pytest.approx(126.09)

    shiller = tmp_path / "shiller.csv"
    shiller.write_text(
        "Date,SP500,Dividend,Earnings\n"
        "2020-01-01,3278.20,58.24,139.47\n"
        "2020-02-01,3277.31,58.51,139.72\n"
    )
    monkeypatch.setattr(validation, "SHILLER_SP500_CSV_URL", str(shiller))
    yields = validation.fetch_dividend_yield(cache_path=None)
    assert yields["dividend_yield"].iloc[0] == pytest.approx(58.24 / 3278.20)


def test_compare_series_aligns_and_summarises():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    implied = pd.Series(np.linspace(0.01, 0.05, 100), index=dates)
    reference = pd.Series(np.linspace(0.011, 0.051, 100), index=dates)

    aligned, summary = validation.compare_series(implied, reference)

    assert summary["n_observations"] == 100
    assert summary["correlation"] == pytest.approx(1.0, abs=1e-9)
    assert summary["bias"] == pytest.approx(-0.001, abs=1e-9)
    assert "by_year" in summary


def test_failure_rate_by_bucket_splits_by_volatility():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    attempted = pd.DataFrame({
        "data_date": dates,
        "expiration_date": dates + pd.Timedelta(days=30),
        "target_dte_days": 30,
    })
    failed = attempted.iloc[:10]
    vix = pd.DataFrame({
        "date": dates,
        "close": np.concatenate([np.full(10, 35.0), np.full(50, 12.0)]),
    })

    result = validation.failure_rate_by_bucket(attempted, failed, vix=vix)

    by_bucket = result["by_vix_bucket"]["failure_rate"]
    assert by_bucket.loc["VIX > 30"] == pytest.approx(1.0)
    assert by_bucket.loc["VIX < 15"] == pytest.approx(0.0)
