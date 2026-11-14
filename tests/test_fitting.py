"""End-to-end recovery of a known density, and the identities the fit obeys.

The strongest check available here: price a chain from Black's formula at a
known volatility, run the whole pipeline on it, and compare the recovered
risk-neutral density with the analytic lognormal.

The constraint set also implies two exact identities, which hold whatever
the data are and are therefore checked on every fit:

    integral f_Q = g'(x_upper) - g'(0) = 1
    E_Q[S_T]     = F (g(0) - g(x_upper)) = F
"""

import numpy as np
import pytest

from implied_stock_distributions.errors import ChainQualityError
from implied_stock_distributions.fitting import (
    build_call_spline_program,
    constraint_bounds,
    construct_risk_neutral_distribution_from_spline,
    fit_shape_constrained_call_spline,
    select_call_equivalent_quotes,
)
from implied_stock_distributions.forwards import estimate_forward_and_rate
from implied_stock_distributions.smoothing import create_density_from_chain

from .synthetic import lognormal_cdf, lognormal_density, synthetic_chain


def _fit(chain, **kwargs):
    estimates, _ = estimate_forward_and_rate(chain)
    return fit_shape_constrained_call_spline(
        chain,
        forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
        **kwargs,
    )


def test_recovers_the_lognormal_density():
    chain, truth = synthetic_chain(strike_step=50.0)
    distribution = construct_risk_neutral_distribution_from_spline(
        _fit(chain, roughness_penalty=10.0)
    )

    grid = distribution["strike_grid"]
    exact = lognormal_density(
        grid, truth["forward"], truth["volatility"], truth["time_to_expiry"]
    )

    traded = (grid > 0.62 * truth["forward"]) & (grid < 1.3 * truth["forward"])
    l1_error = np.trapezoid(
        np.abs(distribution["density"][traded] - exact[traded]), grid[traded]
    )
    assert l1_error < 0.05


def test_recovers_the_lognormal_cdf():
    chain, truth = synthetic_chain(strike_step=50.0)
    distribution = construct_risk_neutral_distribution_from_spline(
        _fit(chain, roughness_penalty=10.0)
    )

    grid = distribution["strike_grid"]
    exact = lognormal_cdf(
        grid, truth["forward"], truth["volatility"], truth["time_to_expiry"]
    )
    assert np.max(np.abs(distribution["cdf"] - exact)) < 0.02


def test_mass_is_one_and_mean_is_the_forward():
    """Both follow from the boundary conditions, not from renormalising."""
    chain, truth = synthetic_chain(strike_step=50.0, noise=0.4, seed=7)
    fit = _fit(chain, roughness_penalty=100.0)
    distribution = construct_risk_neutral_distribution_from_spline(fit)

    assert distribution["mass"] == pytest.approx(1.0, abs=1e-4)
    assert distribution["mean"] == pytest.approx(fit["forward"], rel=1e-4)
    assert distribution["cdf_start"] == pytest.approx(0.0, abs=1e-6)
    assert distribution["cdf_end"] == pytest.approx(1.0, abs=1e-6)


def test_density_is_never_negative():
    """The point of the constraints: no arbitrage even with noisy quotes."""
    chain, _ = synthetic_chain(strike_step=25.0, noise=0.8, skew=-0.35, seed=13)
    fit = _fit(chain, roughness_penalty=100.0)
    distribution = construct_risk_neutral_distribution_from_spline(fit)

    assert distribution["minimum_density"] > -1e-8


def test_call_curve_is_decreasing_and_convex():
    chain, _ = synthetic_chain(strike_step=25.0, noise=0.8, seed=17)
    fit = _fit(chain, roughness_penalty=100.0)

    grid = np.linspace(0.0, fit["x_upper"], 4000)
    first = fit["spline"].derivative(1)(grid)
    second = fit["spline"].derivative(2)(grid)

    # Curvature is constrained at the breakpoints and is linear between
    # them, so non-negativity everywhere follows -- up to whatever slack
    # the solver leaves, which is measured against the size of the density
    # rather than against zero.
    assert np.all(first <= 1e-6)
    assert np.all(first >= -1 - 1e-6)
    assert np.all(second >= -1e-4 * second.max())


def test_a_larger_penalty_gives_a_smoother_fit():
    chain, _ = synthetic_chain(strike_step=25.0, noise=1.0, seed=23)

    rough = _fit(chain, roughness_penalty=1.0)
    smooth = _fit(chain, roughness_penalty=1e4)

    def curvature(fit):
        grid = np.linspace(0.0, fit["x_upper"], 2000)
        return float(np.trapezoid(fit["spline"].derivative(2)(grid) ** 2, grid))

    assert curvature(smooth) < curvature(rough)
    assert (
        smooth["quotes"]["standardized_residual"].abs().mean()
        > rough["quotes"]["standardized_residual"].abs().mean()
    )


def test_extrapolated_mass_is_reported():
    """A chain with a narrow strike range must admit how much it invented."""
    narrow, _ = synthetic_chain(
        strike_step=25.0, lower_moneyness=0.90, upper_moneyness=1.10
    )
    wide, _ = synthetic_chain(
        strike_step=25.0, lower_moneyness=0.50, upper_moneyness=1.50
    )

    narrow_fit = _fit(narrow, roughness_penalty=100.0)
    wide_fit = _fit(wide, roughness_penalty=100.0)

    assert narrow_fit["extrapolated_mass"] > wide_fit["extrapolated_mass"]
    assert narrow_fit["extrapolated_mass"] > 0.01


def test_constraint_bounds_describe_the_same_program():
    chain, _ = synthetic_chain(strike_step=100.0)
    estimates, _ = estimate_forward_and_rate(chain)

    quotes = select_call_equivalent_quotes(
        chain, estimates["forward"], estimates["discount_factor"]
    )
    x = quotes["strike_price"].to_numpy(float) / estimates["forward"]
    y = quotes["call_equivalent_price"].to_numpy(float) / (
        estimates["discount_factor"] * estimates["forward"]
    )

    program = build_call_spline_program(x, y, np.full_like(y, 1e-4), 2.5)
    matrix, lower, upper = constraint_bounds(program)

    assert matrix.shape[0] == 3 * len(program["breakpoints"])
    assert np.all(lower <= upper)
    # The six boundary conditions are the rows where lower equals upper.
    assert int(np.sum(lower == upper)) == 6


def test_duplicate_strikes_are_rejected():
    """Two settlement conventions merged into one chain must not fit quietly."""
    import pandas as pd

    chain, _ = synthetic_chain(strike_step=50.0)
    estimates, _ = estimate_forward_and_rate(chain)

    # An out-of-the-money call, so it survives quote selection.
    duplicated = chain.loc[
        chain["put_call"].eq("call")
        & chain["strike_price"].gt(estimates["forward"])
    ].head(1).copy()
    duplicated["mid_price"] *= 1.02

    contaminated = pd.concat([chain, duplicated], ignore_index=True)

    with pytest.raises(ChainQualityError):
        select_call_equivalent_quotes(
            contaminated, estimates["forward"], estimates["discount_factor"]
        )


def test_iv_baseline_also_recovers_the_lognormal():
    """The baseline must be a fair one: right answer on clean data."""
    chain, truth = synthetic_chain(strike_step=25.0)
    estimates, _ = estimate_forward_and_rate(chain)

    result = create_density_from_chain(
        chain,
        forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
    )

    grid = result["strike_grid"]
    exact = lognormal_density(
        grid, truth["forward"], truth["volatility"], truth["time_to_expiry"]
    )
    traded = (grid > 0.62 * truth["forward"]) & (grid < 1.3 * truth["forward"])

    assert np.trapezoid(
        np.abs(result["density"][traded] - exact[traded]), grid[traded]
    ) < 0.05
    assert result["iv_of_strike"](truth["forward"]) == pytest.approx(
        truth["volatility"], abs=1e-3
    )


def test_iv_baseline_uses_both_wings():
    """It must see the same strikes as the constrained fit, not calls only."""
    chain, _ = synthetic_chain(strike_step=25.0)
    estimates, _ = estimate_forward_and_rate(chain)

    result = create_density_from_chain(
        chain,
        forward=estimates["forward"],
        risk_free_rate=estimates["risk_free_rate"],
    )

    strikes = result["observed_strikes"]
    assert strikes.min() < 0.75 * estimates["forward"]
    assert strikes.max() > 1.15 * estimates["forward"]
