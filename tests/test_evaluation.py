"""Scoring, checked against closed forms.

The Gaussian CRPS has an exact expression, which pins down both the
integration scheme and how fine the grid has to be. Everything else in the
project is scored through the same function.
"""

import numpy as np
import pytest
from scipy.stats import norm

from implied_stock_distributions.benchmarks import gaussian_crps
from implied_stock_distributions.errors import ChainQualityError
from implied_stock_distributions.evaluation import (
    cdf_from_density,
    crps_from_cdf,
    crps_from_density_grid,
    mean_relative_crps_for_gamma,
    physical_density_from_rn_grid,
    probability_integral_transform,
)

MEAN = 4500.0
SIGMA = 250.0


def _gaussian_grid(points=20001, width=10.0):
    grid = np.linspace(MEAN - width * SIGMA, MEAN + width * SIGMA, points)
    return grid, norm.cdf(grid, MEAN, SIGMA)


@pytest.mark.parametrize("realized", [3800.0, 4200.0, 4500.0, 5000.0, 5400.0])
def test_crps_matches_the_gaussian_closed_form(realized):
    grid, cdf = _gaussian_grid()
    numeric = crps_from_cdf(grid, cdf, realized)
    exact = gaussian_crps(MEAN, SIGMA, realized)
    assert numeric == pytest.approx(exact, rel=2e-3)


def test_crps_of_a_point_mass_on_the_outcome_is_one_cell_of_ramp():
    """The score integrates the piecewise-linear interpolant of the CDF.

    Tabulated on a grid of spacing ``h``, a step at the outcome becomes a
    ramp across the cell just below it, so the exact score of that
    interpolant is ``h / 3`` -- and it vanishes as the grid is refined.
    """
    for points in (1001, 100001):
        grid = np.linspace(4000.0, 5000.0, points)
        cdf = (grid >= 4500.0).astype(float)
        h = grid[1] - grid[0]
        assert crps_from_cdf(grid, cdf, 4500.0) == pytest.approx(h / 3, rel=1e-9)


def test_crps_handles_an_outcome_outside_the_support():
    """Below the grid the integrand is one, so the penalty is the distance."""
    grid = np.linspace(4000.0, 5000.0, 20001)
    cdf = np.clip((grid - 4000.0) / 1000.0, 0.0, 1.0)

    inside = crps_from_cdf(grid, cdf, 4000.0)
    outside = crps_from_cdf(grid, cdf, 3900.0)

    assert outside == pytest.approx(inside + 100.0, rel=1e-9)


def test_density_and_cdf_routes_agree():
    grid = np.linspace(MEAN - 10 * SIGMA, MEAN + 10 * SIGMA, 20001)
    density = norm.pdf(grid, MEAN, SIGMA)

    from_density = crps_from_density_grid(grid, density, 4300.0)
    from_cdf = crps_from_cdf(grid, norm.cdf(grid, MEAN, SIGMA), 4300.0)

    assert from_density == pytest.approx(from_cdf, rel=1e-4)


def test_rejects_a_cdf_that_is_not_a_distribution():
    grid = np.linspace(4000.0, 5000.0, 1001)
    with pytest.raises(ChainQualityError):
        crps_from_cdf(grid, np.linspace(0.0, 0.6, 1001), 4500.0)


def test_pit_is_uniform_for_calibrated_forecasts():
    """Draw outcomes from the forecast; the PITs must look uniform."""
    generator = np.random.default_rng(11)
    grid, cdf = _gaussian_grid(points=4001)

    draws = generator.normal(MEAN, SIGMA, size=4000)
    values = np.array([
        probability_integral_transform(grid, cdf, draw) for draw in draws
    ])

    assert values.mean() == pytest.approx(0.5, abs=0.02)
    counts, _ = np.histogram(values, bins=10, range=(0, 1))
    assert counts.min() > 0.75 * counts.mean()


def test_pit_detects_a_biased_forecast():
    grid, cdf = _gaussian_grid(points=4001)
    generator = np.random.default_rng(5)

    # Outcomes drawn well below the forecast centre.
    draws = generator.normal(MEAN - SIGMA, SIGMA, size=2000)
    values = np.array([
        probability_integral_transform(grid, cdf, draw) for draw in draws
    ])
    assert values.mean() < 0.40


def test_crra_tilt_moves_mass_upward_and_stays_normalised():
    grid = np.linspace(1.0, 9000.0, 20001)
    density = norm.pdf(grid, MEAN, SIGMA)
    density = density / np.trapezoid(density, grid)

    tilted = physical_density_from_rn_grid(grid, density, gamma=3.0)

    assert np.trapezoid(tilted, grid) == pytest.approx(1.0, rel=1e-8)
    assert np.trapezoid(grid * tilted, grid) > np.trapezoid(grid * density, grid)

    unchanged = physical_density_from_rn_grid(grid, density, gamma=0.0)
    assert np.allclose(unchanged, density, rtol=1e-8)


def test_gamma_objective_uses_relative_scores():
    """A high-index observation must not dominate a low-index one.

    Two observations with identical *relative* errors, one at SPX 1000 and
    one at SPX 5000, must contribute equally. Averaging absolute CRPS would
    give the second five times the weight.
    """
    def observation(level):
        grid = np.linspace(1.0, 4.0 * level, 8001)
        density = norm.pdf(grid, level, 0.06 * level)
        density = density / np.trapezoid(density, grid)
        return {
            "strike_grid": grid,
            "fQ": density,
            "realized_value": 0.97 * level,
            "forward": level,
        }

    small = mean_relative_crps_for_gamma(1.0, [observation(1000.0)])
    large = mean_relative_crps_for_gamma(1.0, [observation(5000.0)])
    both = mean_relative_crps_for_gamma(
        1.0, [observation(1000.0), observation(5000.0)]
    )

    assert small == pytest.approx(large, rel=5e-3)
    assert both == pytest.approx(0.5 * (small + large), rel=1e-6)


def test_grid_with_coincident_points_is_handled():
    """The fitted grid merges a linspace with the spline breakpoints.

    Storing it in a narrower float type collapses neighbouring points onto
    each other. A repeated abscissa carries no information and contributes a
    zero-width interval, so it is dropped rather than treated as an error.
    """
    grid = np.linspace(1.0, 9000.0, 4001)
    density = norm.pdf(grid, MEAN, SIGMA)
    density = density / np.trapezoid(density, grid)

    exact = crps_from_density_grid(grid, density, 4300.0)
    narrowed = crps_from_density_grid(
        grid.astype(np.float32), density.astype(np.float32), 4300.0
    )
    assert narrowed == pytest.approx(exact, rel=1e-4)

    repeated = np.concatenate([grid, grid[:10]])
    order = np.argsort(repeated)
    assert crps_from_density_grid(
        repeated[order],
        np.concatenate([density, density[:10]])[order],
        4300.0,
    ) == pytest.approx(exact, rel=1e-6)


def test_tilted_distribution_round_trips_through_the_scorers():
    from implied_stock_distributions.evaluation import (
        pit_for_gamma,
        relative_crps_for_gamma,
        tilted_distribution,
    )

    grid = np.linspace(1.0, 9000.0, 4001)
    density = norm.pdf(grid, MEAN, SIGMA)
    density = density / np.trapezoid(density, grid)

    observation = {
        "strike_grid": grid,
        "fQ": density,
        "realized_value": 4300.0,
        "forward": MEAN,
        "traded_range": (3000.0, 6000.0),
    }

    tilted_grid, tilted_density, cdf = tilted_distribution(observation, 2.0)
    assert np.trapezoid(tilted_density, tilted_grid) == pytest.approx(1.0, rel=1e-6)
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-6)

    assert relative_crps_for_gamma(2.0, [observation]).shape == (1,)
    assert 0.0 <= pit_for_gamma(2.0, [observation])[0] <= 1.0

    # A range that already holds all the mass must not change the answer;
    # a narrow one must.
    covering = relative_crps_for_gamma(2.0, [observation], tilt_range="traded")
    full = relative_crps_for_gamma(2.0, [observation])
    assert covering[0] == pytest.approx(full[0], rel=1e-9)

    narrow = relative_crps_for_gamma(2.0, [observation], tilt_range=(4400.0, 4600.0))
    assert narrow[0] != pytest.approx(full[0], rel=1e-6)


def test_restricted_tilt_is_continuous():
    """Outside the range the kernel holds its boundary value; it never jumps."""
    grid = np.linspace(1.0, 9000.0, 9001)
    density = norm.pdf(grid, MEAN, SIGMA)
    density = density / np.trapezoid(density, grid)

    low, high = 4300.0, 4700.0
    tilted = physical_density_from_rn_grid(grid, density, 4.0, tilt_range=(low, high))
    ratio = tilted / density

    # The likelihood ratio is constant below the range, constant above it,
    # rises through it, and meets both constants without a jump.
    below, above = ratio[grid < low], ratio[grid > high]
    assert np.allclose(below, below[0]) and np.allclose(above, above[0])
    inside = ratio[(grid >= low) & (grid <= high)]
    assert np.all(np.diff(inside) > 0)
    assert inside[0] == pytest.approx(below[0], rel=1e-3)
    assert inside[-1] == pytest.approx(above[0], rel=1e-3)


def test_cdf_from_density_rejects_negative_mass():
    grid = np.linspace(1.0, 100.0, 501)
    density = np.full_like(grid, 0.01)
    density[100] = -5.0
    with pytest.raises(ChainQualityError):
        cdf_from_density(grid, density)
