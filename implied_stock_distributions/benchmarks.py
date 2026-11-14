"""Reference forecasts, so that a CRPS number means something.

A relative CRPS of 0.03 is uninterpretable on its own, and "better than my
other smoothing method" is a weak claim. These give the option-implied
distributions something to beat.

For a calibrated Gaussian forecast the expected CRPS is ``sigma / sqrt(pi)``,
which sets the scale: at sixteen percent annualised volatility that is about
2.6 percent of spot at thirty days and 4.5 percent at ninety. The closed
forms here are also the exact checks used in the test suite -- ``gaussian_crps``
and ``lognormal_crps`` are what the tabulated quadrature in
``evaluation.crps_from_cdf`` is verified against.

Every forecast here accepts an explicit ``strike_grid``. Pass the same one
the option-implied methods are tabulated on: before the quadrature was made
exact, the benchmark's own 4000-point grid gave it about half a per cent
less quadrature bias than the option methods carried, which understated
their skill score against it by roughly that much. Sharing the grid removes
the question rather than arguing about its size.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .evaluation import crps_from_cdf, probability_integral_transform

__all__ = [
    "gaussian_crps",
    "lognormal_crps",
    "lognormal_cdf",
    "lognormal_forecast",
    "climatology_forecast",
    "atm_implied_volatility",
]


def gaussian_crps(mean, standard_deviation, realized_value):
    """Closed-form CRPS of a normal forecast (Gneiting and Raftery, 2007).

    ``CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi) ]``, with
    ``z = (y - mu) / sigma``.
    """
    z = (np.asarray(realized_value, float) - mean) / standard_deviation
    return standard_deviation * (
        z * (2.0 * stats.norm.cdf(z) - 1.0)
        + 2.0 * stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def lognormal_crps(mean_log, sigma_log, realized_value):
    """Closed-form CRPS of a lognormal forecast.

    For ``log X ~ N(m, s^2)`` and an outcome ``y > 0``, writing
    ``w = (log y - m) / s``,

        CRPS = y (2 Phi(w) - 1)
               - 2 exp(m + s^2 / 2) [ Phi(w - s) + Phi(s / sqrt(2)) - 1 ].

    This is the exact value the tabulated quadrature has to reproduce, and
    it is what makes the accuracy of ``evaluation.crps_from_cdf`` a tested
    property rather than an assertion.
    """
    y = np.asarray(realized_value, dtype=float)
    m = np.asarray(mean_log, dtype=float)
    s = np.asarray(sigma_log, dtype=float)

    w = (np.log(y) - m) / s

    return (
        y * (2.0 * stats.norm.cdf(w) - 1.0)
        - 2.0 * np.exp(m + 0.5 * s**2) * (
            stats.norm.cdf(w - s)
            + stats.norm.cdf(s / np.sqrt(2.0))
            - 1.0
        )
    )


def lognormal_cdf(strike_grid, forward, volatility, time_to_expiry):
    """CDF of a lognormal with mean ``forward`` on the given grid."""
    strike_grid = np.asarray(strike_grid, dtype=float)
    sigma_root_t = volatility * np.sqrt(time_to_expiry)

    cdf = np.zeros_like(strike_grid)
    positive = strike_grid > 0

    z = (
        np.log(strike_grid[positive] / forward) + 0.5 * sigma_root_t**2
    ) / sigma_root_t

    cdf[positive] = stats.norm.cdf(z)
    return cdf


def lognormal_forecast(
    forward,
    volatility,
    time_to_expiry,
    realized_value,
    n_grid: int = 4000,
    width: float = 8.0,
    strike_grid=None,
):
    """Score a lognormal forecast centred on the forward.

    Supplying the chain's at-the-money implied volatility makes this the
    natural "the market's level and the market's volatility, but no smile"
    benchmark; supplying a trailing realised volatility makes it a
    historical-volatility benchmark instead.

    Parameters
    ----------
    strike_grid : array_like, optional
        Tabulate on this grid instead of an internally chosen one. Pass
        ``evaluation.common_strike_grid(...)``, the same grid the two
        option-implied methods use, so that the skill score compares
        forecasts rather than quadratures. Note that this truncates the
        lognormal to the shared support and renormalises it there, exactly
        as the option methods are truncated -- which is what makes the
        comparison fair, and is reported as ``truncated_mass``.
    """
    sigma_root_t = volatility * np.sqrt(time_to_expiry)

    if strike_grid is None:
        upper = forward * np.exp(width * sigma_root_t)
        grid = np.linspace(1e-6 * forward, upper, n_grid)
    else:
        grid = np.unique(np.asarray(strike_grid, dtype=float))
        grid = grid[np.isfinite(grid) & (grid > 0)]

    cdf = lognormal_cdf(grid, forward, volatility, time_to_expiry)

    truncated_mass = float(cdf[-1] - cdf[0])
    cdf = (cdf - cdf[0]) / truncated_mass

    score = crps_from_cdf(grid, cdf, realized_value)

    return {
        "crps": score,
        "relative_crps": score / forward,
        "pit": probability_integral_transform(grid, cdf, realized_value),
        "strike_grid": grid,
        "cdf": cdf,
        "truncated_mass": truncated_mass,
    }


def climatology_forecast(
    reference_level,
    historical_log_returns,
    realized_value,
    n_grid: int = 4000,
    strike_grid=None,
):
    """Score the empirical distribution of past returns over the horizon.

    The forecast is the reference level scaled by the historical h-period
    log returns supplied, which uses no option information at all. Pass
    only returns observed before the forecast date to keep it honest.

    This is the benchmark that makes Part 3 a real question. "The tilted
    density beats the untilted risk-neutral density" is a low bar -- the
    untilted one is not a forecast and is not claimed to be. "The
    option-implied physical density beats a time-series model at the same
    horizon, out of sample" is the claim a quant researcher cares about,
    and it is a result whichever way it comes out.
    """
    returns = np.asarray(historical_log_returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    if returns.size < 20:
        raise ValueError("Too few historical returns for a climatology.")

    outcomes = np.sort(reference_level * np.exp(returns))

    if strike_grid is None:
        grid = np.linspace(outcomes[0] * 0.5, outcomes[-1] * 1.5, n_grid)
    else:
        grid = np.unique(np.asarray(strike_grid, dtype=float))
        grid = grid[np.isfinite(grid) & (grid > 0)]

    cdf = np.searchsorted(outcomes, grid, side="right") / outcomes.size

    # An empirical CDF tabulated on a shared grid need not start at zero or
    # end at one; renormalise onto the shared support the same way every
    # other forecast here is.
    span = float(cdf[-1] - cdf[0])
    if span <= 0:
        raise ValueError(
            "The climatology places no mass on the supplied strike grid."
        )
    cdf = (cdf - cdf[0]) / span

    score = crps_from_cdf(grid, cdf, realized_value)

    return {
        "crps": score,
        "relative_crps": score / reference_level,
        "pit": probability_integral_transform(grid, cdf, realized_value),
        "strike_grid": grid,
        "cdf": cdf,
        "truncated_mass": span,
        "n_returns": int(returns.size),
    }


def atm_implied_volatility(iv_of_strike, forward):
    """Fitted implied volatility at the forward."""
    return float(np.asarray(iv_of_strike(forward)).ravel()[0])
