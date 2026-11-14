"""One chain end to end: parity fit, both smoothers, a benchmark, the scores.

The analysis notebook and the synthetic demo both call ``score_chain``, so
the numbers they report come from one implementation of the steps below.

1. Forward and discount factor from put-call parity (``forwards``).
2. One out-of-the-money quote per strike, as a call price (``fitting``).
3. The shape-constrained call-price spline and its density (``fitting``).
4. The implied-volatility baseline on the same quotes (``smoothing``).
5. The baseline recentred on the parity forward, which separates a location
   difference from a shape difference (``evaluation.recentre_on_forward``).
6. A lognormal at the chain's own at-the-money volatility (``benchmarks``).
7. CRPS and PIT of every forecast against the realised value, all tabulated
   on one shared strike grid (``evaluation``).
"""

from __future__ import annotations

import numpy as np

from .benchmarks import atm_implied_volatility, lognormal_forecast
from .evaluation import (
    cdf_from_density,
    common_strike_grid,
    crps_from_cdf,
    evaluate_distribution,
    recentre_on_forward,
)
from .fitting import (
    construct_risk_neutral_distribution_from_spline,
    fit_shape_constrained_call_spline,
    select_call_equivalent_quotes,
)
from .forwards import estimate_forward_and_rate
from .smoothing import create_density_from_chain
from .validation import risk_neutral_skewness

__all__ = ["score_chain"]


def score_chain(chain, realized_value: float, roughness_penalty: float) -> dict:
    """Fit both methods to one chain and score them against the outcome.

    Raises a ``ChainError`` (or ``ValueError``) when the chain cannot be
    used; callers collect those as failures.

    Returns
    -------
    dict
        ``scores``: one flat row of numbers for the results table.
        ``cached``: what the CRRA tilt in Part 3 needs to re-score the
        risk-neutral density at any gamma without refitting.
        ``estimates``, ``fit``, ``constrained`` and ``iv``: the intermediate
        objects, for plots and for checks against a known truth.
    """
    realized_value = float(realized_value)

    estimates, _ = estimate_forward_and_rate(chain)
    forward = estimates["forward"]
    risk_free_rate = estimates["risk_free_rate"]

    quotes = select_call_equivalent_quotes(
        chain, forward, estimates["discount_factor"]
    )

    fit = fit_shape_constrained_call_spline(
        chain,
        forward=forward,
        risk_free_rate=risk_free_rate,
        roughness_penalty=roughness_penalty,
        quotes=quotes,
    )

    # One grid for every forecast scored against this outcome, so that a
    # difference in tabulation can never look like a difference in forecast.
    grid = common_strike_grid(
        forward,
        extra_points=np.concatenate([
            fit["breakpoints"] * forward,
            quotes["strike_price"].to_numpy(dtype=float),
        ]),
    )

    constrained = construct_risk_neutral_distribution_from_spline(
        fit, strike_grid=grid
    )
    iv_result = create_density_from_chain(
        chain,
        forward=forward,
        risk_free_rate=risk_free_rate,
        quotes=quotes,
        strike_grid=grid,
    )

    recentred_grid, recentred_density = recentre_on_forward(
        iv_result["strike_grid"], iv_result["density"], forward
    )
    recentred_grid, recentred_cdf = cdf_from_density(
        recentred_grid, recentred_density
    )

    atm_volatility = atm_implied_volatility(iv_result["iv_of_strike"], forward)
    benchmark = lognormal_forecast(
        forward, atm_volatility, estimates["time_to_expiry"], realized_value,
        strike_grid=grid,
    )

    constrained_score = evaluate_distribution(
        constrained["strike_grid"], constrained["cdf"], realized_value, forward,
        density=constrained["density"],
    )
    iv_score = evaluate_distribution(
        iv_result["strike_grid"], iv_result["cdf"], realized_value, forward,
        density=iv_result["density"],
    )

    scores = {
        "forward": forward,
        "risk_free_rate": risk_free_rate,
        "atm_volatility": atm_volatility,
        "realized_value": realized_value,
        "n_quotes": fit["n_quotes"],
        "solver_inaccurate": fit["solver_inaccurate"],
        "mass_below_traded": fit["mass_below_traded"],
        "mass_above_traded": fit["mass_above_traded"],

        # Where the baseline's mean actually sits, and how hard its smile
        # is working.
        "negative_mass_clipped": iv_result["negative_mass_clipped"],
        "mass_before_normalisation": iv_result["mass_before_normalisation"],
        "iv_mean_over_forward": iv_result["mean_over_forward"],
        "n_smile_knots": iv_result["n_smile_knots"],

        # Shape, for the SKEW validation.
        "risk_neutral_skewness": risk_neutral_skewness(
            constrained["strike_grid"], constrained["density"]
        ),

        "crps_constrained": constrained_score["crps"],
        "relative_crps_constrained": constrained_score["relative_crps"],
        "pit_constrained": constrained_score["pit"],

        "crps_iv": iv_score["crps"],
        "relative_crps_iv": iv_score["relative_crps"],
        "pit_iv": iv_score["pit"],

        "relative_crps_iv_recentred": (
            crps_from_cdf(recentred_grid, recentred_cdf, realized_value) / forward
        ),

        "crps_lognormal": benchmark["crps"],
        "relative_crps_lognormal": benchmark["relative_crps"],
        "pit_lognormal": benchmark["pit"],
    }

    cached = {
        "realized_value": realized_value,
        "forward": float(forward),
        "traded_range": fit["strike_bounds"],
        "strike_grid": constrained["strike_grid"],
        "fQ": constrained["density"],
    }

    return {
        "scores": scores,
        "cached": cached,
        "estimates": estimates,
        "fit": fit,
        "constrained": constrained,
        "iv": iv_result,
    }
