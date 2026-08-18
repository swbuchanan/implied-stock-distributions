import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


from .fitting import fit_shape_constrained_call_spline, construct_risk_neutral_distribution_from_spline
from .forwards import estimate_forward_and_rate
from .smoothing import create_density_from_chain


def crps_from_cdf(
    strike_grid,
    cdf,
    realized_value,
):
    strike_grid = np.asarray(strike_grid, dtype=float)
    cdf = np.asarray(cdf, dtype=float)

    valid = (
        np.isfinite(strike_grid) &\
        np.isfinite(cdf)
    )
    strike_grid = strike_grid[valid]
    cdf = cdf[valid]

    order = np.argsort(strike_grid)

    strike_grid = strike_grid[order]
    cdf = cdf[order]

    if np.any(np.diff(strike_grid) <= 0):
        raise ValueError("Strike grid must be strictly increasing.")

    if not (
        strike_grid.min()
        <= realized_value
        <= strike_grid.max()
    ):
        raise ValueError("The realized value is outside the density grid.")

    if not (
        np.isclose(cdf[0], 0, atol=0.001) &\
        np.isclose(cdf[-1], 1, atol=0.001)
    ):
        raise ValueError(f"Something is wrong with this CDF: it goes from {cdf[0]} to {cdf[-1]}: {cdf}")


    indicator = (
        strike_grid >= realized_value
    ).astype(float)

    return float(
        np.trapezoid(
            (cdf - indicator) ** 2,
            strike_grid,
        )
    )

def evaluate_risk_neutral_prediction_from_chain(
    chain,
    realized_value,
    option_type="call", # TODO: where does option type come into these things; I know it's not used here but I think somewhere it is relevant?
):
    """
    Construct and evaluate the risk-neutral density for one chain.

    This function uses Fengler's method of shape-constrained smoothing

    realized_values should be a number containing the actual realized value
    """

    expiry_dates = (
        chain["expiration_date"]
        .dropna()
        .drop_duplicates()
    )

    if len(expiry_dates) != 1:
        raise ValueError(
            "Expected exactly one expiration date."
        )

    expiry_date = pd.Timestamp(
        expiry_dates.iloc[0]
    ).normalize()

    # if expiry_date not in realized_values.index:
    #     raise KeyError(
    #         f"No realized value found for {expiry_date.date()}."
    #     )

    # realized_value = float(
    #     realized_values.loc[expiry_date]
    # )

    estimates, _ = estimate_forward_and_rate(
        chain,
        use_spread_weights=False,
    )

    # strike_grid, smooth_prices, density, iv_curve = (
    #     create_density_from_chain(
    #         chain=chain,
    #         forward=estimates["forward"],
    #         risk_free_rate=estimates["risk_free_rate"],
    #         option_type=option_type,
    #     )
    # )


    # fit a spline to the data
    fit = fit_shape_constrained_call_spline(
        chain,
        roughness_penalty=1e-1, # TODO: think about what this should be
        upper_support_multiple=2.5,
    )

    # turn the spline into an actual distribution
    distribution = construct_risk_neutral_distribution_from_spline(
        fit
    )

    score = crps_from_cdf(
        strike_grid=distribution["strike_grid"],
        cdf=distribution["cdf"],
        realized_value=realized_value,
    )

    return {
        "crps": score,
        "relative_crps": score / estimates["forward"], # the index level changes substantially from 2005 to 2024, so we want to keep a relative score
        "realized_value": realized_value,
        "density": distribution["density"],
        "cdf": distribution["cdf"],
        "strike_grid": distribution["strike_grid"],
        "forward": estimates["forward"],
        "risk_free_rate": estimates["risk_free_rate"],
        "density_mass": np.trapezoid(
            distribution["density"],
            distribution["strike_grid"],
        ),
    }


def crps_from_density_grid(
    strike_grid,
    density,
    realized_value,
    negative_tolerance=1e-10,
):
    """
    Calculate CRPS from a density evaluated on a strike grid.

    If normalize=True, the density is normalized over the supplied grid.
    This function is not intended to be used when a cdf is available.
    I may use it to demonstrate the slightly worse method of constructing
    the pdf directly by smoothing, or I may not use it at all.
    """

    strike_grid = np.asarray(strike_grid, dtype=float)
    density = np.asarray(density, dtype=float)

    if strike_grid.shape != density.shape:
        raise ValueError(
            "strike_grid and density must have the same shape."
        )

    valid = (
        np.isfinite(strike_grid)
        & np.isfinite(density)
    )
    strike_grid = strike_grid[valid]
    density = density[valid]

    order = np.argsort(strike_grid)
    strike_grid = strike_grid[order]
    density = density[order]

    if np.any(np.diff(strike_grid) <= 0):
        raise ValueError("Strike grid must be strictly increasing.")

    if density.min() < -negative_tolerance:
        raise ValueError(
            "Density contains meaningful negative values."
        )

    # Remove only negligible floating-point negatives.
    # now this should already be done
    # density = np.maximum(density, 0)

    mass = np.trapezoid(density, strike_grid)

    if not np.isclose(mass, 1, atol=0.001):
        raise ValueError(
            f"Density mass is {mass:.4f}, not approximately one."
        )

    # this is stupid; the code above constructs the cdf directly
    cdf = cumulative_trapezoid(
        density,
        strike_grid,
        initial=0,
    )
    cdf = np.clip(cdf, 0, 1)

    indicator = (
        strike_grid >= realized_value
    ).astype(float)

    crps = np.trapezoid(
        (cdf - indicator) ** 2,
        strike_grid,
    )

    # Account for a realization outside the finite support.
    if realized_value < strike_grid[0]:
        crps += strike_grid[0] - realized_value
    elif realized_value > strike_grid[-1]:
        crps += realized_value - strike_grid[-1]

    return float(crps)



def evaluate_risk_neutral_prediction_from_chain_iv_smoothing(
    chain,
    realized_value,
    option_type="call",
    normalize_density=False
):
    """
    Construct and evaluate a risk-neutral density for one chain using
    the method of smoothing on log iv space or something like that
    Point is this method shouldn't work and I hope it gives bad scores
    but probably it's going to be even better or most likely just
    like essentially the same who cares
    """

    expiry_dates = (
        chain["expiration_date"]
        .dropna()
        .drop_duplicates()
    )

    if len(expiry_dates) != 1:
        raise ValueError(
            "Expected exactly one expiration date."
        )

    # expiry_date = pd.Timestamp(
    #     expiry_dates.iloc[0]
    # ).normalize()


    estimates, _ = estimate_forward_and_rate(
        chain,
        use_spread_weights=False,
    )

    results = create_density_from_chain(
        chain
    )

    score = crps_from_density_grid(
        strike_grid=results["strike_grid"],
        density=results["density"],
        realized_value=realized_value,
    )


    return {
        "crps": score,
        "relative_crps": score / estimates["forward"], # the index level changes substantially from 2005 to 2024, so we want to keep a relative score
        "realized_value": realized_value,
        "density": results["density"],
        "strike_grid": results["strike_grid"],
        "forward": estimates["forward"],
        "risk_free_rate": estimates["risk_free_rate"],
        # "density_mass": np.trapezoid(
        #     distribution["density"],
        #     distribution["strike_grid"],
        # ),
    }



def physical_density_from_rn(fQ, s_min, s_max, n_grid=500, gamma=4):
    """
    Convert a callable risk-neutral pdf fQ(s) into a callable physical pdf fP(s)
    under a CRRA/power-utility assumption with coefficient gamma.

    Parameters
    ----------
    fQ : callable
        Function fQ(s): risk-neutral pdf (should integrate to 1 over its support).
    s_min, s_max : float
        Bounds of the support (integration domain).
    n_grid : int, optional
        Number of grid points used for numerical integration and interpolation (default 500).
    gamma : float, optional
        Coefficient of relative risk aversion (default 4).
        See Taylor's book for some reasons why I chose the value 4.
        (Basically it was calculated previously by some authors
        for S&P 500 options.)

    Returns
    -------
    fP_func : callable
        Function fP(s): normalized physical (real-world) pdf.
    s_grid : ndarray
        Grid used for normalization and interpolation.
    fP_grid : ndarray
        Values of fP(s) on that grid.

    Notes
    -----
    f_P(s) = (s^gamma / E_Q[S_T^gamma]) * f_Q(s)
    
    """
    # evaluate fQ on a grid
    s_grid = np.linspace(s_min, s_max, n_grid)
    fQ_grid = fQ(s_grid)
    fQ_grid = np.clip(fQ_grid, 0, None)  # remove tiny negatives

    # compute E_Q[S_T^gamma]
    Z = np.trapezoid((s_grid ** gamma) * fQ_grid, s_grid)

    fP_grid = (s_grid ** gamma) * fQ_grid / Z

    # normalize
    norm = np.trapezoid(fP_grid, s_grid)
    if norm > 0:
        fP_grid /= norm

    fP_func = CubicSpline(s_grid, fP_grid, extrapolate=False)

    return fP_func, s_grid, fP_grid

# I think we prefer this version since we're using the cached densities
def physical_density_from_rn_grid(
    strike_grid,
    fQ,
    gamma,
):
    strike_grid = np.asarray(strike_grid, dtype=float)
    fQ = np.asarray(fQ, dtype=float)

    # Scaling by a constant does not change the normalized density,
    # but helps with numerical stability.
    reference_price = np.median(strike_grid)

    weights = (
        strike_grid / reference_price
    ) ** gamma

    unnormalized_fP = weights * fQ

    Z = np.trapezoid(
        unnormalized_fP,
        strike_grid,
    )

    if not np.isfinite(Z) or Z <= 0:
        raise ValueError(
            "Could not normalize physical density."
        )

    fP = unnormalized_fP / Z

    return fP


# Objective function to minimize with respect to gamma
def average_relative_crps_for_gamma(
    gamma,
    cached_predictions,
):
    scores = []

    for obs in cached_predictions:

        strike_grid = obs["strike_grid"]
        fQ = obs["fQ"]
        realized_value = obs["realized_value"]

        fP = physical_density_from_rn_grid(
            strike_grid=strike_grid,
            fQ=fQ,
            gamma=gamma,
        )

        score = crps_from_density_grid(
            strike_grid=strike_grid,
            density=fP,
            realized_value=realized_value,
        )

        scores.append(score)

    return np.mean(scores)
