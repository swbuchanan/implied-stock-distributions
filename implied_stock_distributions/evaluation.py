"""Scoring the recovered distributions, and the risk-neutral to physical map.

Everything is scored through one function, ``crps_from_cdf``. The
shape-constrained fit produces its CDF analytically as ``1 + g'(K / F)``,
and the implied-volatility baseline produces one by integrating its
density, but from there the two are treated identically -- the same
integration scheme, the same handling of realisations outside the support,
and the same normalisation by the forward.

The continuous ranked probability score for a forecast CDF ``F`` and an
outcome ``y`` is

    CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx,

in the units of the forecast variable. Scores are also reported relative to
the forward, because the index level rises roughly fivefold across the
sample; using the *realised* level as the denominator instead would put the
outcome being predicted into the denominator of its own error measure.

Quadrature
----------
The integrand jumps by one at the realised value. Applying the trapezoid
rule across that jump is ``O(h)`` accurate, not ``O(h^2)``, and the error is
one-signed: it *overstates* the score, by an amount that shrinks as the grid
is refined near the outcome. That made the score depend on how each method
happened to tabulate its CDF -- the constrained fit merges the spline
breakpoints (one per traded strike) into its grid, the baseline used a plain
``linspace`` -- which is a difference in quadrature masquerading as a
difference in method.

``crps_from_cdf`` now inserts the outcome as a grid point and integrates the
piecewise-linear interpolant cell by cell in closed form. On a cell of width
``h`` whose endpoint values of ``F - 1{x >= y}`` are ``a`` and ``b``, the
integrand is a quadratic and

    integral = h (a^2 + a b + b^2) / 3.

This is exact for the piecewise-linear interpolant of the tabulated CDF --
which is the same object ``probability_integral_transform`` evaluates with
``np.interp`` -- so the score and the PIT now describe one distribution
rather than two slightly different ones. It removes the bias, removes most
of the per-observation noise, and makes the score invariant to where the
grid points fall. Use ``common_strike_grid`` so that competing methods are
tabulated identically as well.
"""

from __future__ import annotations

import numpy as np

from .config import CRPS_MASS_TOLERANCE, DENSITY_GRID_POINTS, UPPER_SUPPORT_MULTIPLE
from .errors import ChainQualityError

__all__ = [
    "cdf_from_density",
    "common_strike_grid",
    "crps_from_cdf",
    "crps_from_density_grid",
    "probability_integral_transform",
    "evaluate_distribution",
    "distribution_moments",
    "recentre_on_forward",
    "physical_density_from_rn_grid",
    "tilted_distribution",
    "relative_crps_for_gamma",
    "mean_relative_crps_for_gamma",
    "relative_crps_curve",
    "gamma_plausible_interval",
    "pit_for_gamma",
    "implied_equity_risk_premium",
    "lognormal_tilt_parameters",
    "crps_location_sensitivity",
    "tilt_moment_summary",
]


def _clean_grid(strike_grid, values):
    """Sort, drop non-finite points and check the grid is strictly increasing."""
    strike_grid = np.asarray(strike_grid, dtype=float)
    values = np.asarray(values, dtype=float)

    if strike_grid.shape != values.shape:
        raise ValueError("Grid and values must have the same shape.")

    keep = np.isfinite(strike_grid) & np.isfinite(values)
    strike_grid, values = strike_grid[keep], values[keep]

    order = np.argsort(strike_grid, kind="stable")
    strike_grid, values = strike_grid[order], values[order]

    # The fitted grid merges a regular linspace with the spline breakpoints,
    # so coincident abscissae are possible -- and certain if the grid has
    # been round-tripped through a narrower float type. Collapse them rather
    # than failing: a repeated point carries no information, and a zero-width
    # interval would silently drop out of the sums anyway.
    if strike_grid.size > 1:
        distinct = np.concatenate([[True], np.diff(strike_grid) > 0])
        strike_grid, values = strike_grid[distinct], values[distinct]

    if strike_grid.size < 2:
        raise ValueError("Strike grid must have at least two distinct points.")

    return strike_grid, values


def common_strike_grid(
    forward: float,
    upper_support_multiple: float = UPPER_SUPPORT_MULTIPLE,
    n_grid: int = DENSITY_GRID_POINTS,
    extra_points=(),
    lower_fraction: float = 1e-4,
    min_relative_spacing: float = 1e-6,
):
    """One strike grid for every method scored on a given chain.

    Two forecasts that differ only in how finely they happen to tabulate
    their CDFs are not two forecasts, and before the quadrature fix in
    ``crps_from_cdf`` that difference was worth about half a per cent of the
    score -- in favour of whichever method merged more points into its grid.
    Building the grid once, here, and handing it to both methods and to
    every benchmark removes the question entirely.

    ``extra_points`` should carry the traded strikes and the spline
    breakpoints, so that neither method is evaluated on a grid that resolves
    its own kinks better than its competitor's.

    Points closer together than ``min_relative_spacing * forward`` are
    collapsed. Merging a linspace with a set of breakpoints otherwise
    produces pairs of abscissae a floating-point hair apart, and anything
    that differentiates through such a grid -- the baseline's interpolating
    cubic spline, in particular -- divides by that hair.
    """
    forward = float(forward)
    if not np.isfinite(forward) or forward <= 0:
        raise ValueError("Forward must be finite and positive.")

    upper = float(upper_support_multiple) * forward
    grid = np.linspace(lower_fraction * forward, upper, int(n_grid))

    extra = np.asarray(list(extra_points), dtype=float).ravel()
    extra = extra[np.isfinite(extra) & (extra > 0) & (extra <= upper)]

    merged = np.unique(np.concatenate([grid, extra]))

    tolerance = float(min_relative_spacing) * forward
    if tolerance > 0 and merged.size > 1:
        keep = np.concatenate([[True], np.diff(merged) > tolerance])
        merged = merged[keep]

    return merged


def cdf_from_density(strike_grid, density):
    """Cumulative trapezoidal integral of a density, clipped to [0, 1]."""
    strike_grid, density = _clean_grid(strike_grid, density)

    if np.min(density) < -1e-8:
        raise ChainQualityError("Density contains meaningful negative values.")

    density = np.maximum(density, 0.0)

    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(strike_grid)
    )])

    return strike_grid, np.clip(cdf, 0.0, 1.0)


def crps_from_cdf(strike_grid, cdf, realized_value, mass_tolerance=CRPS_MASS_TOLERANCE):
    """CRPS of a forecast CDF tabulated on a grid, integrated exactly.

    The outcome is inserted as a grid point so that the indicator is
    constant on every cell, and each cell then contributes the closed-form
    integral of a quadratic. See the module docstring for why the trapezoid
    rule is not good enough here.

    A realisation outside the tabulated support is handled analytically
    rather than rejected: below the grid the integrand is one over the
    interval between the outcome and the first grid point, and above it the
    same holds at the other end.
    """
    strike_grid, cdf = _clean_grid(strike_grid, cdf)
    realized_value = float(realized_value)

    if not np.isfinite(realized_value):
        raise ValueError("The realised value must be finite.")

    if not (
        abs(cdf[0]) <= mass_tolerance
        and abs(cdf[-1] - 1.0) <= mass_tolerance
    ):
        raise ChainQualityError(
            f"CDF runs from {cdf[0]:.6f} to {cdf[-1]:.6f}; expected 0 to 1 "
            f"within {mass_tolerance}."
        )

    score = 0.0

    # Mass the grid does not cover, where the squared difference is one.
    # Clamping the outcome into the support afterwards is exact: below the
    # grid the indicator is one throughout it, above the grid it is zero.
    if realized_value < strike_grid[0]:
        score += strike_grid[0] - realized_value
        outcome = strike_grid[0]
    elif realized_value > strike_grid[-1]:
        score += realized_value - strike_grid[-1]
        outcome = strike_grid[-1]
    else:
        outcome = realized_value

    # Insert the outcome as a knot, so that no cell straddles the jump.
    position = int(np.searchsorted(strike_grid, outcome))
    if 0 < position < strike_grid.size and strike_grid[position] != outcome:
        value_at_outcome = float(np.interp(outcome, strike_grid, cdf))
        strike_grid = np.insert(strike_grid, position, outcome)
        cdf = np.insert(cdf, position, value_at_outcome)

    # On the interior of cell [x_i, x_i+1] the indicator equals 1 exactly
    # when x_i >= outcome, because the outcome is now a grid point.
    indicator = (strike_grid[:-1] >= outcome).astype(float)

    left = cdf[:-1] - indicator
    right = cdf[1:] - indicator
    widths = np.diff(strike_grid)

    # integral over a cell of (a + (b - a) t / h)^2 dt = h (a^2 + ab + b^2) / 3
    score += float(np.sum(
        widths * (left * left + left * right + right * right) / 3.0
    ))

    return score


def crps_from_density_grid(strike_grid, density, realized_value, **kwargs):
    """CRPS of a density tabulated on a grid, via its cumulative integral."""
    strike_grid, cdf = cdf_from_density(strike_grid, density)
    return crps_from_cdf(strike_grid, cdf, realized_value, **kwargs)


def probability_integral_transform(strike_grid, cdf, realized_value):
    """Return ``F(y)``: the forecast CDF evaluated at the outcome.

    Pooled across forecasts these values are uniform on [0, 1] exactly when
    the forecast distributions are calibrated, which makes their histogram
    the most direct diagnostic available here.

    The linear interpolation here is the same interpolant ``crps_from_cdf``
    integrates, so the two diagnostics describe one distribution.
    """
    strike_grid, cdf = _clean_grid(strike_grid, cdf)
    return float(
        np.interp(float(realized_value), strike_grid, cdf, left=0.0, right=1.0)
    )


def distribution_moments(strike_grid, density, support=None):
    """Mean, standard deviation and log-return moments of a tabulated density.

    The log-return moments are taken under the forecast distribution of
    ``log(S_T)``, which is the object the CRRA tilt acts on additively and
    the object Cboe's SKEW index describes.

    ``support`` restricts every moment to ``low <= S_T <= high`` (the density
    is renormalised there). Pass the traded strike range when the higher
    log moments matter: ``log(S_T / F)`` is about -2 at ``0.1 F``, so one per
    cent of extrapolated mass far below the lowest strike is enough to move
    the log-skewness by several units -- the tail model, not the prices,
    would then decide the answer. Cboe's SKEW is likewise computed from the
    traded strikes only.
    """
    strike_grid, density = _clean_grid(strike_grid, density)

    if support is not None:
        low, high = support
        inside = (strike_grid >= low) & (strike_grid <= high)
        if inside.sum() < 2:
            raise ChainQualityError("The support holds fewer than two grid points.")
        strike_grid, density = strike_grid[inside], density[inside]

    mass = float(np.trapezoid(density, strike_grid))
    if not np.isfinite(mass) or mass <= 0:
        raise ChainQualityError("Density does not integrate to a positive mass.")

    density = density / mass

    mean = float(np.trapezoid(strike_grid * density, strike_grid))
    variance = float(
        np.trapezoid((strike_grid - mean) ** 2 * density, strike_grid)
    )

    # Moments of log(S_T). The grid can start at zero, where the logarithm
    # is undefined; the density there is negligible, so the positive part
    # is renormalised rather than the whole thing being abandoned.
    positive = strike_grid > 0
    levels = strike_grid[positive]
    log_levels = np.log(levels)
    positive_density = density[positive]
    positive_mass = float(np.trapezoid(positive_density, levels))

    def log_moment(order, centre=0.0):
        return float(np.trapezoid(
            (log_levels - centre) ** order * positive_density, levels
        ) / positive_mass)

    log_mean = log_moment(1)
    log_variance = log_moment(2, log_mean)
    log_third = log_moment(3, log_mean)

    log_sd = float(np.sqrt(max(log_variance, 0.0)))
    log_skewness = log_third / log_sd**3 if log_sd > 0 else float("nan")

    return {
        "mass": mass,
        "mean": mean,
        "standard_deviation": float(np.sqrt(max(variance, 0.0))),
        "log_mean": log_mean,
        "log_standard_deviation": log_sd,
        "log_skewness": log_skewness,
    }


def recentre_on_forward(strike_grid, density, forward, method: str = "scale"):
    """Move a density's mean onto the forward, leaving its shape alone.

    The shape-constrained fit satisfies ``E_Q[S_T] = F`` as an identity. The
    implied-volatility baseline does not: clipping the negative regions,
    renormalising and truncating at ``x_upper * F`` all move the mean, and
    CRPS is very sensitive to location -- at thirty days a tenth of a per
    cent of error in the mean is a few index points against a score of
    around a hundred.

    Scoring this recentred density as a third arm splits the comparison
    into a location part and a shape part. If the constrained fit's
    advantage is entirely location, that is a sharper result than the one
    the raw comparison suggests, and a more interesting one: it says the
    value of the constrained fit is that it inherits the parity forward.

    ``method`` is ``"scale"`` (multiply the support by ``F / mean``, which
    keeps the support positive and holds the distribution of *returns*
    fixed) or ``"shift"`` (add ``F - mean``, which moves the mean while
    holding the standard deviation in index points fixed).
    """
    strike_grid, density = _clean_grid(strike_grid, density)
    forward = float(forward)

    moments = distribution_moments(strike_grid, density)
    mean = moments["mean"]

    if not np.isfinite(mean) or mean <= 0:
        raise ChainQualityError("Cannot recentre a density with a non-positive mean.")

    if method == "scale":
        factor = forward / mean
        new_grid = strike_grid * factor
        # A density transforms with the Jacobian of the change of variable.
        new_density = density / factor
    elif method == "shift":
        new_grid = strike_grid + (forward - mean)
        new_density = density.copy()
    else:
        raise ValueError("method must be 'scale' or 'shift'.")

    return new_grid, new_density


def evaluate_distribution(strike_grid, cdf, realized_value, forward, density=None):
    """Score one distribution against one outcome."""
    score = crps_from_cdf(strike_grid, cdf, realized_value)

    result = {
        "crps": score,
        "relative_crps": score / float(forward),
        "pit": probability_integral_transform(strike_grid, cdf, realized_value),
        "realized_value": float(realized_value),
        "forward": float(forward),
    }

    if density is not None:
        moments = distribution_moments(strike_grid, density)
        result.update(
            mean=moments["mean"],
            mean_over_forward=moments["mean"] / float(forward),
            log_standard_deviation=moments["log_standard_deviation"],
            log_skewness=moments["log_skewness"],
        )

    return result


# --------------------------------------------------------------------------
# Risk-neutral to physical
# --------------------------------------------------------------------------

def physical_density_from_rn_grid(
    strike_grid,
    risk_neutral_density,
    gamma,
    tilt_range=None,
):
    """Tilt a risk-neutral density into a physical one under CRRA utility.

    With power utility and terminal wealth proportional to the index, the
    pricing kernel is ``m(S) proportional to S^-gamma``, so

        f_P(s) = s^gamma f_Q(s) / integral y^gamma f_Q(y) dy.

    Parameters
    ----------
    tilt_range : tuple[float, float], optional
        Hold the kernel flat outside this strike range: the weight is
        ``s^gamma`` inside it and stays at its boundary value beyond either
        end. The weight ``s^gamma`` is 39 times larger at ``2.5 F`` than at
        ``F`` when ``gamma`` is four, so an unrestricted tilt is partly
        acting on extrapolated tail mass rather than on anything a price
        implies. Supplying the traded-strike range is the robustness check.
    """
    strike_grid = np.asarray(strike_grid, dtype=float)
    density = np.asarray(risk_neutral_density, dtype=float)

    # Holding the kernel at its boundary value keeps the tilted density
    # continuous. (Setting the weight to one outside the range would make
    # the relative tilt between the inside and the outside depend on the
    # arbitrary normalising constant below.)
    levels = strike_grid if tilt_range is None else np.clip(strike_grid, *tilt_range)

    # Scaling by a constant leaves the normalised density unchanged but
    # keeps s**gamma away from overflow at SPX levels.
    reference = float(np.median(strike_grid))
    weights = (levels / reference) ** float(gamma)

    unnormalized = weights * density
    total = float(np.trapezoid(unnormalized, strike_grid))

    if not np.isfinite(total) or total <= 0:
        raise ChainQualityError("Could not normalise the physical density.")

    return unnormalized / total


def tilted_distribution(observation, gamma, tilt_range=None):
    """Apply the CRRA tilt to one cached prediction.

    ``tilt_range`` may be ``"traded"``, which uses the observation's own
    traded strike range, an explicit ``(low, high)`` pair, or ``None`` for
    the full support.

    Returns the strike grid, the tilted density and its CDF.
    """
    grid = np.asarray(observation["strike_grid"], dtype=float)
    risk_neutral = np.asarray(observation["fQ"], dtype=float)

    if tilt_range == "traded":
        tilt_range = observation.get("traded_range")

    density = physical_density_from_rn_grid(
        grid, risk_neutral, gamma, tilt_range=tilt_range
    )

    grid, cdf = cdf_from_density(grid, density)
    return grid, density, cdf


def relative_crps_for_gamma(gamma, cached_predictions, tilt_range=None):
    """Relative CRPS of each tilted density, as an array.

    Scores are divided by each observation's forward. Averaging *absolute*
    CRPS instead -- as an earlier version did -- weights each observation by
    its index level, and with SPX ranging from roughly 700 to 6000 across
    the sample that gives the recent years several times the weight of
    2008, so the resulting estimate of gamma describes the recent subsample
    rather than the whole history.
    """
    if not cached_predictions:
        raise ValueError("No cached predictions were supplied.")

    scores = np.empty(len(cached_predictions))

    for index, observation in enumerate(cached_predictions):
        grid, _, cdf = tilted_distribution(observation, gamma, tilt_range)
        scores[index] = (
            crps_from_cdf(grid, cdf, observation["realized_value"])
            / float(observation["forward"])
        )

    return scores


def mean_relative_crps_for_gamma(gamma, cached_predictions, tilt_range=None):
    """Mean relative CRPS of the tilted densities, as a function of gamma."""
    return float(
        np.mean(relative_crps_for_gamma(gamma, cached_predictions, tilt_range))
    )


def relative_crps_curve(gamma_grid, cached_predictions, tilt_range=None):
    """Per-observation scores over a grid of gamma, as one matrix.

    Returns an array of shape ``(len(gamma_grid), len(cached_predictions))``.
    Computing it once is what makes the paired-difference interval below
    affordable: the alternative re-scores every observation for every gamma
    twice over.
    """
    gamma_grid = np.asarray(gamma_grid, dtype=float)

    return np.vstack([
        relative_crps_for_gamma(gamma, cached_predictions, tilt_range)
        for gamma in gamma_grid
    ])


def gamma_plausible_interval(
    gamma_grid,
    score_matrix,
    blocks,
    confidence: float = 0.95,
    replicates: int | None = None,
    random_state: int | None = None,
):
    """Which values of gamma the data cannot distinguish from the best one.

    The quantity with a standard error is the *paired difference*

        D(gamma) = L(gamma) - L(gamma-hat),

    averaged over a common set of chains. Chain-to-chain variation in the
    level of the score -- which is enormous, since it is dominated by how
    far each expiration's outcome happened to land from its forward --
    cancels in the difference. Using the standard error of the *level*
    instead, as an earlier version did, overstates the width by one to two
    orders of magnitude, and that width was then cited as evidence that
    gamma is weakly identified. It is weakly identified (see
    ``crps_location_sensitivity`` for why, structurally), but not by
    anything like that margin.

    Parameters
    ----------
    gamma_grid : array_like
        The grid the score matrix was evaluated on.
    score_matrix : array_like
        ``(n_gamma, n_observations)`` from ``relative_crps_curve``.
    blocks : array_like
        Block label per observation -- the expiration month -- so that the
        bootstrap resamples whole expirations rather than chains.

    Returns
    -------
    dict
        ``gamma_hat``, the interval endpoints, the paired standard error at
        each gamma, and the mean difference curve.
    """
    from .stats import block_bootstrap_mean

    gamma_grid = np.asarray(gamma_grid, dtype=float)
    score_matrix = np.asarray(score_matrix, dtype=float)

    if score_matrix.shape[0] != gamma_grid.size:
        raise ValueError("score_matrix must have one row per gamma.")

    curve = score_matrix.mean(axis=1)
    best = int(np.argmin(curve))
    gamma_hat = float(gamma_grid[best])

    bootstrap_kwargs = {"confidence": confidence}
    if replicates is not None:
        bootstrap_kwargs["replicates"] = replicates
    if random_state is not None:
        bootstrap_kwargs["random_state"] = random_state

    differences = score_matrix - score_matrix[best]

    standard_errors = np.empty(gamma_grid.size)
    lower = np.empty(gamma_grid.size)
    upper = np.empty(gamma_grid.size)

    for index in range(gamma_grid.size):
        if index == best:
            standard_errors[index] = 0.0
            lower[index] = upper[index] = 0.0
            continue

        result = block_bootstrap_mean(
            differences[index], blocks, **bootstrap_kwargs
        )
        standard_errors[index] = result["standard_error"]
        lower[index] = result["lower"]
        upper[index] = result["upper"]

    mean_difference = differences.mean(axis=1)

    # gamma is not rejected when its loss is not significantly worse than
    # the best one -- that is, when the interval for the paired difference
    # still reaches zero.
    plausible = lower <= 0.0
    plausible_gammas = gamma_grid[plausible]

    # A one-standard-error rule on the same paired differences, for the
    # readers who expect that convention rather than a bootstrap interval.
    within_one_se = mean_difference <= standard_errors
    one_se_gammas = gamma_grid[within_one_se]

    return {
        "gamma_hat": gamma_hat,
        "curve": curve,
        "mean_difference": mean_difference,
        "standard_error": standard_errors,
        "lower": lower,
        "upper": upper,
        "plausible_lower": (
            float(plausible_gammas.min()) if plausible_gammas.size else np.nan
        ),
        "plausible_upper": (
            float(plausible_gammas.max()) if plausible_gammas.size else np.nan
        ),
        "one_se_lower": float(one_se_gammas.min()) if one_se_gammas.size else np.nan,
        "one_se_upper": float(one_se_gammas.max()) if one_se_gammas.size else np.nan,
        "confidence": confidence,
        "n_observations": int(score_matrix.shape[1]),
    }


def pit_for_gamma(gamma, cached_predictions, tilt_range=None):
    """Probability integral transforms of the tilted densities."""
    return np.array([
        probability_integral_transform(
            *tilted_distribution(observation, gamma, tilt_range)[::2],
            observation["realized_value"],
        )
        for observation in cached_predictions
    ])


# --------------------------------------------------------------------------
# What the CRRA tilt actually does
# --------------------------------------------------------------------------

def implied_equity_risk_premium(gamma, volatility):
    """Annualised equity risk premium implied by a CRRA tilt: ``gamma sigma^2``.

    For a lognormal ``f_Q`` the tilt is *exactly* an exponential tilt in log
    space. Writing ``log S_T ~ N(m, v)`` under ``Q``, the density
    ``s^gamma f_Q(s)`` renormalised is ``N(m + gamma v, v)``: the
    log-variance is unchanged and the mean of ``S_T`` is multiplied by
    ``exp(gamma v)``. With ``v = sigma^2 tau`` the expected excess log
    growth rate is ``gamma sigma^2`` per year, independently of the horizon.

    This is the first sanity check any reader will apply to an estimate of
    gamma, and it is one line of code. At ``sigma = 18%``, ``gamma = 3``
    implies 9.7% a year; at ``sigma = 22%``, ``gamma = 5`` implies 24%.
    Estimates that imply an implausible premium are not estimates of risk
    aversion, whatever the objective says.
    """
    return np.asarray(gamma, dtype=float) * np.asarray(volatility, dtype=float) ** 2


def lognormal_tilt_parameters(mean_log, variance_log, gamma):
    """Parameters of the tilted lognormal: ``(m + gamma v, v)``.

    Exposed so the identity behind ``implied_equity_risk_premium`` can be
    checked numerically against ``physical_density_from_rn_grid`` rather
    than asserted.
    """
    return (
        float(mean_log) + float(gamma) * float(variance_log),
        float(variance_log),
    )


def crps_location_sensitivity(volatility, time_to_expiry, gammas=(1, 2, 3, 4)):
    """How much of the CRPS a CRRA tilt can possibly move.

    CRPS is a proper scoring rule, so minimising it identifies the true
    distribution given enough data -- but "enough" depends on how strongly
    the score responds. For a calibrated Gaussian forecast the expected
    CRPS is ``sigma_T / sqrt(pi)``; displacing the mean by ``delta``
    while the outcome distribution stays put raises it to

        sigma_T [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]  integrated
        over the outcome distribution,

    which this evaluates numerically. The tilt displaces the mean by
    ``gamma sigma^2 tau``, so the entire gamma signal in the mean score is
    of order one per cent -- the same order as the quadrature artefact the
    module docstring describes. A flat objective is a structural property
    of using CRPS to estimate a location parameter, not a shortage of data,
    and saying so turns a null result into an argument.

    Returns a list of ``(gamma, relative_increase)`` pairs.
    """
    from scipy import stats

    sigma_t = float(volatility) * np.sqrt(float(time_to_expiry))

    # Expected CRPS of a Gaussian forecast displaced by delta, when the
    # outcome is drawn from the undisplaced Gaussian. Integrate over the
    # outcome distribution on a wide grid.
    outcomes = np.linspace(-8.0, 8.0, 4001) * sigma_t
    weights = stats.norm.pdf(outcomes, loc=0.0, scale=sigma_t)
    weights = weights / np.trapezoid(weights, outcomes)

    def expected_crps(displacement):
        z = (outcomes - displacement) / sigma_t
        pointwise = sigma_t * (
            z * (2.0 * stats.norm.cdf(z) - 1.0)
            + 2.0 * stats.norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        return float(np.trapezoid(pointwise * weights, outcomes))

    baseline = expected_crps(0.0)

    rows = []
    for gamma in gammas:
        displacement = float(gamma) * float(volatility) ** 2 * float(time_to_expiry)
        rows.append((float(gamma), expected_crps(displacement) / baseline - 1.0))

    return rows


def tilt_moment_summary(
    strike_grid, density, gammas=(1, 2, 3, 4), tilt_range=None, support=None
):
    """What the tilt does to a density's location, width and skew.

    On a lognormal the tilt is pure location. On a realistically
    left-skewed ``f_Q`` it is not: it lifts the mean *and* narrows the
    distribution *and* cuts the left skew, all tied to the one parameter.
    That is the substantive limitation of a one-parameter CRRA kernel --
    it has one knob for two problems, the equity premium and the variance
    risk premium -- and it is the argument for wanting a second parameter
    rather than for wanting a longer sample.

    ``support`` is passed to ``distribution_moments``; give it the traded
    strike range, or the log-skewness describes the extrapolated tail.

    Returns a list of dicts, one per gamma, with the moments and their
    change from the untilted density.
    """
    strike_grid, density = _clean_grid(strike_grid, density)
    base = distribution_moments(strike_grid, density, support=support)

    rows = []
    for gamma in (0.0, *gammas):
        tilted = physical_density_from_rn_grid(
            strike_grid, density, gamma, tilt_range=tilt_range
        )
        moments = distribution_moments(strike_grid, tilted, support=support)
        rows.append({
            "gamma": float(gamma),
            "mean": moments["mean"],
            "mean_change": moments["mean"] / base["mean"] - 1.0,
            "log_standard_deviation": moments["log_standard_deviation"],
            "log_sd_change": (
                moments["log_standard_deviation"] / base["log_standard_deviation"] - 1.0
                if base["log_standard_deviation"] > 0 else float("nan")
            ),
            "log_skewness": moments["log_skewness"],
        })

    return rows
