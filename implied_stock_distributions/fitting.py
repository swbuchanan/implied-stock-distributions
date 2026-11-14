"""Shape-constrained (Fengler-style) smoothing of the call-price curve.

The curve is fitted in normalised coordinates

    x = K / F,        g(x) = C(K) / (D * F),

where ``F`` is the forward and ``D`` the discount factor. In these
coordinates the Breeden-Litzenberger relation and the no-arbitrage shape
restrictions are all linear or convex in the spline coefficients:

    f_Q(K) = g''(K / F) / F,       F_Q(K) = 1 + g'(K / F).

The program minimises a spread-standardised pricing error plus Fengler's
roughness penalty, subject to constraints that make the fitted curve a
complete probability distribution on ``0 <= S_T <= x_upper * F``:

    g(0) = 1,   g'(0) = -1,   g(x_upper) = 0,   g'(x_upper) = 0,
    g'' >= 0 everywhere,   -1 <= g' <= 0,   g >= max(1 - x, 0).

Two consequences of that boundary set are worth stating, because they are
free correctness checks on the whole pipeline:

* ``integral f_Q = g'(x_upper) - g'(0) = 1`` exactly, so the fitted density
  is normalised by construction and does not need rescaling afterwards.
* ``E_Q[S_T] = F * (g(0) - g(x_upper)) = F`` exactly, so the fitted
  distribution is automatically consistent with the forward implied by
  put-call parity.

Knots and data are separable
----------------------------
``knot_x`` sets the spline's breakpoints and ``observed_x`` sets the data
the curve is fitted to; by default they coincide, since Fengler puts a knot
at every observed strike. They must be separable for cross-validation to
mean anything: holding out every fifth strike while letting the knot vector
follow the training set removes a fifth of the basis functions too, so what
is being compared across folds is not one model at different penalties but
different function spaces. ``selection.cross_validate_chain`` therefore
fixes ``knot_x`` from the full strike set and holds out only the data.

Reference
---------
Fengler, M. R. (2009). "Arbitrage-Free Smoothing of the Implied Volatility
Surface." *Quantitative Finance*, 9(4), 417-428.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .config import (
    DENSITY_GRID_POINTS,
    MIN_ERROR_ABSOLUTE,
    MIN_ERROR_FRACTION_OF_FORWARD,
    MIN_QUOTES_FOR_SPLINE,
    N_TAIL_KNOTS,
    ROUGHNESS_PENALTY,
    UPPER_SUPPORT_MULTIPLE,
)
from .errors import ChainQualityError, InsufficientQuotesError, SolverError
from .forwards import estimate_forward_and_rate

__all__ = [
    "select_call_equivalent_quotes",
    "pricing_error_scale",
    "build_call_spline_program",
    "constraint_bounds",
    "solve_call_spline_program",
    "fit_shape_constrained_call_spline",
    "construct_risk_neutral_distribution_from_spline",
]

DEGREE = 3


# --------------------------------------------------------------------------
# Quote selection
# --------------------------------------------------------------------------

def select_call_equivalent_quotes(
    chain: pd.DataFrame,
    forward: float,
    discount_factor: float,
) -> pd.DataFrame:
    """Return one out-of-the-money quote per strike, as a call price.

    Out-of-the-money puts below the forward and out-of-the-money calls at
    or above it are the liquid side of each strike. Puts are converted with
    ``C = P + D (F - K)``; the parity term is linear in strike and so has
    no effect on the second derivative that becomes the density.

    The same selection is used by the implied-volatility baseline, so that
    the two methods see identical information.
    """
    quotes = chain.copy()

    if "quote_is_two_sided" in quotes.columns:
        quotes = quotes.loc[quotes["quote_is_two_sided"].fillna(False)].copy()
    else:
        quotes = quotes.loc[
            (quotes["bid_price"] > 0)
            & (quotes["ask_price"] >= quotes["bid_price"])
        ].copy()

    if "mid_price" not in quotes.columns:
        quotes["mid_price"] = (quotes["bid_price"] + quotes["ask_price"]) / 2

    quotes["spread"] = quotes["ask_price"] - quotes["bid_price"]

    out_of_the_money = (
        quotes["put_call"].eq("put") & quotes["strike_price"].lt(forward)
    ) | (
        quotes["put_call"].eq("call") & quotes["strike_price"].ge(forward)
    )

    quotes = quotes.loc[out_of_the_money].copy()

    quotes["call_equivalent_price"] = np.where(
        quotes["put_call"].eq("call"),
        quotes["mid_price"],
        quotes["mid_price"]
        + discount_factor * (forward - quotes["strike_price"]),
    )

    values = quotes[
        ["strike_price", "call_equivalent_price", "spread"]
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0)
        & (values[:, 1] >= 0)
        & (values[:, 2] >= 0)
    )

    quotes = (
        quotes.loc[valid]
        .sort_values("strike_price")
        .reset_index(drop=True)
    )

    if quotes["strike_price"].duplicated().any():
        raise ChainQualityError(
            "Duplicate selected strikes found. This usually means quotes "
            "from two settlement conventions were merged into one chain."
        )

    return quotes


def pricing_error_scale(
    spread: np.ndarray,
    forward: float,
    discount_factor: float,
) -> np.ndarray:
    """Assumed noise on each normalised quote, from its half-spread.

    The floor is expressed as a fraction of the forward rather than in
    index points, so that it means the same thing at SPX 1200 and SPX 6000.
    An absolute floor covers the minimum tick.
    """
    half_spread = np.asarray(spread, dtype=float) / 2.0

    floor = max(
        MIN_ERROR_FRACTION_OF_FORWARD * float(forward),
        MIN_ERROR_ABSOLUTE,
    )

    return np.maximum(half_spread, floor) / (discount_factor * forward)


# --------------------------------------------------------------------------
# Convex program
# --------------------------------------------------------------------------

def build_call_spline_program(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    error_scale: np.ndarray,
    x_upper: float,
    n_tail_knots: int = N_TAIL_KNOTS,
    knot_x: np.ndarray | None = None,
) -> dict:
    """Assemble the matrices of the shape-constrained smoothing problem.

    Construction is kept separate from solving so that the spline basis,
    the roughness quadrature and the constraint set can be checked on their
    own in the test suite.

    Parameters
    ----------
    observed_x, observed_y, error_scale : array_like
        The data the curve is fitted to, in normalised coordinates.
    x_upper : float
        Upper end of the support, in units of the forward.
    n_tail_knots : int
        Knots placed in each unobserved tail.
    knot_x : array_like, optional
        Strikes at which to place interior knots. Defaults to
        ``observed_x``, which is Fengler's construction. Supply the full
        strike set when fitting to a subset, so that holding data out does
        not also change the function space -- see the module docstring.

    Returns
    -------
    dict
        ``basis_matrix`` (observations by coefficients), ``roughness_matrix``
        (the Gram matrix of the second-derivative quadrature), the constraint
        blocks, the knot vector and the breakpoints.
    """
    observed_x = np.asarray(observed_x, dtype=float)
    observed_y = np.asarray(observed_y, dtype=float)
    error_scale = np.asarray(error_scale, dtype=float)

    if not (observed_x.shape == observed_y.shape == error_scale.shape):
        raise ValueError("Observations and error scales must have one shape.")

    if np.any(np.diff(observed_x) <= 0):
        raise ValueError("Observed strikes must be strictly increasing.")

    if knot_x is None:
        knot_x = observed_x
    else:
        knot_x = np.unique(np.asarray(knot_x, dtype=float))
        knot_x = knot_x[np.isfinite(knot_x) & (knot_x > 0)]
        if knot_x.size < 2:
            raise ValueError("knot_x needs at least two distinct positive values.")

    if x_upper <= max(observed_x.max(), knot_x.max()):
        raise ValueError("x_upper must exceed the largest observed or knot strike.")

    # Knots in the unobserved tails give the spline enough freedom to meet
    # the boundary conditions without distorting the traded region.
    left_tail = np.linspace(0.0, knot_x.min(), n_tail_knots + 2)[1:-1]
    right_tail = np.linspace(knot_x.max(), x_upper, n_tail_knots + 2)[1:-1]

    internal_knots = np.unique(
        np.concatenate([left_tail, knot_x, right_tail])
    )
    internal_knots = internal_knots[
        (internal_knots > 0) & (internal_knots < x_upper)
    ]

    breakpoints = np.concatenate([[0.0], internal_knots, [x_upper]])

    knots = np.concatenate([
        np.repeat(0.0, DEGREE + 1),
        internal_knots,
        np.repeat(x_upper, DEGREE + 1),
    ])

    n_coefficients = len(knots) - DEGREE - 1

    basis = BSpline(knots, np.eye(n_coefficients), DEGREE, extrapolate=False)

    basis_matrix = basis(observed_x)
    value_basis = basis(breakpoints)
    first_basis = basis.derivative(1)(breakpoints)
    second_basis = basis.derivative(2)(breakpoints)

    # g'' is linear between breakpoints, so its square is quadratic and
    # two-point Gauss-Legendre integrates it exactly on each interval.
    left, right = breakpoints[:-1], breakpoints[1:]
    midpoints = (left + right) / 2
    half_widths = (right - left) / 2
    offsets = half_widths / np.sqrt(3.0)

    quadrature_points = np.concatenate([midpoints - offsets, midpoints + offsets])
    quadrature_weights = np.concatenate([half_widths, half_widths])

    second_at_quadrature = basis.derivative(2)(quadrature_points)

    roughness_matrix = (
        second_at_quadrature.T
        @ (quadrature_weights[:, None] * second_at_quadrature)
    )
    # Symmetrise away the floating-point asymmetry.
    roughness_matrix = 0.5 * (roughness_matrix + roughness_matrix.T)

    intrinsic_values = np.maximum(1.0 - breakpoints, 0.0)

    return {
        "knots": knots,
        "degree": DEGREE,
        "breakpoints": breakpoints,
        "n_coefficients": n_coefficients,
        "observed_x": observed_x,
        "observed_y": observed_y,
        "error_scale": error_scale,
        "knot_x": knot_x,
        "basis_matrix": basis_matrix,
        "value_basis": value_basis,
        "first_basis": first_basis,
        "second_basis": second_basis,
        "roughness_matrix": roughness_matrix,
        "intrinsic_values": intrinsic_values,
        "x_upper": float(x_upper),
    }


def _solve_with_cvxpy(program, roughness_penalty):
    import cvxpy as cp

    coefficients = cp.Variable(program["n_coefficients"])

    standardized_errors = cp.multiply(
        1.0 / program["error_scale"],
        program["basis_matrix"] @ coefficients - program["observed_y"],
    )

    # Fengler's objective in sum form. Averaging the data term instead
    # would make a chain with three hundred strikes carry no more weight
    # against the penalty than one with thirty, which over-smooths the
    # later years of the sample relative to the earlier ones. The
    # consequence for cross-validation -- that a fold fitted to 80% of the
    # quotes feels an effectively larger penalty -- is handled by scaling
    # the penalty in ``selection.cross_validate_chain``.
    data_loss = cp.sum_squares(standardized_errors)
    roughness = cp.quad_form(
        coefficients, cp.psd_wrap(program["roughness_matrix"])
    )

    # The tests check this same constraint description directly.
    matrix, lower, upper = constraint_bounds(program)
    constrained = matrix @ coefficients

    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)

    constraints = [
        constrained[finite_lower] >= lower[finite_lower],
        constrained[finite_upper] <= upper[finite_upper],
    ]

    problem = cp.Problem(
        cp.Minimize(data_loss + roughness_penalty * roughness), constraints
    )
    # An OPTIMAL_INACCURATE status is recorded in the result as
    # ``solver_inaccurate``, so cvxpy's warning about it is only noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate")
        problem.solve(solver=cp.CLARABEL)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise SolverError(f"Spline optimisation failed: {problem.status}")

    return (
        np.asarray(coefficients.value).ravel(),
        str(problem.status),
        float(problem.value),
    )


def constraint_bounds(program: dict):
    """Express every shape restriction as ``lower <= A c <= upper``.

    Stacking the value, slope and curvature blocks in this form keeps the
    boundary equalities and the interior inequalities in one place, which
    is what the solver and the tests consume.
    """
    values = program["value_basis"]
    slopes = program["first_basis"]
    second = program["second_basis"]

    n = len(program["breakpoints"])
    infinity = np.inf

    value_lower = program["intrinsic_values"].astype(float).copy()
    value_upper = np.ones(n)
    value_lower[0] = value_upper[0] = 1.0      # g(0) = 1
    value_lower[-1] = value_upper[-1] = 0.0    # g(x_upper) = 0

    slope_lower = np.full(n, -1.0)
    slope_upper = np.zeros(n)
    slope_lower[0] = slope_upper[0] = -1.0     # g'(0) = -1
    slope_lower[-1] = slope_upper[-1] = 0.0    # g'(x_upper) = 0

    second_lower = np.zeros(n)                 # non-negative density
    second_upper = np.full(n, infinity)
    second_upper[0] = 0.0                      # g''(0) = 0
    second_upper[-1] = 0.0                     # g''(x_upper) = 0

    matrix = np.vstack([values, slopes, second])
    lower = np.concatenate([value_lower, slope_lower, second_lower])
    upper = np.concatenate([value_upper, slope_upper, second_upper])

    return matrix, lower, upper


def solve_call_spline_program(
    program: dict, roughness_penalty: float = ROUGHNESS_PENALTY
):
    """Solve an assembled program; return coefficients, status and objective."""
    if roughness_penalty < 0:
        raise ValueError("roughness_penalty must be non-negative.")
    return _solve_with_cvxpy(program, roughness_penalty)


# --------------------------------------------------------------------------
# Public fitting entry point
# --------------------------------------------------------------------------

def fit_shape_constrained_call_spline(
    chain: pd.DataFrame,
    forward: float | None = None,
    risk_free_rate: float | None = None,
    roughness_penalty: float = ROUGHNESS_PENALTY,
    upper_support_multiple: float = UPPER_SUPPORT_MULTIPLE,
    n_tail_knots: int = N_TAIL_KNOTS,
    quotes: pd.DataFrame | None = None,
    knot_strikes=None,
) -> dict:
    """Fit the complete shape-constrained call-price curve for one chain.

    Parameters
    ----------
    chain : pandas.DataFrame
        Quotes for one observation date and one expiration.
    forward, risk_free_rate : float, optional
        Supply both to reuse an estimate rather than recomputing it. Passing
        them is the normal path inside loops, where the parity fit has
        already been done once for the chain.
    roughness_penalty : float
        Weight on Fengler's penalty. Defaults to the value in ``config``,
        which is set by cross-validation rather than by inspection.
    upper_support_multiple : float
        The fitted distribution is supported on ``[0, this * F]``.
    quotes : pandas.DataFrame, optional
        Pre-selected quotes, used by the cross-validation routine to hold
        strikes out. Defaults to every out-of-the-money quote in the chain.
    knot_strikes : array_like, optional
        Strikes (in index points, not normalised) at which to place the
        spline's interior knots. Defaults to the strikes in ``quotes``.
        Cross-validation supplies the *full* strike set here while passing
        a training subset as ``quotes``, so that holding data out does not
        also shrink the basis.

    Returns
    -------
    dict
        The fitted ``spline`` along with the forward, discount factor,
        breakpoints, the quotes used with fitted prices and residuals, and
        solver diagnostics including ``solver_inaccurate``.
    """
    if upper_support_multiple <= 1:
        raise ValueError("upper_support_multiple must exceed 1.")
    if n_tail_knots < 0:
        raise ValueError("n_tail_knots must be non-negative.")

    if forward is None or risk_free_rate is None:
        estimates, _ = estimate_forward_and_rate(chain)
        forward = estimates["forward"] if forward is None else forward
        risk_free_rate = (
            estimates["risk_free_rate"] if risk_free_rate is None
            else risk_free_rate
        )

    forward = float(forward)
    risk_free_rate = float(risk_free_rate)

    if not np.isfinite(forward) or forward <= 0:
        raise ChainQualityError("Forward must be finite and positive.")

    time_values = chain["time_to_expiry_years"].dropna().unique()
    if len(time_values) != 1:
        raise ChainQualityError("Expected one time-to-expiry value.")

    time_to_expiry = float(time_values[0])
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ChainQualityError("Time to expiry must be finite and positive.")

    discount_factor = float(np.exp(-risk_free_rate * time_to_expiry))

    if quotes is None:
        quotes = select_call_equivalent_quotes(chain, forward, discount_factor)
    quotes = quotes.reset_index(drop=True)

    if len(quotes) < MIN_QUOTES_FOR_SPLINE:
        raise InsufficientQuotesError(
            f"{len(quotes)} usable quotes; {MIN_QUOTES_FOR_SPLINE} required."
        )

    observed_x = quotes["strike_price"].to_numpy(dtype=float) / forward
    observed_y = (
        quotes["call_equivalent_price"].to_numpy(dtype=float)
        / (discount_factor * forward)
    )
    error_scale = pricing_error_scale(
        quotes["spread"].to_numpy(dtype=float), forward, discount_factor
    )

    if knot_strikes is None:
        knot_x = observed_x
    else:
        knot_x = np.asarray(knot_strikes, dtype=float) / forward

    x_upper = max(
        upper_support_multiple,
        1.05 * observed_x.max(),
        1.05 * float(np.max(knot_x)),
    )

    program = build_call_spline_program(
        observed_x, observed_y, error_scale, x_upper, n_tail_knots,
        knot_x=knot_x,
    )

    coefficients, status, objective_value = solve_call_spline_program(
        program, roughness_penalty
    )

    spline = BSpline(
        program["knots"], coefficients, DEGREE, extrapolate=False
    )

    quotes = quotes.copy()
    quotes["fitted_call_price"] = (
        discount_factor * forward * spline(observed_x)
    )
    quotes["pricing_residual"] = (
        quotes["fitted_call_price"] - quotes["call_equivalent_price"]
    )
    quotes["standardized_residual"] = (
        quotes["pricing_residual"].to_numpy(dtype=float)
        / (error_scale * discount_factor * forward)
    )

    # Probability mass outside the traded strike range is extrapolation,
    # not measurement, so it is reported rather than buried.
    def cdf_at(x):
        return 1.0 + float(spline.derivative(1)(x))

    mass_below_traded = cdf_at(observed_x.min())
    mass_above_traded = 1.0 - cdf_at(observed_x.max())

    return {
        "spline": spline,
        "forward": forward,
        "risk_free_rate": risk_free_rate,
        "discount_factor": discount_factor,
        "time_to_expiry": time_to_expiry,
        "x_upper": program["x_upper"],
        "breakpoints": program["breakpoints"],
        "quotes": quotes,
        "n_quotes": int(len(quotes)),
        "n_knots": int(program["n_coefficients"]),
        "roughness_penalty": float(roughness_penalty),
        "strike_bounds": (
            float(observed_x.min() * forward),
            float(observed_x.max() * forward),
        ),
        "mass_below_traded": float(mass_below_traded),
        "mass_above_traded": float(mass_above_traded),
        "extrapolated_mass": float(mass_below_traded + mass_above_traded),
        "objective_value": float(objective_value),
        "solver_status": status,
        "solver_inaccurate": "inaccurate" in str(status).lower(),
    }


def construct_risk_neutral_distribution_from_spline(
    fit: dict,
    n_grid: int = DENSITY_GRID_POINTS,
    strike_grid=None,
) -> dict:
    """Evaluate call prices, the risk-neutral density and CDF from a fit.

    The returned ``mass`` and ``mean`` are diagnostics, not corrections:
    the constraint set makes them exactly one and exactly the forward, so a
    departure indicates a solver problem rather than something to normalise
    away.

    Parameters
    ----------
    strike_grid : array_like, optional
        Tabulate on this grid rather than on a linspace merged with the
        breakpoints. Pass ``evaluation.common_strike_grid(...)`` so that
        this fit and the implied-volatility baseline are scored on
        identical abscissae -- before the exact quadrature in
        ``evaluation.crps_from_cdf``, merging the breakpoints in here while
        the baseline used a plain linspace was worth about half a per cent
        of the score on its own.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    spline = fit["spline"]
    forward = fit["forward"]
    discount_factor = fit["discount_factor"]
    x_upper = fit["x_upper"]

    if strike_grid is None:
        x_grid = np.unique(np.concatenate([
            np.linspace(0.0, x_upper, n_grid),
            fit["breakpoints"],
        ]))
    else:
        strike_grid = np.asarray(strike_grid, dtype=float)
        x_grid = np.unique(strike_grid / forward)
        # The spline does not extrapolate, so clip to its support. The
        # boundary conditions make the density zero at both ends anyway.
        x_grid = x_grid[(x_grid >= 0.0) & (x_grid <= x_upper)]
        x_grid = np.unique(np.concatenate([[0.0], x_grid, [x_upper]]))

    strike_grid = forward * x_grid

    normalized_prices = spline(x_grid)
    first_derivative = spline.derivative(1)(x_grid)
    second_derivative = spline.derivative(2)(x_grid)

    call_prices = discount_factor * forward * normalized_prices

    # Breeden-Litzenberger in normalised coordinates.
    density = second_derivative / forward
    cdf = 1.0 + first_derivative

    # The solver returns tiny negative values and a CDF a hair outside
    # [0, 1]; clipping those is numerical tidying, not reshaping.
    density = np.maximum(density, 0.0)
    cdf = np.clip(cdf, 0.0, 1.0)

    return {
        "strike_grid": strike_grid,
        "call_prices": call_prices,
        "density": density,
        "cdf": cdf,
        "forward": forward,
        "mass": float(np.trapezoid(density, strike_grid)),
        "mean": float(np.trapezoid(strike_grid * density, strike_grid)),
        "minimum_density": float(np.min(second_derivative / forward)),
        "cdf_start": float(cdf[0]),
        "cdf_end": float(cdf[-1]),
        "mass_below_traded": fit.get("mass_below_traded"),
        "mass_above_traded": fit.get("mass_above_traded"),
        "strike_bounds": fit.get("strike_bounds"),
        "support": (0.0, forward * x_upper),
    }
