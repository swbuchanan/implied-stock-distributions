import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .forwards import estimate_forward_and_rate

# def fit_shape_constrained_call_spline_old(
#     chain: pd.DataFrame,
#     forward: float | None = None,
#     risk_free_rate: float | None = None,
#     roughness_penalty: float = 1e-4,
#     use_spread_weights: bool = True,
#     price_error_floor: float = 0.05,
#     n_grid: int = 1000,
# ):
#     """
#     Fit a Fengler-style arbitrage-free natural cubic spline to one option
#     expiration.

#     The function constructs a smooth call-price curve as a function of
#     forward moneyness. It fits the curve directly in option-price space,
#     where the absence of static strike arbitrage can be expressed using
#     linear constraints on the spline.

#     The input should contain quotes from one observation date and one exact
#     expiration. Out-of-the-money puts below the forward are converted to
#     parity-equivalent calls, while out-of-the-money calls at or above the
#     forward are used directly.

#     The fitted curve uses the normalized variables

#         x = K / F

#     and

#         g(x) = C(K) / (D * F),

#     where K is strike, F is the forward price and D is the discount factor.

#     The spline coefficients are obtained by minimizing

#         weighted pricing error
#         + roughness_penalty * integral[g''(x)^2 dx],

#     subject to the following no-arbitrage restrictions:

#     - The spline is a natural cubic spline, so its second derivative is zero
#       at the first and last fitted strikes.
#     - The call-price curve is convex, so g''(x) >= 0.
#     - Its slope remains between -1 and 0.
#     - Its value respects the normalized call-price bounds.
#     - The fitted call price remains nonnegative.

#     Because a cubic spline has a second derivative that is linear between
#     consecutive knots, imposing nonnegative second derivatives at every
#     knot ensures convexity throughout the fitted strike range.

#     Parameters
#     ----------
#     chain : pandas.DataFrame
#         Quotes belonging to one observation date and one exact expiration.

#         The following columns are required:

#         - ``strike_price``
#         - ``put_call``
#         - ``bid_price``
#         - ``ask_price``
#         - ``time_to_expiry_years``

#         If ``mid_price`` is absent, it is calculated as the bid-ask
#         midpoint. If ``quote_is_two_sided`` is present, only rows for which
#         it is true are retained.

#     forward : float or None, optional
#         Forward price corresponding to the chain's expiration.

#         If None, the forward is obtained from
#         ``estimate_forward_and_rate(chain)``. If ``risk_free_rate`` is also
#         None, both quantities are obtained from the same estimation call.

#     risk_free_rate : float or None, optional
#         Continuously compounded annualized risk-free rate expressed as a
#         decimal; for example, 0.05 represents 5%.

#         It is used to calculate the discount factor

#             D = exp(-risk_free_rate * time_to_expiry).

#         If None, it is obtained from
#         ``estimate_forward_and_rate(chain)``.

#     roughness_penalty : float, optional
#         Nonnegative smoothing parameter multiplying

#             integral[g''(x)^2 dx].

#         Larger values produce a smoother and less curved call-price
#         function, while smaller values allow the spline to follow the
#         observed quotes more closely.

#         Its appropriate scale depends on whether spread weighting is used.
#         It should be selected systematically using training data rather
#         than chosen separately by visual inspection for every chain.

#     use_spread_weights : bool, optional
#         If True, price errors are divided by the quote half-spread. This
#         gives less weight to observations with wide bid-ask spreads.

#         Spread weighting is a practical extension to Fengler's original
#         unweighted objective.

#     price_error_floor : float, optional
#         Minimum price-error scale, in option-price points, used when spread
#         weighting is enabled.

#         It prevents quotes with zero or extremely small recorded spreads
#         from receiving excessively large optimization weights.

#     n_grid : int, optional
#         Number of regularly spaced points used to evaluate the fitted
#         spline after optimization.

#         This affects only the returned output grid. It does not determine
#         the number of spline knots because every retained observed strike
#         is used as a breakpoint.

#     Returns
#     -------
#     dict
#         Dictionary containing the following entries:

#         ``strike_grid`` : numpy.ndarray
#             Strike values over the fitted domain.

#         ``smooth_prices`` : numpy.ndarray
#             Fitted present-value call prices evaluated on ``strike_grid``.

#         ``density`` : numpy.ndarray
#             Breeden-Litzenberger risk-neutral density over the fitted
#             strike range, calculated as

#                 f_Q(K) = g''(K / F) / F.

#         ``cdf`` : numpy.ndarray
#             Risk-neutral CDF values over the fitted strike range,
#             calculated as

#                 F_Q(K) = 1 + g'(K / F).

#             Because the unobserved tails are not completed by this
#             function, the first value need not be zero and the last value
#             need not be one.

#         ``spline`` : scipy.interpolate.BSpline
#             Fitted normalized call-price spline. Its input is normalized
#             strike ``K / F``, not the strike itself.

#         ``quotes`` : pandas.DataFrame
#             Filtered quotes used in the optimization, including
#             ``call_equivalent_price``, ``fitted_call_price`` and
#             ``pricing_residual``.

#         ``strike_bounds`` : tuple[float, float]
#             Lowest and highest strikes covered by the fitted spline.

#         ``interior_probability_mass`` : float
#             Probability mass contained between ``strike_bounds``. This is
#             equal to the change in the fitted CDF across the fitted range
#             and need not equal one.

#         ``minimum_density`` : float
#             Smallest fitted density value. It should be nonnegative apart
#             from very small numerical solver errors.

#         ``objective_value`` : float
#             Final value of the optimization objective.

#         ``solver_status`` : str
#             Status reported by CVXPY.

#         ``forward`` : float
#             Forward used in normalization and put-call conversion.

#         ``discount_factor`` : float
#             Discount factor used in the fit.

#         ``time_to_expiry`` : float
#             Time to expiration in years.

#     Raises
#     ------
#     ValueError
#         If the forward or time to expiry is invalid, if the chain contains
#         multiple time-to-expiry values, if duplicate selected strikes are
#         found, or if fewer than eight usable quotes remain.

#     RuntimeError
#         If the convex optimization problem does not return an optimal or
#         approximately optimal solution.

#     Notes
#     -----
#     Puts are converted to parity-equivalent calls using

#         C = P + D * (F - K).

#     The additive parity term is linear in strike and therefore has zero
#     second derivative. Consequently, this transformation does not change
#     the Breeden-Litzenberger density information.

#     This function implements only the single-maturity part of Fengler's
#     method. It does not implement the initial total-variance pre-smoother
#     or the backwards calendar-arbitrage constraints needed to construct an
#     entire arbitrage-free volatility surface.

#     It also does not complete the distribution outside the observed strike
#     range. A separate tail model is required before the resulting CDF can
#     be used as a complete predictive distribution for CRPS.

#     References
#     ----------
#     Fengler, M. R. (2009).
#     "Arbitrage-Free Smoothing of the Implied Volatility Surface."
#     Quantitative Finance, 9(4), 417-428.
#     """

#     if forward is None or risk_free_rate is None:
#         estimates, _ = estimate_forward_and_rate(chain)

#         forward = (
#             estimates["forward"]
#             if forward is None
#             else forward
#         )

#         risk_free_rate = (
#             estimates["risk_free_rate"]
#             if risk_free_rate is None
#             else risk_free_rate
#         )

#     forward = float(forward)
#     risk_free_rate = float(risk_free_rate)

#     if not np.isfinite(forward) or forward <= 0:
#         raise ValueError("Forward must be finite and positive.")

#     time_values = (
#         chain["time_to_expiry_years"]
#         .dropna()
#         .unique()
#     )

#     if len(time_values) != 1:
#         raise ValueError(
#             "Expected one time-to-expiry value."
#         )

#     time_to_expiry = float(time_values[0])

#     if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
#         raise ValueError(
#             "Time to expiry must be finite and positive."
#         )

#     discount_factor = np.exp(
#         -risk_free_rate * time_to_expiry
#     )

#     quotes = chain.copy()

#     # Prefer reliable, two-sided quotes.
#     if "quote_is_two_sided" in quotes.columns:
#         quotes = quotes.loc[
#             quotes["quote_is_two_sided"].fillna(False)
#         ].copy()
#     else:
#         quotes = quotes.loc[
#             (quotes["bid_price"] > 0)
#             & (quotes["ask_price"] >= quotes["bid_price"])
#         ].copy()

#     if "mid_price" not in quotes.columns:
#         quotes["mid_price"] = (
#             quotes["bid_price"] + quotes["ask_price"]
#         ) / 2

#     quotes["spread"] = (
#         quotes["ask_price"] - quotes["bid_price"]
#     )

#     # Use OTM puts below the forward and OTM calls above it.
#     use_quote = (
#         (
#             quotes["put_call"].eq("put")
#             & quotes["strike_price"].lt(forward)
#         )
#         | (
#             quotes["put_call"].eq("call")
#             & quotes["strike_price"].ge(forward)
#         )
#     )

#     quotes = quotes.loc[use_quote].copy()

#     # Convert puts to parity-equivalent call prices:
#     #
#     # C = P + D(F - K)
#     quotes["call_equivalent_price"] = np.where(
#         quotes["put_call"].eq("call"),
#         quotes["mid_price"],
#         quotes["mid_price"]
#         + discount_factor
#         * (forward - quotes["strike_price"]),
#     )

#     required_values = quotes[
#         [
#             "strike_price",
#             "call_equivalent_price",
#             "spread",
#         ]
#     ].to_numpy(dtype=float)

#     valid = (
#         np.isfinite(required_values).all(axis=1)
#         & quotes["strike_price"].gt(0)
#         & quotes["call_equivalent_price"].ge(0)
#         & quotes["spread"].ge(0)
#     )

#     quotes = (
#         quotes.loc[valid]
#         .sort_values("strike_price")
#         .reset_index(drop=True)
#     )

#     if quotes["strike_price"].duplicated().any():
#         raise ValueError(
#             "Duplicate strikes found in selected quotes."
#         )

#     if len(quotes) < 8:
#         raise ValueError(
#             "Too few valid quotes to fit a cubic spline."
#         )

#     # Normalize:
#     #
#     # x = K / F
#     # g(x) = C(K) / (D F)
#     x = (
#         quotes["strike_price"].to_numpy(dtype=float)
#         / forward
#     )

#     y = (
#         quotes["call_equivalent_price"].to_numpy(dtype=float)
#         / (discount_factor * forward)
#     )

#     if use_spread_weights:
#         error_scale = (
#             np.maximum(
#                 quotes["spread"].to_numpy(dtype=float) / 2,
#                 price_error_floor,
#             )
#             / (discount_factor * forward)
#         )
#     else:
#         error_scale = np.ones_like(y)

#     # In a natural smoothing spline, each distinct observation location
#     # is a breakpoint.
#     breakpoints = x
#     x_left = float(x[0])
#     x_right = float(x[-1])

#     degree = 3
#     internal_knots = breakpoints[1:-1]

#     # Open cubic B-spline knot vector.
#     knots = np.concatenate([
#         np.repeat(x_left, degree + 1),
#         internal_knots,
#         np.repeat(x_right, degree + 1),
#     ])

#     n_coefficients = len(knots) - degree - 1

#     # Each column is one B-spline basis function.
#     basis = BSpline(
#         knots,
#         np.eye(n_coefficients),
#         degree,
#         extrapolate=False,
#     )

#     observed_basis = basis(x)

#     endpoint_values = basis(
#         np.array([x_left, x_right])
#     )

#     endpoint_slopes = basis.derivative(1)(
#         np.array([x_left, x_right])
#     )

#     second_at_breakpoints = basis.derivative(2)(
#         breakpoints
#     )

#     # Compute the Fengler roughness penalty exactly:
#     #
#     # integral [g''(x)]^2 dx
#     #
#     # g'' is linear inside each interval, so two-point Gaussian
#     # quadrature exactly integrates its square.
#     interval_left = breakpoints[:-1]
#     interval_right = breakpoints[1:]

#     interval_midpoints = (
#         interval_left + interval_right
#     ) / 2

#     interval_half_widths = (
#         interval_right - interval_left
#     ) / 2

#     offset = interval_half_widths / np.sqrt(3)

#     quadrature_points = np.concatenate([
#         interval_midpoints - offset,
#         interval_midpoints + offset,
#     ])

#     quadrature_weights = np.concatenate([
#         interval_half_widths,
#         interval_half_widths,
#     ])

#     second_at_quadrature = basis.derivative(2)(
#         quadrature_points
#     )

#     coefficients = cp.Variable(n_coefficients)

#     fitted_normalized_prices = (
#         observed_basis @ coefficients
#     )

#     standardized_errors = cp.multiply(
#         1 / error_scale,
#         fitted_normalized_prices - y,
#     )

#     data_loss = (
#         cp.sum_squares(standardized_errors)
#         / len(y)
#     )

#     roughness = cp.sum(
#         cp.multiply(
#             quadrature_weights,
#             cp.square(
#                 second_at_quadrature @ coefficients
#             ),
#         )
#     )

#     fitted_endpoint_values = (
#         endpoint_values @ coefficients
#     )

#     fitted_endpoint_slopes = (
#         endpoint_slopes @ coefficients
#     )

#     fitted_second_derivatives = (
#         second_at_breakpoints @ coefficients
#     )

#     constraints = [
#         # Natural cubic spline: zero second derivative at each end.
#         fitted_second_derivatives[0] == 0,
#         fitted_second_derivatives[-1] == 0,

#         # Convexity. Because g'' is linear between breakpoints,
#         # checking every breakpoint enforces it everywhere.
#         fitted_second_derivatives >= 0,

#         # Boundary slope restrictions. Together with convexity these
#         # imply -1 <= g'(x) <= 0 throughout the fitted interval.
#         fitted_endpoint_slopes[0] >= -1,
#         fitted_endpoint_slopes[-1] <= 0,

#         # Left call-price bounds.
#         fitted_endpoint_values[0]
#         >= max(1 - x_left, 0),

#         fitted_endpoint_values[0] <= 1,

#         # Right call price must remain nonnegative.
#         fitted_endpoint_values[-1] >= 0,
#     ]

#     problem = cp.Problem(
#         cp.Minimize(
#             data_loss
#             + roughness_penalty * roughness
#         ),
#         constraints,
#     )

#     problem.solve(solver=cp.CLARABEL)

#     if problem.status not in {
#         cp.OPTIMAL,
#         # cp.OPTIMAL_INACCURATE,
#     }:
#         raise RuntimeError(
#             f"Spline optimization failed: {problem.status}"
#         )

#     normalized_call_spline = BSpline(
#         knots,
#         np.asarray(coefficients.value).ravel(),
#         degree,
#         extrapolate=False,
#     )

#     # Include all breakpoints in the output grid.
#     x_grid = np.unique(np.concatenate([
#         np.linspace(x_left, x_right, n_grid),
#         breakpoints,
#     ]))

#     strike_grid = forward * x_grid

#     smooth_prices = (
#         discount_factor
#         * forward
#         * normalized_call_spline(x_grid)
#     )

#     density = (
#         normalized_call_spline.derivative(2)(x_grid)
#         / forward
#     )

#     cdf = (
#         1 + normalized_call_spline.derivative(1)(x_grid)
#     )

#     quotes["fitted_call_price"] = (
#         discount_factor
#         * forward
#         * normalized_call_spline(x)
#     )

#     quotes["pricing_residual"] = (
#         quotes["fitted_call_price"]
#         - quotes["call_equivalent_price"]
#     )

#     return {
#         "strike_grid": strike_grid,
#         "smooth_prices": smooth_prices,
#         "density": density,
#         "cdf": cdf,
#         "spline": normalized_call_spline,
#         "quotes": quotes,
#         "strike_bounds": (
#             forward * x_left,
#             forward * x_right,
#         ),
#         "interior_probability_mass": (
#             cdf[-1] - cdf[0]
#         ),
#         "minimum_density": float(density.min()),
#         "objective_value": float(problem.value),
#         "solver_status": problem.status,
#         "forward": forward,
#         "discount_factor": discount_factor,
#         "time_to_expiry": time_to_expiry,
#     }



def fit_shape_constrained_call_spline(
    chain: pd.DataFrame,
    forward: float | None = None,
    risk_free_rate: float | None = None,
    roughness_penalty: float = 1e-4,
    upper_support_multiple: float = 2.5,
    n_tail_knots: int = 3,
    use_spread_weights: bool = True,
    price_error_floor: float = 0.05,
):
    """
    Fit a complete shape-constrained call-price spline to one option chain.

    The curve is fitted in normalized coordinates,

        x = K / F
        g(x) = C(K) / (D * F),

    where F is the forward and D is the discount factor.

    The objective combines quote-fitting error with Fengler's roughness
    penalty,

        integral [g''(x)]^2 dx.

    The constraints ensure that the fitted curve corresponds to a complete
    nonnegative risk-neutral distribution on

        0 <= S_T <= upper_support_multiple * F.

    Notes
    -----
    This is a Fengler-style single-maturity fit with an additional explicit
    compact-support assumption. The compact-support conditions are required
    because observed option strikes do not identify the complete tails.
    """

    if roughness_penalty < 0:
        raise ValueError(
            "roughness_penalty must be nonnegative."
        )

    if upper_support_multiple <= 1:
        raise ValueError(
            "upper_support_multiple must exceed 1."
        )

    if n_tail_knots < 0:
        raise ValueError(
            "n_tail_knots must be nonnegative."
        )

    if forward is None or risk_free_rate is None:
        estimates, _ = estimate_forward_and_rate(chain)

        forward = (
            estimates["forward"]
            if forward is None
            else forward
        )

        risk_free_rate = (
            estimates["risk_free_rate"]
            if risk_free_rate is None
            else risk_free_rate
        )

    forward = float(forward)
    risk_free_rate = float(risk_free_rate)

    if not np.isfinite(forward) or forward <= 0:
        raise ValueError(
            "Forward must be finite and positive."
        )

    time_values = (
        chain["time_to_expiry_years"]
        .dropna()
        .unique()
    )

    if len(time_values) != 1:
        raise ValueError(
            "Expected one time-to-expiry value."
        )

    time_to_expiry = float(time_values[0])

    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ValueError(
            "Time to expiry must be finite and positive."
        )

    discount_factor = np.exp(
        -risk_free_rate * time_to_expiry
    )

    quotes = chain.copy()

    if "quote_is_two_sided" in quotes.columns:
        quotes = quotes.loc[
            quotes["quote_is_two_sided"].fillna(False)
        ].copy()
    else:
        quotes = quotes.loc[
            (quotes["bid_price"] > 0)
            & (quotes["ask_price"] >= quotes["bid_price"])
        ].copy()

    quotes["spread"] = (
        quotes["ask_price"] - quotes["bid_price"]
    )

    # Use OTM puts below the forward and OTM calls above it.
    use_quote = (
        (
            quotes["put_call"].eq("put")
            & quotes["strike_price"].lt(forward)
        )
        | (
            quotes["put_call"].eq("call")
            & quotes["strike_price"].ge(forward)
        )
    )

    quotes = quotes.loc[use_quote].copy()

    # Convert puts to parity-equivalent calls:
    # C = P + D(F - K).
    quotes["call_equivalent_price"] = np.where(
        quotes["put_call"].eq("call"),
        quotes["mid_price"],
        quotes["mid_price"]
        + discount_factor
        * (forward - quotes["strike_price"]),
    )

    values = quotes[
        [
            "strike_price",
            "call_equivalent_price",
            "spread",
        ]
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(values).all(axis=1)
        & np.asarray(quotes["strike_price"].gt(0))
        & quotes["call_equivalent_price"].ge(0)
        & quotes["spread"].ge(0)
    )

    quotes = (
        quotes.loc[valid]
        .sort_values("strike_price")
        .reset_index(drop=True)
    )

    if quotes["strike_price"].duplicated().any():
        raise ValueError(
            "Duplicate selected strikes found."
        )

    if len(quotes) < 8:
        raise ValueError(
            "Too few valid quotes to fit the spline."
        )

    observed_x = (
        quotes["strike_price"].to_numpy(dtype=float)
        / forward
    )

    observed_y = (
        quotes["call_equivalent_price"].to_numpy(dtype=float)
        / (discount_factor * forward)
    )

    if use_spread_weights:
        error_scale = (
            np.maximum(
                quotes["spread"].to_numpy(dtype=float) / 2,
                price_error_floor,
            )
            / (discount_factor * forward)
        )
    else:
        error_scale = np.ones_like(observed_y)

    # Full normalized support: 0 <= S_T/F <= x_upper.
    x_upper = max(
        upper_support_multiple,
        1.05 * observed_x.max(),
    )

    # Add knots in the unobserved tails so the spline has enough
    # flexibility to join the market-observed region smoothly.
    left_tail_knots = np.linspace(
        0,
        observed_x.min(),
        n_tail_knots + 2,
    )[1:-1]

    right_tail_knots = np.linspace(
        observed_x.max(),
        x_upper,
        n_tail_knots + 2,
    )[1:-1]

    internal_knots = np.unique(np.concatenate([
        left_tail_knots,
        observed_x,
        right_tail_knots,
    ]))

    internal_knots = internal_knots[
        (internal_knots > 0)
        & (internal_knots < x_upper)
    ]

    breakpoints = np.concatenate([
        [0.0],
        internal_knots,
        [x_upper],
    ])

    degree = 3

    knots = np.concatenate([
        np.repeat(0.0, degree + 1),
        internal_knots,
        np.repeat(x_upper, degree + 1),
    ])

    n_coefficients = len(knots) - degree - 1

    # Each column is one cubic B-spline basis function.
    basis = BSpline(
        knots,
        np.eye(n_coefficients),
        degree,
        extrapolate=False,
    )

    observed_basis = basis(observed_x)
    value_basis = basis(breakpoints)
    first_basis = basis.derivative(1)(breakpoints)
    second_basis = basis.derivative(2)(breakpoints)

    # Exact two-point Gaussian quadrature for
    # integral [g''(x)]² dx.
    interval_left = breakpoints[:-1]
    interval_right = breakpoints[1:]

    interval_midpoints = (
        interval_left + interval_right
    ) / 2

    interval_half_widths = (
        interval_right - interval_left
    ) / 2

    offsets = interval_half_widths / np.sqrt(3)

    quadrature_points = np.concatenate([
        interval_midpoints - offsets,
        interval_midpoints + offsets,
    ])

    quadrature_weights = np.concatenate([
        interval_half_widths,
        interval_half_widths,
    ])

    second_quadrature_basis = basis.derivative(2)(
        quadrature_points
    )

    coefficients = cp.Variable(n_coefficients)

    fitted_observations = (
        observed_basis @ coefficients
    )

    standardized_errors = cp.multiply(
        1 / error_scale,
        fitted_observations - observed_y,
    )

    data_loss = (
        cp.sum_squares(standardized_errors)
        / len(observed_y)
    )

    roughness = cp.sum(
        cp.multiply(
            quadrature_weights,
            cp.square(
                second_quadrature_basis @ coefficients
            ),
        )
    )

    fitted_values = value_basis @ coefficients
    fitted_slopes = first_basis @ coefficients
    fitted_second = second_basis @ coefficients

    intrinsic_values = np.maximum(
        1 - breakpoints,
        0,
    )

    constraints = [
        # Complete-distribution boundary conditions.
        fitted_values[0] == 1,
        fitted_slopes[0] == -1,

        fitted_values[-1] == 0,
        fitted_slopes[-1] == 0,

        # Natural boundaries make the density vanish smoothly.
        fitted_second[0] == 0,
        fitted_second[-1] == 0,

        # Nonnegative Breeden-Litzenberger density.
        fitted_second >= 0,

        # These are theoretically implied by the preceding constraints,
        # but including them improves numerical robustness.
        fitted_slopes >= -1,
        fitted_slopes <= 0,
        fitted_values >= intrinsic_values,
        fitted_values <= 1,
    ]

    problem = cp.Problem(
        cp.Minimize(
            data_loss
            + roughness_penalty * roughness
        ),
        constraints,
    )

    problem.solve(solver=cp.CLARABEL)

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise RuntimeError(
            f"Spline optimization failed: {problem.status}"
        )

    normalized_spline = BSpline(
        knots,
        np.asarray(coefficients.value).ravel(),
        degree,
        extrapolate=False,
    )

    quotes["fitted_call_price"] = (
        discount_factor
        * forward
        * normalized_spline(observed_x)
    )

    quotes["pricing_residual"] = (
        quotes["fitted_call_price"]
        - quotes["call_equivalent_price"]
    )

    return {
        "spline": normalized_spline,
        "forward": forward,
        "risk_free_rate": risk_free_rate,
        "discount_factor": discount_factor,
        "time_to_expiry": time_to_expiry,
        "x_upper": x_upper,
        "breakpoints": breakpoints,
        "quotes": quotes,
        "objective_value": float(problem.value),
        "solver_status": problem.status,
    }






def construct_risk_neutral_distribution_from_spline(
    fit: dict,
    n_grid: int = 1000,
) -> dict:
    """
    Evaluate call prices, the risk-neutral PDF and CDF from a fitted spline.

    Returns a probability distribution and some information about it.
    """

    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    spline = fit["spline"]
    forward = fit["forward"]
    discount_factor = fit["discount_factor"]
    x_upper = fit["x_upper"]
    breakpoints = fit["breakpoints"]

    x_grid = np.unique(np.concatenate([
        np.linspace(0, x_upper, n_grid),
        breakpoints,
    ]))

    strike_grid = forward * x_grid

    normalized_prices = spline(x_grid)
    first_derivative = spline.derivative(1)(x_grid)
    second_derivative = spline.derivative(2)(x_grid)

    call_prices = (
        discount_factor
        * forward
        * normalized_prices
    )

    # Breeden-Litzenberger:
    #
    # f_Q(K) = C''(K) / D = g''(K/F) / F
    density = second_derivative / forward

    # F_Q(K) = 1 + C'(K)/D = 1 + g'(K/F)
    cdf = 1 + first_derivative

    numerical_mass = np.trapezoid(
        density,
        strike_grid,
    )

    numerical_mean = np.trapezoid(
        strike_grid * density,
        strike_grid,
    )

    return {
        "strike_grid": strike_grid,
        "call_prices": call_prices,
        "density": density,
        "cdf": cdf,
        "mass": float(numerical_mass),
        "mean": float(numerical_mean),
        "minimum_density": float(density.min()),
        "cdf_start": float(cdf[0]),
        "cdf_end": float(cdf[-1]),
        "support": (
            0.0,
            forward * x_upper,
        ),
    }

