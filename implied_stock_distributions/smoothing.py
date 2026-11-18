"""Implied-volatility smoothing: the unconstrained baseline.

The comparison this project makes is between a shape-constrained fit in
price space and the more common practice of smoothing the volatility smile
and converting back to prices. For that comparison to say anything about
*constraints*, the baseline has to be given the same information and the
same treatment everywhere else. Four things matter:

* **Quote selection.** The baseline uses the same out-of-the-money puts and
  calls as the constrained fit, converted to call-equivalent prices. Using
  only quoted calls would replace the liquid SPX put wing with deep
  in-the-money calls, which are wide and rarely traded, and would lose the
  left tail that dominates the score.
* **Smoothing strength.** ``make_splrep``'s ``s`` is an upper bound on the
  weighted sum of squared residuals, not a penalty weight. With weights set
  to the reciprocal of each quote's log-volatility noise, the natural bound
  is the number of observations. A fixed number such as ``0.05 * n`` leaves
  the bound slack at realistic noise levels, and the fit then collapses to
  a single cubic in log-moneyness -- a four-parameter smile that cannot
  represent SPX skew, and a much weaker baseline than intended.
* **Support.** Volatility is extrapolated flat beyond the traded strikes,
  which is standard practice, so that the baseline density covers the same
  range as the constrained one. Comparing a truncated density against a
  complete one would measure the tail treatment, not the smoothing.
* **Tabulation.** Pass ``strike_grid`` -- from ``evaluation.common_strike_grid``
  -- so both methods are scored on identical grids. Left to itself this
  function used a plain ``linspace`` while the constrained fit merged its
  spline breakpoints, one per traded strike, into its own grid. Under the
  old trapezoid quadrature that difference alone was worth about half a per
  cent of the score, in the constrained fit's favour.

Two things the baseline still does *not* get, and they are the point of the
comparison rather than oversights:

* any guarantee of a non-negative density -- the fitted price curve can be
  locally concave, and the clipping and renormalisation below are exactly
  the illegitimate steps whose cost the project is trying to measure; and
* a mean equal to the forward. The constrained fit has ``E_Q[S_T] = F`` by
  construction; after clipping, renormalising and truncating at
  ``x_upper * F`` this one does not. ``mean_over_forward`` is reported
  alongside ``mass_before_normalisation`` so the size of that displacement
  is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, make_splrep

from .config import (
    IV_GRID_POINTS,
    IV_MAX_LOG_NOISE,
    IV_MIN_LOG_NOISE,
    IV_SMOOTHING_CHI_SQUARE_MULTIPLE,
    MIN_ERROR_ABSOLUTE,
    MIN_ERROR_FRACTION_OF_FORWARD,
    UPPER_SUPPORT_MULTIPLE,
)
from .errors import ChainQualityError, InsufficientQuotesError
from .fitting import select_call_equivalent_quotes
from .forwards import estimate_forward_and_rate
from .pricing import black_call_price, black_vega, implied_volatility_call

__all__ = [
    "implied_vols_from_quotes",
    "fit_log_iv_curve",
    "create_density_from_chain",
    "baseline_diagnostics",
]


def implied_vols_from_quotes(
    quotes: pd.DataFrame,
    forward: float,
    time_to_expiry: float,
    risk_free_rate: float,
):
    """Invert call-equivalent prices to Black implied volatilities.

    Returns the strikes, the implied volatilities and an estimate of the
    noise on each observation in log-volatility units, obtained by dividing
    the quote half-spread by vega.
    """
    strikes = quotes["strike_price"].to_numpy(dtype=float)
    prices = quotes["call_equivalent_price"].to_numpy(dtype=float)

    # The same error floor the constrained fit applies in
    # ``fitting.pricing_error_scale``. Without it the two arms are given
    # different beliefs about how noisy the same quote is: a quote whose
    # bid and ask are one tick apart is not known to a tenth of a tick, and
    # a baseline told otherwise chases the tick grid and returns a smile
    # whose second derivative oscillates. Handicapping the baseline through
    # its noise model would be exactly the kind of asymmetry this
    # comparison is supposed to have removed.
    floor = max(MIN_ERROR_FRACTION_OF_FORWARD * float(forward), MIN_ERROR_ABSOLUTE)
    half_spread = np.maximum(quotes["spread"].to_numpy(dtype=float) / 2.0, floor)

    implied = implied_volatility_call(
        market_price=prices,
        forward=forward,
        strike=strikes,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
    )

    usable = np.isfinite(implied) & (implied > 0) & (strikes > 0)

    strikes = strikes[usable]
    implied = implied[usable]
    half_spread = half_spread[usable]

    vega = black_vega(
        forward, strikes, implied, time_to_expiry, risk_free_rate
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        volatility_noise = np.where(vega > 0, half_spread / vega, np.nan)
        log_noise = volatility_noise / implied

    log_noise = np.where(np.isfinite(log_noise), log_noise, IV_MAX_LOG_NOISE)
    log_noise = np.clip(log_noise, IV_MIN_LOG_NOISE, IV_MAX_LOG_NOISE)

    order = np.argsort(strikes)
    return strikes[order], implied[order], log_noise[order]


def fit_log_iv_curve(
    strikes,
    implied_volatilities,
    log_noise,
    forward: float,
    chi_square_multiple: float = IV_SMOOTHING_CHI_SQUARE_MULTIPLE,
):
    """Smooth log implied volatility against log-moneyness.

    Fitting the logarithm keeps fitted volatilities positive. The smoothing
    bound is ``chi_square_multiple * n``, which is the expected weighted
    residual sum when the noise estimate is right and the curve is correct.

    Returns a callable of strike, extrapolating flat beyond the observed
    range, together with the observed log-moneyness bounds.
    """
    strikes = np.asarray(strikes, dtype=float)
    implied_volatilities = np.asarray(implied_volatilities, dtype=float)
    log_noise = np.asarray(log_noise, dtype=float)

    if forward <= 0:
        raise ChainQualityError("Forward must be positive.")

    if len(strikes) < 6:
        raise InsufficientQuotesError(
            "At least six implied volatilities are required to smooth."
        )

    log_moneyness = np.log(strikes / forward)
    log_iv = np.log(implied_volatilities)

    if np.any(np.diff(log_moneyness) <= 0):
        raise ChainQualityError("Duplicate strikes found in the IV data.")

    weights = 1.0 / log_noise
    smoothing = chi_square_multiple * len(strikes)

    log_iv_spline = make_splrep(
        log_moneyness, log_iv, k=3, w=weights, s=smoothing
    )

    left, right = float(log_moneyness[0]), float(log_moneyness[-1])

    def iv_of_strike(strike):
        strike = np.asarray(strike, dtype=float)
        if np.any(strike <= 0):
            raise ValueError("Strikes must be positive.")
        # Flat extrapolation outside the traded range.
        clamped = np.clip(np.log(strike / forward), left, right)
        return np.exp(log_iv_spline(clamped))

    # How many knots the bound actually bought. This is the number to watch:
    # `s = n` is the statistically right residual sum, but FITPACK reaches it
    # with as many knots as it needs, and on a chain with a hundred tightly
    # quoted strikes that can be thirty. A thirty-knot smile is smooth to
    # look at and still has enough curvature freedom to make the second
    # derivative of the price curve -- which is the density -- swing far
    # outside anything a distribution could be. Report it rather than
    # discovering it from the density.
    iv_of_strike.n_knots = int(len(log_iv_spline.t) - 2 * 3 - 2)
    iv_of_strike.weighted_residual_sum = float(
        np.sum((weights * (log_iv_spline(log_moneyness) - log_iv)) ** 2)
    )
    iv_of_strike.smoothing_bound = float(smoothing)

    return iv_of_strike, (left, right)


def create_density_from_chain(
    chain: pd.DataFrame,
    forward: float | None = None,
    risk_free_rate: float | None = None,
    upper_support_multiple: float = UPPER_SUPPORT_MULTIPLE,
    n_grid: int = IV_GRID_POINTS,
    chi_square_multiple: float = IV_SMOOTHING_CHI_SQUARE_MULTIPLE,
    quotes: pd.DataFrame | None = None,
    strike_grid=None,
) -> dict:
    """Risk-neutral density from an unconstrained smile fit.

    The smoothed volatility curve is priced back to calls on a strike grid
    covering the same support as the constrained fit, then differentiated
    twice. Negative density values are clipped and the result is
    renormalised; both steps are recorded in the output so their size can
    be reported rather than hidden. The displacement of the mean away from
    the forward that they cause is recorded too.

    Parameters
    ----------
    strike_grid : array_like, optional
        Tabulate on this grid instead of an internally chosen ``linspace``.
        Pass ``evaluation.common_strike_grid(...)`` so that this method and
        the constrained fit are scored on identical abscissae.

    Returns
    -------
    dict
        ``strike_grid``, ``density``, ``cdf``, ``price_smooth``,
        ``iv_of_strike``, and the diagnostics ``negative_mass_clipped``,
        ``mass_before_normalisation``, ``mean`` and ``mean_over_forward``.
    """
    if forward is None or risk_free_rate is None:
        estimates, _ = estimate_forward_and_rate(chain)
        forward = estimates["forward"] if forward is None else forward
        risk_free_rate = (
            estimates["risk_free_rate"] if risk_free_rate is None
            else risk_free_rate
        )

    forward = float(forward)
    risk_free_rate = float(risk_free_rate)

    time_values = chain["time_to_expiry_years"].dropna().unique()
    if len(time_values) != 1:
        raise ChainQualityError("Expected one time-to-expiry value per chain.")
    time_to_expiry = float(time_values[0])

    discount_factor = float(np.exp(-risk_free_rate * time_to_expiry))

    if quotes is None:
        quotes = select_call_equivalent_quotes(chain, forward, discount_factor)

    strikes, implied, log_noise = implied_vols_from_quotes(
        quotes, forward, time_to_expiry, risk_free_rate
    )

    iv_of_strike, log_moneyness_bounds = fit_log_iv_curve(
        strikes, implied, log_noise, forward,
        chi_square_multiple=chi_square_multiple,
    )

    x_upper = max(upper_support_multiple, 1.05 * strikes.max() / forward)

    # The density is always produced by differentiating on a UNIFORM grid,
    # whatever grid it is finally reported on. Differentiating an
    # interpolating cubic spline through a merged grid -- a linspace united
    # with the constrained fit's breakpoints -- divides by the spacing
    # between abscissae that can be arbitrarily close, and the second
    # derivative is then dominated by that rather than by the smile. The
    # shared grid the score needs and the well-conditioned grid the
    # differencing needs are two different objects, so build both.
    internal_grid = np.linspace(1e-4 * forward, x_upper * forward, n_grid)

    iv_internal = iv_of_strike(internal_grid)
    price_internal = black_call_price(
        forward, internal_grid, iv_internal, time_to_expiry, risk_free_rate
    )

    price_curve = CubicSpline(internal_grid, price_internal, extrapolate=False)
    second_derivative = price_curve.derivative(2)(internal_grid)

    # Breeden-Litzenberger on the reconstructed price curve.
    internal_density = np.exp(risk_free_rate * time_to_expiry) * second_derivative
    internal_density = np.nan_to_num(internal_density, nan=0.0)

    # The negative mass is a property of the fit, so it is measured where
    # the fit was differentiated rather than after any resampling.
    negative_mass = float(
        np.trapezoid(np.minimum(internal_density, 0.0), internal_grid)
    )

    if strike_grid is None:
        strike_grid = internal_grid
        raw_density = internal_density
        price_smooth = price_internal
    else:
        strike_grid = np.unique(np.asarray(strike_grid, dtype=float))
        strike_grid = strike_grid[np.isfinite(strike_grid) & (strike_grid > 0)]
        if strike_grid.size < 4:
            raise ValueError("A supplied strike grid needs at least four points.")

        raw_density = np.interp(
            strike_grid, internal_grid, internal_density, left=0.0, right=0.0
        )
        price_smooth = black_call_price(
            forward, strike_grid, iv_of_strike(strike_grid),
            time_to_expiry, risk_free_rate,
        )

    density = np.clip(raw_density, 0.0, None)
    mass = float(np.trapezoid(density, strike_grid))

    if not np.isfinite(mass) or mass <= 0:
        raise ChainQualityError(
            "Cannot normalise density: integral is non-positive or non-finite."
        )

    density = density / mass

    # Clipping, renormalising and truncating all move the mean. The
    # constrained fit has E[S_T] = F as an identity; this one does not, and
    # CRPS is sensitive enough to location that the gap needs reporting.
    mean = float(np.trapezoid(strike_grid * density, strike_grid))

    cdf = np.concatenate([[0.0], np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(strike_grid)
    )])
    cdf = np.clip(cdf, 0.0, 1.0)

    return {
        "strike_grid": strike_grid,
        "price_smooth": price_smooth,
        "density": density,
        "cdf": cdf,
        "iv_of_strike": iv_of_strike,
        "observed_strikes": strikes,
        "observed_iv": implied,
        "log_moneyness_bounds": log_moneyness_bounds,
        "forward": forward,
        "risk_free_rate": risk_free_rate,
        "time_to_expiry": time_to_expiry,
        "negative_mass_clipped": negative_mass,
        "mass_before_normalisation": mass,
        "mean": mean,
        "mean_over_forward": mean / forward,
        "n_smile_knots": getattr(iv_of_strike, "n_knots", None),
        "smile_weighted_residual_sum": getattr(
            iv_of_strike, "weighted_residual_sum", None
        ),
        "smile_smoothing_bound": getattr(iv_of_strike, "smoothing_bound", None),
        "n_quotes": int(len(strikes)),
        "support": (float(strike_grid[0]), float(strike_grid[-1])),
    }


def baseline_diagnostics(
    scores: pd.DataFrame, horizon_column: str = "target_dte_days"
) -> pd.DataFrame:
    """How hard the baseline's smile works, and how much density it clips.

    Two per-chain diagnostics from ``create_density_from_chain``, summarised
    by horizon and overall: the number of knots in the smoothed smile and
    the negative density mass clipped before renormalising. A healthy fit
    has a knot or two and almost nothing clipped. When stale quotes break
    the spread-based noise estimate, FITPACK adds knots to chase them, and
    the resulting density is mostly clipped mass; on the synthetic demo
    that happens on a large share of chains, and these distributions are
    what say how often it happens on real ones.
    """
    frame = pd.DataFrame({
        "horizon": scores[horizon_column].astype(int).astype(str),
        "knots": pd.to_numeric(scores["n_smile_knots"], errors="coerce"),
        "clipped": pd.to_numeric(
            scores["negative_mass_clipped"], errors="coerce"
        ).abs(),
    })
    order = sorted(frame["horizon"].unique(), key=int) + ["all"]
    frame = pd.concat([frame, frame.assign(horizon="all")], ignore_index=True)

    grouped = frame.groupby("horizon")
    table = pd.DataFrame({
        "chains": grouped.size(),
        "knots_median": grouped["knots"].median(),
        "knots_p90": grouped["knots"].quantile(0.9),
        "knots_max": grouped["knots"].max(),
        "clipped_mass_median": grouped["clipped"].median(),
        "clipped_mass_p90": grouped["clipped"].quantile(0.9),
        "clipped_mass_max": grouped["clipped"].max(),
        "share_clipped_over_10pct": grouped["clipped"].apply(
            lambda values: float(np.mean(values > 0.1))
        ),
    }).reindex(order)
    table.index.name = horizon_column
    return table
