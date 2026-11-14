"""Black's formula for options on a forward, and its inversion.

The forward-based parameterisation is used throughout because the forward
inferred from put-call parity already carries the expected dividends and
financing costs of the index, so no separate dividend model is needed.

All functions accept scalars or arrays and broadcast.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "black_call_price",
    "black_put_price",
    "black_vega",
    "implied_volatility_call",
]


def _d1_d2(forward, strike, volatility, time_to_expiry):
    """Return d1, d2 and sigma*sqrt(T), with zero-volatility handled."""
    sigma_root_t = volatility * np.sqrt(time_to_expiry)

    # Guard the division; the degenerate branch is patched by the caller.
    safe = np.where(sigma_root_t > 0, sigma_root_t, 1.0)

    d1 = (np.log(forward / strike) + 0.5 * sigma_root_t**2) / safe
    d2 = d1 - sigma_root_t
    return d1, d2, sigma_root_t


def black_call_price(
    forward,
    strike,
    volatility,
    time_to_expiry,
    risk_free_rate,
):
    """Present value of a European call under Black's formula.

    ``C = D * [F * N(d1) - K * N(d2)]``, with ``D = exp(-r * T)`` the
    continuously compounded discount factor.

    Parameters
    ----------
    forward, strike, volatility, time_to_expiry, risk_free_rate : array_like
        Broadcastable. Volatility is annualised and expressed as a decimal;
        time to expiry is in years.

    Returns
    -------
    numpy.ndarray or float
        Discounted call value. A zero volatility returns the discounted
        intrinsic value.
    """
    forward, strike, volatility, time_to_expiry, risk_free_rate = np.broadcast_arrays(
        *(np.asarray(a, dtype=float)
          for a in (forward, strike, volatility, time_to_expiry, risk_free_rate))
    )

    if np.any(forward <= 0) or np.any(strike <= 0):
        raise ValueError("Forward and strike must be positive.")
    if np.any(volatility < 0):
        raise ValueError("Volatility must be non-negative.")
    if np.any(time_to_expiry <= 0):
        raise ValueError("Time to expiry must be positive.")

    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    d1, d2, sigma_root_t = _d1_d2(forward, strike, volatility, time_to_expiry)

    value = discount_factor * (
        forward * norm.cdf(d1) - strike * norm.cdf(d2)
    )

    intrinsic = discount_factor * np.maximum(forward - strike, 0.0)
    value = np.where(sigma_root_t > 0, value, intrinsic)

    return value if value.ndim else float(value)


def black_put_price(
    forward,
    strike,
    volatility,
    time_to_expiry,
    risk_free_rate,
):
    """Present value of a European put under Black's formula.

    ``P = D * [K * N(-d2) - F * N(-d1)]``.
    """
    forward, strike, volatility, time_to_expiry, risk_free_rate = np.broadcast_arrays(
        *(np.asarray(a, dtype=float)
          for a in (forward, strike, volatility, time_to_expiry, risk_free_rate))
    )

    if np.any(forward <= 0) or np.any(strike <= 0):
        raise ValueError("Forward and strike must be positive.")
    if np.any(volatility < 0):
        raise ValueError("Volatility must be non-negative.")
    if np.any(time_to_expiry <= 0):
        raise ValueError("Time to expiry must be positive.")

    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    d1, d2, sigma_root_t = _d1_d2(forward, strike, volatility, time_to_expiry)

    value = discount_factor * (
        strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    )

    intrinsic = discount_factor * np.maximum(strike - forward, 0.0)
    value = np.where(sigma_root_t > 0, value, intrinsic)

    return value if value.ndim else float(value)


def black_vega(
    forward,
    strike,
    volatility,
    time_to_expiry,
    risk_free_rate,
):
    """Sensitivity of the option value to volatility.

    ``dC/dsigma = D * F * phi(d1) * sqrt(T)``, identical for calls and puts.
    Used to convert a quoted bid-ask spread into a volatility-space noise
    estimate.
    """
    forward, strike, volatility, time_to_expiry, risk_free_rate = np.broadcast_arrays(
        *(np.asarray(a, dtype=float)
          for a in (forward, strike, volatility, time_to_expiry, risk_free_rate))
    )

    discount_factor = np.exp(-risk_free_rate * time_to_expiry)
    d1, _, _ = _d1_d2(forward, strike, volatility, time_to_expiry)

    value = discount_factor * forward * norm.pdf(d1) * np.sqrt(time_to_expiry)
    return value if value.ndim else float(value)


def implied_volatility_call(
    market_price,
    forward,
    strike,
    time_to_expiry,
    risk_free_rate,
    volatility_bounds=(1e-7, 5.0),
    tolerance=1e-10,
    max_iterations=80,
):
    """Black implied volatility of a European call, inverted numerically.

    The Black price is strictly increasing in volatility, so a vectorised
    bisection converges monotonically for every element at once. This
    replaces the previous per-quote ``brentq`` loop, which dominated the
    runtime of the implied-volatility baseline.

    Quotes that violate the no-arbitrage price bounds
    ``D * max(F - K, 0) <= C < D * F`` return ``nan``; a price exactly at
    the lower bound returns zero.

    Parameters
    ----------
    market_price, forward, strike, time_to_expiry, risk_free_rate : array_like
        Broadcastable inputs.
    volatility_bounds : tuple[float, float]
        Bracketing interval for the search.
    tolerance : float
        Absolute width of the bracket at which the search stops.
    max_iterations : int
        Bisection steps. Eighty halvings take the default bracket below
        1e-16, so the tolerance binds first in practice.

    Returns
    -------
    numpy.ndarray or float
        Annualised implied volatility, or ``nan`` where not identified.
    """
    scalar_input = all(
        np.isscalar(a) or np.ndim(a) == 0
        for a in (market_price, forward, strike, time_to_expiry, risk_free_rate)
    )

    market_price, forward, strike, time_to_expiry, risk_free_rate = (
        np.broadcast_arrays(
            *(np.asarray(a, dtype=float)
              for a in (market_price, forward, strike,
                        time_to_expiry, risk_free_rate))
        )
    )

    lower_bound, upper_bound = volatility_bounds
    if not 0 < lower_bound < upper_bound:
        raise ValueError("Volatility bounds must satisfy 0 < lower < upper.")

    finite = (
        np.isfinite(market_price)
        & np.isfinite(forward)
        & np.isfinite(strike)
        & np.isfinite(time_to_expiry)
        & np.isfinite(risk_free_rate)
    )

    usable = finite & (forward > 0) & (strike > 0) & (time_to_expiry > 0)

    discount_factor = np.where(
        usable, np.exp(-risk_free_rate * time_to_expiry), 1.0
    )

    minimum_price = discount_factor * np.maximum(forward - strike, 0.0)
    maximum_price = discount_factor * forward

    # An option worth its intrinsic value to within a rounding error of
    # the discounted forward carries no volatility information. The
    # tolerance is relative to the index level rather than absolute, so
    # that it means the same thing at SPX 1200 and SPX 6000.
    at_lower_bound = usable & (
        np.abs(market_price - minimum_price) <= 1e-12 * maximum_price
    )

    in_range = (
        usable
        & (market_price >= minimum_price)
        & (market_price < maximum_price)
        & ~at_lower_bound
    )

    # Safe placeholder inputs keep the vectorised pricer away from the
    # degenerate elements we are going to overwrite anyway.
    safe_forward = np.where(usable, forward, 1.0)
    safe_strike = np.where(usable, strike, 1.0)
    safe_time = np.where(usable, time_to_expiry, 1.0)

    low = np.full(safe_forward.shape, float(lower_bound))
    high = np.full(safe_forward.shape, float(upper_bound))

    def price_at(volatility):
        return black_call_price(
            safe_forward, safe_strike, volatility, safe_time, risk_free_rate
        )

    # Anything the bracket cannot reach is not identified.
    bracketed = in_range & (price_at(high) >= market_price)

    for _ in range(max_iterations):
        if np.all(high - low <= tolerance):
            break
        middle = 0.5 * (low + high)
        too_cheap = price_at(middle) < market_price
        low = np.where(too_cheap, middle, low)
        high = np.where(too_cheap, high, middle)

    result = np.where(bracketed, 0.5 * (low + high), np.nan)
    result = np.where(at_lower_bound, 0.0, result)

    return float(result) if scalar_input else result
