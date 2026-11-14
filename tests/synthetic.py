"""Synthetic option chains with known answers.

Every test in this suite compares against something exact: a Black-Scholes
chain has a known forward, a known rate and an analytic lognormal density,
so the whole pipeline can be checked end to end without any market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from implied_stock_distributions.pricing import black_call_price, black_put_price

DEFAULT_FORWARD = 4000.0
DEFAULT_RATE = 0.045
DEFAULT_VOLATILITY = 0.18


def synthetic_chain(
    forward: float = DEFAULT_FORWARD,
    rate: float = DEFAULT_RATE,
    volatility: float = DEFAULT_VOLATILITY,
    days_to_expiry: int = 30,
    strike_step: float = 50.0,
    lower_moneyness: float = 0.55,
    upper_moneyness: float = 1.35,
    spread: float = 0.5,
    noise: float = 0.0,
    skew: float = 0.0,
    seed: int = 0,
):
    """Build an arbitrage-free chain from Black's formula.

    Parameters
    ----------
    spread : float
        Bid-ask width in index points, applied symmetrically about the
        mid-price. It is widened to at least four times ``noise``, so the
        quotes stay consistent with their own spreads -- a market whose mids
        wobble by more than its quoted width would (rightly) fail the parity
        quality gate in ``forwards``.
    noise : float
        Standard deviation of the perturbation added to each mid-price, in
        index points. Zero gives an exactly arbitrage-free chain.
    skew : float
        Slope of a linear volatility smile in log-moneyness. A negative
        value gives the usual equity-index shape.
    """
    time_to_expiry = days_to_expiry / 365.0
    spread = max(spread, 4.0 * noise)
    discount_factor = float(np.exp(-rate * time_to_expiry))
    generator = np.random.default_rng(seed)

    first = np.floor(lower_moneyness * forward / strike_step) * strike_step
    strikes = np.arange(first, upper_moneyness * forward + strike_step, strike_step)

    rows = []
    for strike in strikes:
        local_volatility = volatility + skew * np.log(strike / forward)
        for kind, price_function in (("call", black_call_price),
                                     ("put", black_put_price)):
            mid = float(price_function(
                forward, strike, local_volatility, time_to_expiry, rate
            ))
            if noise:
                mid = max(mid + float(generator.normal(0.0, noise)), 0.05)

            rows.append({
                "strike_price": float(strike),
                "put_call": kind,
                "mid_price": mid,
                "bid_price": max(mid - spread / 2, 0.05),
                "ask_price": mid + spread / 2,
                "quote_is_two_sided": True,
                "time_to_expiry_years": time_to_expiry,
                "underlying_price": forward * discount_factor,
                "data_date": pd.Timestamp("2024-05-22"),
                "expiration_date": pd.Timestamp("2024-06-21"),
                "target_dte_days": 30,
            })

    truth = {
        "forward": forward,
        "rate": rate,
        "volatility": volatility,
        "time_to_expiry": time_to_expiry,
        "discount_factor": discount_factor,
        "skew": skew,
    }
    return pd.DataFrame(rows), truth


def lognormal_density(values, forward, volatility, time_to_expiry):
    """Analytic risk-neutral density implied by Black's formula."""
    values = np.asarray(values, dtype=float)
    density = np.zeros_like(values)

    positive = values > 0
    sigma_root_t = volatility * np.sqrt(time_to_expiry)

    z = (
        np.log(values[positive] / forward) + 0.5 * sigma_root_t**2
    ) / sigma_root_t

    density[positive] = (
        np.exp(-0.5 * z**2) / (values[positive] * sigma_root_t * np.sqrt(2 * np.pi))
    )
    return density


def lognormal_cdf(values, forward, volatility, time_to_expiry):
    """Analytic risk-neutral CDF implied by Black's formula."""
    from scipy.stats import norm

    values = np.asarray(values, dtype=float)
    cdf = np.zeros_like(values)

    positive = values > 0
    sigma_root_t = volatility * np.sqrt(time_to_expiry)

    cdf[positive] = norm.cdf(
        (np.log(values[positive] / forward) + 0.5 * sigma_root_t**2) / sigma_root_t
    )
    return cdf
