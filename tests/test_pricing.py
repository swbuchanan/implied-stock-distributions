"""Black's formula and its inversion, checked against exact identities."""

import numpy as np
import pytest

from implied_stock_distributions.pricing import (
    black_call_price,
    black_put_price,
    black_vega,
    implied_volatility_call,
)

FORWARD = 4000.0
TIME = 0.25
RATE = 0.045
STRIKES = np.array([1500.0, 2800.0, 3600.0, 4000.0, 4400.0, 5600.0, 9000.0])
VOLATILITIES = np.array([0.08, 0.15, 0.22, 0.35, 0.60, 1.20])


def test_put_call_parity_holds_exactly():
    """C - P must equal D (F - K) at every strike and volatility."""
    discount = np.exp(-RATE * TIME)

    for volatility in VOLATILITIES:
        call = black_call_price(FORWARD, STRIKES, volatility, TIME, RATE)
        put = black_put_price(FORWARD, STRIKES, volatility, TIME, RATE)
        assert np.allclose(call - put, discount * (FORWARD - STRIKES), atol=1e-9)


def test_prices_respect_no_arbitrage_bounds():
    discount = np.exp(-RATE * TIME)

    for volatility in VOLATILITIES:
        call = black_call_price(FORWARD, STRIKES, volatility, TIME, RATE)
        assert np.all(call >= discount * np.maximum(FORWARD - STRIKES, 0.0) - 1e-9)
        assert np.all(call <= discount * FORWARD + 1e-9)


def test_zero_volatility_gives_discounted_intrinsic():
    discount = np.exp(-RATE * TIME)
    call = black_call_price(FORWARD, STRIKES, 0.0, TIME, RATE)
    assert np.allclose(call, discount * np.maximum(FORWARD - STRIKES, 0.0))


def test_implied_volatility_inverts_the_pricer():
    """Round-tripping price and volatility must return the input.

    Only where the option carries time value: a quote whose price equals
    its intrinsic value to within floating-point resolution contains no
    volatility information at all, and the inverter is right to return
    zero there rather than a fitted number.
    """
    discount = np.exp(-RATE * TIME)

    for volatility in VOLATILITIES:
        for time in (1 / 365, 30 / 365, 0.5, 2.0):
            price = black_call_price(FORWARD, STRIKES, volatility, time, RATE)
            intrinsic = np.exp(-RATE * time) * np.maximum(FORWARD - STRIKES, 0.0)

            informative = (price - intrinsic) > 1e-8 * discount * FORWARD
            if not informative.any():
                continue

            recovered = implied_volatility_call(price, FORWARD, STRIKES, time, RATE)
            assert np.allclose(recovered[informative], volatility, atol=1e-7)


def test_uninformative_quotes_return_zero_not_a_fitted_number():
    """Deep wings priced at intrinsic must not be given a volatility."""
    far = 40_000.0
    price = black_call_price(FORWARD, far, 0.15, 1 / 365, RATE)
    assert price == 0.0
    assert implied_volatility_call(price, FORWARD, far, 1 / 365, RATE) == 0.0


def test_implied_volatility_accepts_scalars():
    price = black_call_price(FORWARD, 4200.0, 0.2, TIME, RATE)
    recovered = implied_volatility_call(price, FORWARD, 4200.0, TIME, RATE)
    assert isinstance(recovered, float)
    assert recovered == pytest.approx(0.2, abs=1e-8)


def test_implied_volatility_rejects_impossible_prices():
    discount = np.exp(-RATE * TIME)

    below_intrinsic = 0.9 * discount * (FORWARD - 3600.0)
    assert np.isnan(implied_volatility_call(below_intrinsic, FORWARD, 3600.0, TIME, RATE))

    above_forward = 1.01 * discount * FORWARD
    assert np.isnan(implied_volatility_call(above_forward, FORWARD, 3600.0, TIME, RATE))

    at_intrinsic = discount * (FORWARD - 3600.0)
    assert implied_volatility_call(at_intrinsic, FORWARD, 3600.0, TIME, RATE) == 0.0


def test_vega_matches_a_finite_difference():
    step = 1e-5
    for strike in STRIKES[:-1]:
        analytic = black_vega(FORWARD, strike, 0.2, TIME, RATE)
        numeric = (
            black_call_price(FORWARD, strike, 0.2 + step, TIME, RATE)
            - black_call_price(FORWARD, strike, 0.2 - step, TIME, RATE)
        ) / (2 * step)
        assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-6)
