import numpy as np
from scipy.interpolate import CubicSpline, make_splrep


from .pricing import implied_volatility_call, black_call_price, black_put_price



def compute_implied_vols(
    strikes,
    prices,
    forward,
    time_to_expiry,
    option_type,
    risk_free_rate=0.0,
):
    """Calculate valid Black implied volatilities for observed quotes."""

    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if strikes.shape != prices.shape:
        raise ValueError("Strikes and prices must have the same shape.")

    option_type = option_type.lower()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    valid = (
        np.isfinite(strikes)
        & np.isfinite(prices)
        & (strikes > 0)
        & (prices >= 0)
    )

    strikes = strikes[valid]
    prices = prices[valid]

    # Convert puts to parity-equivalent call prices so that the existing
    # call IV inverter can be reused:
    #
    # C = P + D(F - K)
    if option_type == "put":
        discount_factor = np.exp(
            -risk_free_rate * time_to_expiry
        )
        prices = prices + discount_factor * (
            forward - strikes
        )

    implied_volatilities = np.array([
        implied_volatility_call(
            market_price=price,
            forward=forward,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
        )
        for strike, price in zip(strikes, prices)
    ])

    valid_iv = (
        np.isfinite(implied_volatilities)
        & (implied_volatilities > 0)
    )

    strikes = strikes[valid_iv]
    implied_volatilities = implied_volatilities[valid_iv]

    order = np.argsort(strikes)

    return strikes[order], implied_volatilities[order]

def fit_iv_logmoneyness(
    strikes,
    implied_volatilities,
    forward,
    n_grid=500,
    smoothing_factor=None,
):
    """
    Fit log(implied volatility) as a smooth function of log-moneyness.

    Log-moneyness is defined as log(K / F). Fitting log volatility ensures
    that the resulting fitted volatilities remain positive.
    """

    strikes = np.asarray(strikes, dtype=float)
    implied_volatilities = np.asarray(
        implied_volatilities,
        dtype=float,
    )

    valid = (
        np.isfinite(strikes)
        & np.isfinite(implied_volatilities)
        & (strikes > 0)
        & (implied_volatilities > 0)
    )

    strikes = strikes[valid]
    implied_volatilities = implied_volatilities[valid]

    if forward <= 0:
        raise ValueError("Forward must be positive.")

    if len(strikes) < 4:
        raise ValueError(
            "At least four valid implied volatilities are required."
        )

    log_moneyness = np.log(strikes / forward)
    log_iv = np.log(implied_volatilities)

    order = np.argsort(log_moneyness)
    log_moneyness = log_moneyness[order]
    log_iv = log_iv[order]
    strikes = strikes[order]

    if np.any(np.diff(log_moneyness) <= 0):
        raise ValueError("Duplicate strikes found in the IV data.")

    if smoothing_factor is None:
        smoothing_factor = 0.05 * len(strikes)

    log_iv_spline = make_splrep(
        log_moneyness,
        log_iv,
        k=3,
        s=smoothing_factor,
    )

    def iv_of_strike(strike):
        strike = np.asarray(strike, dtype=float)

        if np.any(strike <= 0):
            raise ValueError("Strikes must be positive.")

        fitted_log_iv = log_iv_spline(
            np.log(strike / forward)
        )
        return np.exp(fitted_log_iv)

    # A uniform strike grid is useful for differentiating prices later.
    strike_grid = np.linspace(
        strikes.min(),
        strikes.max(),
        n_grid,
    )

    return iv_of_strike, strike_grid

def smooth_option_curve(
    strikes,
    option_prices,
    forward,
    time_to_expiry,
    option_type,
    risk_free_rate=0.0,
    n_grid=500,
    smoothing_factor=None,
):
    """Construct a smooth option-price curve through IV smoothing."""

    strikes_obs, iv_obs = compute_implied_vols(
        strikes=strikes,
        prices=option_prices,
        forward=forward,
        time_to_expiry=time_to_expiry,
        option_type=option_type,
        risk_free_rate=risk_free_rate,
    )

    iv_of_strike, strike_grid = fit_iv_logmoneyness(
        strikes=strikes_obs,
        implied_volatilities=iv_obs,
        forward=forward,
        n_grid=n_grid,
        smoothing_factor=smoothing_factor,
    )

    iv_on_grid = iv_of_strike(strike_grid)

    pricing_function = (
        black_call_price
        if option_type.lower() == "call"
        else black_put_price
    )

    price_smooth = np.array([
        pricing_function(
            forward=forward,
            strike=strike,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
        )
        for strike, volatility in zip(
            strike_grid,
            iv_on_grid,
        )
    ])

    return strike_grid, price_smooth, iv_of_strike

def create_density_from_chain(
    chain,
    forward,
    risk_free_rate,
    option_type="call",
    n_grid=500,
    smoothing_factor=None,
):
    """
    Construct a preliminary risk-neutral density from one option chain.

    Negative values are retained because they provide a diagnostic of
    smoothing or arbitrage problems.
    """

    quotes = chain.loc[
        chain["put_call"].eq(option_type)
    ].copy()

    if "quote_is_two_sided" in quotes.columns:
        quotes = quotes.loc[
            quotes["quote_is_two_sided"].fillna(False)
        ]

    time_values = (
        quotes["time_to_expiry_years"]
        .dropna()
        .unique()
    )

    if len(time_values) != 1:
        raise ValueError(
            "Expected one time-to-expiry value per chain."
        )

    time_to_expiry = float(time_values[0])

    strike_grid, price_smooth, iv_of_strike = (
        smooth_option_curve(
            strikes=quotes["strike_price"],
            option_prices=quotes["mid_price"],
            forward=forward,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            risk_free_rate=risk_free_rate,
            n_grid=n_grid,
            smoothing_factor=smoothing_factor,
        )
    )

    # Differentiate the reconstructed smooth price curve.
    price_curve = CubicSpline(
        strike_grid,
        price_smooth,
        extrapolate=False,
    )
    second_derivative = price_curve.derivative(2)(
        strike_grid
    )

    # Breeden-Litzenberger:
    # f_Q(K) = exp(rT) * d^2C/dK^2
    density = (
        np.exp(risk_free_rate * time_to_expiry)
        * second_derivative
    )

    return (
        strike_grid,
        price_smooth,
        density,
        iv_of_strike,
    )
