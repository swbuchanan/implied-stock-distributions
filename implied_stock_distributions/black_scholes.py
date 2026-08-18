from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
import datetime
from dateutil.tz import tzutc




def black_call_price(
    forward,
    strike,
    volatility,
    time_to_expiry,
    risk_free_rate,
):
    """
    Calculate the price of a European call option using Black's formula.

    This forward-based form is useful for SPX options because the inferred
    forward price already incorporates expected dividends and financing costs.

    Currently this is set up to take floats only, but I should check again later
    to see if we could get some improvement by vectorizing.

    Parameters
    ----------
    forward : float
        Forward price of the underlying for the given expiration.
    strike : float
        Option strike price.
    volatility : float
        Annualized implied volatility as a decimal, e.g. 0.20 for 20%.
    time_to_expiry : float
        Time until expiration in years.
    risk_free_rate : float
        The risk-free interest rate.

    Returns
    -------
    float
        Present value of the European call option.

    Notes
    -----
    The pricing formula is

        C = D * [F * N(d1) - K * N(d2)]

    where D is the discount factor, F is the forward price, K is strike,
    N is a normal distribution, and d1, d2 are computed below.
    We assume that interest is continuously compounded when calculating
    the discount factor
    """

    if forward <= 0 or strike <= 0:
        raise ValueError("Forward and strike must be positive.")
    if volatility < 0:
        raise ValueError("Volatility must be non-negative.")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive.")

    volatility_sqrt_t = volatility * np.sqrt(time_to_expiry)
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)

    if volatility == 0:
        return discount_factor * max(forward - strike, 0.0)

    d1 = (
        np.log(forward / strike)
        + 0.5 * volatility**2 * time_to_expiry
    ) / volatility_sqrt_t
    d2 = d1 - volatility_sqrt_t

    return discount_factor * (
        forward * norm.cdf(d1)
        - strike * norm.cdf(d2)
    )

def black_put_price(
    forward,
    strike,
    volatility,
    time_to_expiry,
    risk_free_rate
):
    """
    Calculate the price of a European put option using Black's formula.

    This forward-based form is useful for SPX options because the inferred
    forward price already incorporates expected dividends and financing costs.

    Currently this is set up to take floats only, but I should check again later
    to see if we could get some improvement by vectorizing.

    Parameters
    ----------
    forward : float
        Forward price of the underlying for the given expiration.
    strike : float
        Option strike price.
    volatility : float
        Annualized implied volatility as a decimal, e.g. 0.20 for 20%.
    time_to_expiry : float
        Time until expiration in years.
    risk_free_rate : float
        The risk-free interest rate.

    Returns
    -------
    float
        Present value of the European put option.

    Notes
    -----
    The pricing formula is

        P = D * [K * N(-d2) - F * N(-d1)]

    where D is the discount factor, F is the forward price, K is strike,
    N is a normal distribution, and d1, d2 are computed below.
    We assume that interest is continuously compounded when calculating
    the discount factor
    """

    if forward <= 0 or strike <= 0:
        raise ValueError("Forward and strike must be positive.")
    if volatility < 0:
        raise ValueError("Volatility must be non-negative.")
    if time_to_expiry <= 0:
        raise ValueError("Time to expiry must be positive.")

    volatility_sqrt_t = volatility * np.sqrt(time_to_expiry)
    discount_factor = np.exp(-risk_free_rate * time_to_expiry)

    if volatility == 0:
        return discount_factor * max(strike - forward, 0.0)


    d1 = (
        np.log(forward / strike)
        + 0.5 * volatility**2 * time_to_expiry
    ) / volatility_sqrt_t
    d2 = d1 - volatility_sqrt_t

    return discount_factor * (
        strike * norm.cdf(-d2)
        - forward * norm.cdf(-d1)
    )

        
def implied_volatility_call(
    market_price,
    forward,
    strike,
    time_to_expiry,
    risk_free_rate,
    volatility_bounds=(1e-7, 20.0),
):
    """
    Calculate the Black implied volatility of a European call option.
    Uses brentq for numerical inversion.

    Parameters
    ----------
    market_price : float
        Observed call price, normally the bid-ask midpoint.
    forward : float
        Market-implied forward level for the option's expiration.
    strike : float
        Option strike price.
    time_to_expiry : float
        Time until expiration in years.
    risk_free_rate : float
        Annualized continuously compounded risk-free rate, expressed as a
        decimal; for example, 0.05 represents 5%.
    volatility_bounds : tuple[float, float], optional
        Lower and upper volatility bounds used by the root finder.

    Returns
    -------
    float
        Annualized implied volatility. Returns ``np.nan`` if the inputs
        violate Black's price bounds or the volatility is not bracketed.
    """
    inputs = [
        market_price,
        forward,
        strike,
        time_to_expiry,
        risk_free_rate,
    ]

    if not np.all(np.isfinite(inputs)):
        return np.nan

    if forward <= 0 or strike <= 0 or time_to_expiry <= 0:
        return np.nan

    discount_factor = np.exp(
        -risk_free_rate * time_to_expiry
    )

    minimum_price = discount_factor * max(forward - strike, 0.0)
    maximum_price = discount_factor * forward

    if market_price < minimum_price or market_price >= maximum_price:
        return np.nan

    if np.isclose(market_price, minimum_price, atol=1e-12, rtol=0):
        return 0.0

    def objective(volatility):
        return black_call_price(
            forward=forward,
            strike=strike,
            volatility=volatility,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
        ) - market_price

    lower_volatility, upper_volatility = volatility_bounds

    if not 0 < lower_volatility < upper_volatility:
        raise ValueError(
            "Volatility bounds must satisfy 0 < lower < upper."
        )

    if objective(lower_volatility) > 0:
        return np.nan

    if objective(upper_volatility) < 0:
        return np.nan

    return brentq(
        objective,
        lower_volatility,
        upper_volatility,
        xtol=1e-10,
        maxiter=100,
    )


# =========== ZONE OF DEPRECATION ===================== #



# def find_tte_yf_options(
#         expiration_date,
#         last_trade_date
#     ):
#     '''returns time measured in years as a float between two dates
    
#     Inputs:
#     expiration_date (str): 'YYYY-MM-DD'
#     last_trade_date (pandas._libs.tslibs.timestamps.Timestamp)
    
#     Returns:
#     Float of time to expiration in years
#     '''
#     tte = (datetime.datetime.strptime(expiration_date+'-21-30', "%Y-%m-%d-%H-%M").replace(tzinfo=tzutc()) -\
#         last_trade_date).total_seconds()/(60*60*24*365)
    
#     return tte


##Black-Scholes Functions
# def bs_call(S0, K, sigma, t, r):
#     '''
#     Black-Scholes Call Option formula
    
#     Inputs:
#     S0 (float): Stock price at time 0
#     K (float): Strike Price
#     sigma: Yearly volatility
#     t: Time to expiration (years)
#     r: Risk-free Interest rate
    
#     Return:
#     Black-Scholes value of call option (float)
#     '''
    
#     d1 = (np.log(S0/K) + (r + (0.5)*sigma**2)*t)/(sigma*np.sqrt(t))
#     d2 = d1 - sigma*np.sqrt(t)
#     call_value = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    
#     return call_value


# def bs_put(S0, K, sigma, t, r):
#     '''
#     Black-Scholes Put Option formula
    
#     Inputs:
#     S0 (float): Stock price at time 0
#     K (float): Strike Price
#     sigma: Yearly volatility
#     t: Time to expiration (years)
#     r: Risk-free Interest rate
    
#     Return:
#     Black-Scholes value of put option (float)
#     '''
    
#     d1 = (np.log(S0/K) + (r + (0.5)*sigma**2)*t)/(sigma*np.sqrt(t))
#     d2 = d1 - sigma*np.sqrt(t)
#     put_value = -S0*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    
#     return put_value

# def bs_price(S0, K, sigma, t, r, option_type):
#     if option_type not in ["call", "put"]:
#         raise ValueError("Invalid option type.")
#     return bs_call(S0, K, sigma, t, r) if option_type == "call" else bs_put(S0, K, sigma, t, r)



# def implied_volatility_call(market_price, S0, K, t, r, sigma_bounds=(1e-7, 20)):
#     """
#     Returns the implied volatility of a call option given spot price, strike, time to expiration, 
#     and risk-free-interest rate.
    
#     Inputs:
#     market_price (float): Market price of call option
#     S0 (float): Spot price of stock
#     K (float): strike price
#     t (float): time-to-expiration
#     r (float): risk-free-interest rate
    
#     Returns:
#     Implied volatility (float)
#     """
#     def objective(sigma):
#         return bs_call(S0, K, sigma, t, r) - market_price
#     try:
#         return brentq(objective, *sigma_bounds)
#     except ValueError:
#         return np.nan
    

# def implied_volatility_put(market_price, S0, K, t, r, sigma_bounds=(1e-6, 2)):
#     """
#     Returns the implied volatility of a put option given spot price, strike, time to expiration, 
#     and risk-free-interest rate.
    
#     Inputs:
#     market_price (float):   Market price of call option
#     S0 (float):             Spot price of stock
#     K (float):              strike price
#     t (float):              time-to-expiration
#     r (float):              risk-free-interest rate
    
#     Returns:
#     Implied volatility (float)
#     """
#     def objective(sigma):
#         return bs_put(S0, K, sigma, t, r) - market_price
#     try:
#         return brentq(objective, *sigma_bounds)
#     except ValueError:
#         return np.nan


# def bs_implied_volatility(option_type, market_price, S0, K, t, r, sigma_bounds=(1e-6, 2)):
#     """
#     Simple wrapper to use the other functions to calculate implied volatility for either a call or a put.
#     """
#     if option_type not in ["call", "put"]:
#         raise ValueError("Invalid option type.")

#     if option_type == "call":
#         return implied_volatility_call(market_price, S0, K, t, r, sigma_bounds)
#     elif option_type == "put":
#         return implied_volatility_put(market_price, S0, K, t, r, sigma_bounds)
