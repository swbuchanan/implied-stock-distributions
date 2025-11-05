from scipy.optimize import brentq
from scipy.stats import norm
import numpy as np
import datetime
from dateutil.tz import tzutc


def find_tte_yf_options(expiration_date,last_trade_date):
    '''returns time measured in years as a float between two dates
    
    Inputs:
    expiration_date (str): 'YYYY-MM-DD'
    last_trade_date (pandas._libs.tslibs.timestamps.Timestamp)
    
    Returns:
    Float of time to expiration in years
    '''
    tte = (datetime.datetime.strptime(expiration_date+'-21-30', "%Y-%m-%d-%H-%M").replace(tzinfo=tzutc()) -\
last_trade_date).total_seconds()/(60*60*24*365)
    
    return tte


##Black-Scholes Functions
def bs_call(S0, K, sigma, t, r):
    '''
    Black-Scholes Call Option formula
    
    Inputs:
    S0 (float): Stock price at time 0
    K (float): Strike Price
    sigma: Yearly volatility
    t: Time to expiration (years)
    r: Risk-free Interest rate
    
    Return:
    Black-Scholes value of call option (float)
    '''
    
    d1 = (np.log(S0/K) + (r + (0.5)*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    call_value = S0*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
    
    return call_value
    
def bs_put(S0, K, sigma, t, r):
    '''
    Black-Scholes Put Option formula
    
    Inputs:
    S0 (float): Stock price at time 0
    K (float): Strike Price
    sigma: Yearly volatility
    t: Time to expiration (years)
    r: Risk-free Interest rate
    
    Return:
    Black-Scholes value of put option (float)
    '''
    
    d1 = (np.log(S0/K) + (r + (0.5)*sigma**2)*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    put_value = -S0*norm.cdf(-d1) + K*np.exp(-r*t)*norm.cdf(-d2)
    
    return put_value

def bs_price(S0, K, sigma, t, r, option_type):
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type.")
    return bs_call(S0, K, sigma, t, r) if option_type == "call" else bs_put(S0, K, sigma, t, r)

def implied_volatility_call(market_price, S0, K, t, r, sigma_bounds=(1e-7, 20)):
    """
    Returns the implied volatility of a call option given spot price, strike, time to expiration, 
    and risk-free-interest rate.
    
    Inputs:
    market_price (float): Market price of call option
    S0 (float): Spot price of stock
    K (float): strike price
    t (float): time-to-expiration
    r (float): risk-free-interest rate
    
    Returns:
    Implied volatility (float)
    """
    def objective(sigma):
        return bs_call(S0, K, sigma, t, r) - market_price
    try:
        return brentq(objective, *sigma_bounds)
    except ValueError:
        return np.nan
    

def iv_call_v2(market_price, S0, K, t, r, q=0, vol_lo=1e-12, vol_hi=50, max_hi=2000.0, xtol=1e-12, rtol=1e-12, maxiter=200):

    # 2) Define the objective
    def _err(sig):
        return bs_call(S0, K, sig, t, r) - market_price

    # 3) Make sure the bracket straddles zero; expand vol_hi if needed
    f_lo = _err(vol_lo)
    f_hi = _err(vol_hi)
    while f_lo * f_hi > 0 and vol_hi < max_hi:
        vol_hi *= 2.0
        f_hi = _err(vol_hi)

    if f_lo * f_hi > 0:
        # This happens for prices extremely close to bounds (numerical issues) or mismatched model.
        raise ValueError(
            "Could not bracket the root for implied vol. "
            f"err({vol_lo})={f_lo:.6e}, err({vol_hi})={f_hi:.6e}. "
            "The price may be too close to a bound, or inputs (r/q/option type) may be inconsistent."
        )

    # 4) Root find
    return brentq(_err, vol_lo, vol_hi, xtol=xtol, rtol=rtol, maxiter=maxiter)
    
def implied_volatility_put(market_price, S0, K, t, r, sigma_bounds=(1e-6, 2)):
    """
    Returns the implied volatility of a put option given spot price, strike, time to expiration, 
    and risk-free-interest rate.
    
    Inputs:
    market_price (float): Market price of call option
    S0 (float): Spot price of stock
    K (float): strike price
    t (float): time-to-expiration
    r (float): risk-free-interest rate
    
    Returns:
    Implied volatility (float)
    """
    def objective(sigma):
        return bs_put(S0, K, sigma, t, r) - market_price
    try:
        return brentq(objective, *sigma_bounds)
    except ValueError:
        return np.nan


def bs_implied_volatility(option_type, market_price, S0, K, t, r, sigma_bounds=(1e-6, 2)):
    """
    Simple wrapper to use the other functions to calculate implied volatility for either a call or a put.
    """
    if option_type not in ["call", "put"]:
        raise ValueError("Invalid option type.")

    if option_type == "call":
        return implied_volatility_call(market_price, S0, K, t, r, sigma_bounds)
    elif option_type == "put":
        return implied_volatility_put(market_price, S0, K, t, r, sigma_bounds)
