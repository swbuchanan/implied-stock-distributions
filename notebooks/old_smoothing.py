from scipy.interpolate import make_splrep, interp1d

def compute_implied_vols(strikes, prices, spot, T, option_type, r=0.0):
    """
    Apply the Black-Scholes formula to the given arrays of strikes and option prices.
    Return the strikes and the corresponding implied volatilities.
    """
    strikes = np.asarray(strikes)
    prices = np.asarray(prices)

    iv_list = []
    strikes_list = []

    for k, c in zip(strikes, prices):
        try:
            iv = bs_implied_volatility(
                option_type=option_type,
                market_price=c,
                S0=spot,
                K=k,
                t=T,
                r=r,
            )
        except Exception:
            continue

        if iv is None or not np.isfinite(iv):
            continue

        strikes_list.append(k)
        iv_list.append(iv)

    strikes_valid = np.array(strikes_list)
    iv_valid = np.array(iv_list)

    order = np.argsort(strikes_valid)
    return strikes_valid[order], iv_valid[order]

def fit_iv_logmoneyness(strikes, iv, forward, smoothing_factor=None):
    """
    Fit a smooth implied volatility curve σ(K) as a function of log-moneyness.

    The function transforms strikes to log-moneyness coordinates
        lm = log(K / F)
    where F is the forward price, and fits a cubic smoothing spline
    to log(σ(lm)) rather than σ(lm) directly. This guarantees positive
    volatility values after exponentiation and yields a smooth, stable curve.

    Parameters
    ----------
    strikes : array-like
        Strike prices at which implied volatilities are observed.
    iv : array-like
        Observed implied volatilities (in decimals, not percentages)
        corresponding to `strikes`.
    forward : float
        Forward price of the underlying asset, typically computed as
        `F = S * exp(r * T)`.
    smoothing_factor : float, optional
        Spline smoothing parameter passed to `scipy.interpolate.make_splrep`.
        Larger values produce a smoother curve at the cost of fidelity to data.
        If None, a small default proportional to the number of data points
        (0.01 * n) is used.

    Returns
    -------
    iv_of_strike : callable
        Function mapping a strike `K` to its smoothed implied volatility.
        This function evaluates the spline in log-moneyness space internally.
    K_grid : ndarray
        Uniformly spaced grid of strike values spanning the fitted domain.

    Notes
    -----
    - Fitting in log-moneyness space provides better numerical stability and
      more natural curvature around the at-the-money region.
    - The spline is fit to log(σ) to enforce strictly positive volatilities.
    - The resulting curve is smooth and differentiable, suitable for subsequent
      computation of option prices and risk-neutral densities.
    """
    strikes = np.asarray(strikes)
    iv = np.asarray(iv)

    lm_vals = np.log(strikes/forward)

    iv = np.maximum(iv, 1e-4)
    log_iv = np.log(iv)

    # sort by lm for a well-behaved spline
    order = np.argsort(lm_vals)
    lm_sorted = lm_vals[order]
    log_iv_sorted = log_iv[order]
    strikes_sorted = strikes[order]

    if smoothing_factor is None:
        n = len(lm_vals)
        smoothing_factor = 0.05 * n

    # spline in log-moneyness space
    spline_log_iv = make_splrep(
        lm_sorted,
        log_iv_sorted,
        k=3,
        s=smoothing_factor,
    )

    # create a function of strike rather than log-moneyness
    def iv_of_strike(K):
        K = np.asarray(K)
        lm = np.log(K / forward)
        log_iv_val = spline_log_iv(lm)      # spline is defined in lm-space
        return np.exp(log_iv_val)     

    K_grid = np.linspace(strikes_sorted.min(), strikes_sorted.max(), 500)

    return iv_of_strike, K_grid
    

def smooth_option_curve(strikes, option_prices, spot, T, option_type, r=0, forward=None, n_grid=500, smoothing_factor=None):
    """
    Construct a smooth option price curve as a function of strike by
    interpolating and smoothing the implied volatility surface.

    This function performs the following sequence:
        1. Sorts and linearly interpolates the raw option prices across strikes
           to fill in missing or unevenly spaced data.
        2. Computes implied volatilities at each (interpolated) strike.
        3. Fits a smooth implied volatility function σ(K) in log-moneyness space.
        4. Converts the smoothed implied volatility curve back into
           option prices using the Black–Scholes model.

    Parameters
    ----------
    strikes : array-like
        Array of strike prices corresponding to observed option prices.
    option_prices : array-like
        Array of option mid prices (call or put) corresponding to `strikes`.
    spot : float
        Current spot price of the underlying asset.
    T : float
        Time to expiration in years.
    option_type : {'call', 'put'}
        Type of option to process.
    r : float, optional
        Risk-free rate used for discounting. Default is 0.
    forward : float, optional
        Forward price of the underlying. If None, computed as
        `spot * exp(r * T)`.
    n_grid : int, optional
        Number of points to evaluate in the smoothed grid.
        (Currently unused if using log-moneyness smoothing.)
    smoothing_factor : float, optional
        Spline smoothing parameter controlling the degree of smoothness
        in the implied volatility fit. If None, a small default is chosen
        automatically.

    Returns
    -------
    strike_grid : ndarray
        Array of strike prices corresponding to the smoothed curve.
    price_smooth : ndarray
        Array of option prices implied by the smoothed IV curve.
    iv_of_strike : callable
        Function mapping a strike price `K` to its fitted implied volatility.

    Notes
    -----
    - Linear interpolation is used to fill gaps in the (strike, price) data,
      assuming near-linearity of prices away from the money.
    - The smoothing step is done in *log-moneyness* space to ensure
      stability of numerical differentiation when later constructing PDFs.
    - The output price curve is smooth and differentiable, suitable for
      applying the Breeden–Litzenberger method to extract implied densities.
    """
    strikes = np.asarray(strikes)
    option_prices = np.asarray(option_prices)

    # in a few dataframes there are large gaps in the (strikes, option_prices) curve
    # I think linear interpolation should be fine to fill these in
    # the point is that away from the spot price the graph is basically linear,
    # and this process should be stable under perturbations in price space anyway

    order = np.argsort(strikes)
    strikes = strikes[order]
    option_prices = option_prices[order]

    diffs = np.diff(strikes)
    typical_step = np.median(diffs)

    # Define a uniform strike grid between min and max
    strike_fine = np.arange(strikes.min(), strikes.max() + typical_step, typical_step)

    # linearly interpolate option prices
    interp_func = interp1d(strikes, option_prices, kind='linear', fill_value="extrapolate")
    option_prices_interp = interp_func(strike_fine)

    strikes = strike_fine
    option_prices = option_prices_interp
 
    # calculate iv on observed strikes (not exactly observed anymore since we interpolated)
    strikes_obs, iv_obs = compute_implied_vols(
        strikes=strikes,
        prices=option_prices,
        spot=spot,
        T=T,
        option_type=option_type,
        r=r,
    )

    if forward is None:
        forward = spot * np.exp(r * T)

    # fit implied volatility as a function of log-moneyness
    iv_of_strike, strike_grid = fit_iv_logmoneyness(
        strikes=strikes_obs,
        iv=iv_obs,
        forward=forward,
        smoothing_factor=smoothing_factor
    )

    # convert back to prices via Black–Scholes
    iv_on_grid = iv_of_strike(strike_grid)

    price_smooth = np.array([
        bs_price(
            S0=spot,
            K=strike_grid[i],
            sigma=iv_on_grid[i],
            t=T,
            r=r,
            option_type=option_type,
        )
        for i in range(len(strike_grid))
    ])
    
    
    return strike_grid, price_smooth, iv_of_strike

def create_pdf_from_df(df: pd.DataFrame, option_type):

    # I think we can infer the risk-free rate from the option chain using put-call parity
    # TODO: do this

    strike_grid, price_smooth, iv_of_strike = smooth_option_curve(
        strikes = df['strike'],
        option_prices = df[f'{option_type}_mid'],
        spot = df['underlying_last'].mean(), # this column should be constant but we'll take the mean anyway
        T = df['tte_years'].iloc[0],
        option_type = option_type,
    )
    
    f = np.gradient(price_smooth, strike_grid)
    f_2 = np.gradient(f, strike_grid)


    # ensure nonnegative and normalize
    # TODO: this is a hack to ensure that the pdf is nonnegative, but it may not be the best way to do it. We should probably use a more sophisticated method to ensure that the pdf is nonnegative and normalized.
    # should be something based on no-arbitrage and put-call parity
    f_2 = np.clip(f_2, 0, None)
    area = np.trapezoid(f_2, strike_grid)
    if area > 0:
        f_2 /= area

    return strike_grid, price_smooth, CubicSpline(strike_grid, f_2)
