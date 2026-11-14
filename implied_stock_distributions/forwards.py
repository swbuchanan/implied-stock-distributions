"""Forward price and discount factor from put-call parity.

For European options on the same underlying and expiration,

    C(K) - P(K) = D * (F - K),

so a regression of the call-put spread on strike has slope ``-D`` and
intercept ``D * F``. Strikes are centred before the regression because SPX
strikes are large relative to the spread being fitted.

Three details matter for accuracy:

* The fit is restricted to a band around the money. Parity holds at every
  strike in theory, but deep in- and out-of-the-money quotes are stale and
  wide, and because they sit far from the centre they lever the slope --
  which is the discount factor itself. With a thirty-day expiry a tenth of
  a percent of error in ``D`` is more than a percentage point of error in
  the implied rate.
* Quality thresholds are enforced rather than merely reported, so that a
  chain whose parity fit is not credible raises instead of silently
  contaminating downstream densities.
* The thresholds include ``fraction_outside_spreads`` and not only ``R^2``.
  On a relationship this close to deterministic, ``R^2 > 0.999`` is a weak
  gate: the denominator is the spread of the call-put difference across
  strikes, which is enormous, so a fit can clear it while missing the
  quoted market at a quarter of the strikes. The share of pairs whose
  parity residual exceeds the sum of the call and put half-spreads is scale-free, is the
  number an options market maker would ask for, and was already being
  computed in the notebook's diagnostic sweep and then discarded.

A note for the diagnostics sweep: run it with the same ``enforce_quality``
setting the scoring loop uses, or the failure table describes a different
sample from the one the results come from. The sweep should call this with
``enforce_quality=False`` *and* apply the gates afterwards from the
returned ``passes_quality`` flag, so that one pass produces both the
failure table and the analysis sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    IMPLIED_RATE_BOUNDS,
    MAX_FRACTION_OUTSIDE_SPREADS,
    MIN_PARITY_PAIRS,
    MIN_PARITY_R_SQUARED,
    PARITY_MONEYNESS_BAND,
    PARITY_USE_SPREAD_WEIGHTS,
)
from .errors import ChainQualityError, InsufficientQuotesError

__all__ = [
    "estimate_forward_and_rate",
    "matched_parity_pairs",
    "quality_failures",
]


def matched_parity_pairs(chain: pd.DataFrame) -> pd.DataFrame:
    """Return call-put pairs sharing a strike, with their quote spreads."""

    quotes = chain.copy()
    quotes["quote_spread"] = quotes["ask_price"] - quotes["bid_price"]

    if "quote_is_two_sided" in quotes.columns:
        quotes = quotes.loc[quotes["quote_is_two_sided"].fillna(False)]
    else:
        quotes = quotes.loc[
            (quotes["bid_price"] > 0)
            & (quotes["ask_price"] >= quotes["bid_price"])
        ]

    columns = ["strike_price", "mid_price", "quote_spread"]

    calls = quotes.loc[quotes["put_call"].eq("call"), columns].rename(
        columns={"mid_price": "call_mid", "quote_spread": "call_spread"}
    )
    puts = quotes.loc[quotes["put_call"].eq("put"), columns].rename(
        columns={"mid_price": "put_mid", "quote_spread": "put_spread"}
    )

    pairs = calls.merge(
        puts, on="strike_price", how="inner", validate="one_to_one"
    )

    pairs = pairs.dropna(subset=["strike_price", "call_mid", "put_mid"])

    finite = np.isfinite(
        pairs[["strike_price", "call_mid", "put_mid"]].to_numpy(dtype=float)
    ).all(axis=1)

    return (
        pairs.loc[finite]
        .sort_values("strike_price")
        .reset_index(drop=True)
    )


def _weighted_line_fit(x, y, weights=None):
    """Weighted least-squares fit of ``y = a + b x``.

    Returns the intercept, the slope, the fitted values and the weighted
    coefficient of determination.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights, dtype=float)

    root_w = np.sqrt(weights)
    design = np.column_stack([np.ones_like(x), x])

    coefficients, *_ = np.linalg.lstsq(
        design * root_w[:, None], y * root_w, rcond=None
    )

    intercept, slope = float(coefficients[0]), float(coefficients[1])
    fitted = design @ coefficients

    residuals = y - fitted
    weighted_mean = np.average(y, weights=weights)

    ss_residual = float(np.sum(weights * residuals**2))
    ss_total = float(np.sum(weights * (y - weighted_mean) ** 2))

    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else np.nan

    return intercept, slope, fitted, r_squared


def quality_failures(estimates: dict) -> list[str]:
    """Names of the quality gates a parity fit fails, in a stable order.

    Returned rather than raised so that a diagnostic sweep can run once,
    with ``enforce_quality=False``, and still describe exactly the sample
    the scoring loop will keep. Running the sweep unenforced and the
    scoring loop enforced -- as an earlier version did -- means the failure
    table in the diagnostics section is not a description of the analysis
    sample.
    """
    lower_rate, upper_rate = IMPLIED_RATE_BOUNDS
    failures = []

    if estimates["n_pairs"] < MIN_PARITY_PAIRS:
        failures.append("too_few_parity_pairs")

    r_squared = estimates.get("r_squared", np.nan)
    if not np.isfinite(r_squared) or r_squared < MIN_PARITY_R_SQUARED:
        failures.append("low_parity_r_squared")

    outside = estimates.get("fraction_outside_spreads", np.nan)
    if np.isfinite(outside) and outside > MAX_FRACTION_OUTSIDE_SPREADS:
        failures.append("parity_fit_outside_spreads")

    rate = estimates.get("risk_free_rate", np.nan)
    if not (np.isfinite(rate) and lower_rate <= rate <= upper_rate):
        failures.append("implied_rate_out_of_bounds")

    return failures


def estimate_forward_and_rate(
    chain: pd.DataFrame,
    use_spread_weights: bool | None = None,
    moneyness_band: float | None = PARITY_MONEYNESS_BAND,
    enforce_quality: bool = True,
):
    """Estimate the discount factor, forward and implied rate for one chain.

    The chain must cover a single observation date and a single expiration.

    Parameters
    ----------
    chain : pandas.DataFrame
        Quotes for one chain. Requires ``strike_price``, ``put_call``,
        ``mid_price``, ``bid_price``, ``ask_price`` and
        ``time_to_expiry_years``.
    use_spread_weights : bool, optional
        Weight each pair by the reciprocal of the summed squared quote
        spreads, so that tighter markets count for more. Defaults to
        ``config.PARITY_USE_SPREAD_WEIGHTS``; pass a value explicitly only
        when deliberately comparing the two.
    moneyness_band : float or None, optional
        Half-width of the strike window retained for the final fit,
        expressed as a fraction of a first-pass forward estimate. ``None``
        uses every matched pair, which reproduces the earlier behaviour.
    enforce_quality : bool, optional
        Raise ``ChainQualityError`` when the fit fails the thresholds in
        ``config``. Set to ``False`` for diagnostic sweeps that want to see
        the bad fits, and read ``estimates["quality_failures"]`` to apply
        exactly the same gates afterwards.

    Returns
    -------
    tuple[dict, pandas.DataFrame]
        Estimates and the pairs actually used, with fitted values and
        residuals attached. The estimates carry ``fraction_outside_spreads``,
        ``quality_failures`` and ``passes_quality`` whether or not the gates
        are enforced.

    Raises
    ------
    InsufficientQuotesError
        Fewer than three matched pairs.
    ChainQualityError
        Non-positive discount factor, or any failed gate when
        ``enforce_quality`` is set.
    """
    if use_spread_weights is None:
        use_spread_weights = PARITY_USE_SPREAD_WEIGHTS

    pairs = matched_parity_pairs(chain)

    if len(pairs) < 3:
        raise InsufficientQuotesError(
            "At least three matched call-put pairs are required."
        )

    time_values = chain["time_to_expiry_years"].dropna().unique()
    if len(time_values) != 1:
        raise ChainQualityError(
            "Expected exactly one time-to-expiry value per chain."
        )

    time_to_expiry = float(time_values[0])
    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        raise ChainQualityError("Time to expiry must be finite and positive.")

    def fit(subset):
        centre = float(subset["strike_price"].median())
        x = subset["strike_price"].to_numpy(dtype=float) - centre
        y = (subset["call_mid"] - subset["put_mid"]).to_numpy(dtype=float)

        weights = None
        if use_spread_weights:
            variance_proxy = (
                subset["call_spread"].to_numpy(dtype=float) ** 2
                + subset["put_spread"].to_numpy(dtype=float) ** 2
            )
            positive = variance_proxy[variance_proxy > 0]
            if positive.size:
                floor = max(0.01 * float(np.median(positive)), 1e-12)
                weights = 1.0 / np.maximum(variance_proxy, floor)

        intercept, slope, fitted, r_squared = _weighted_line_fit(x, y, weights)
        return centre, x, y, intercept, slope, fitted, r_squared, weights

    # First pass on every pair, only to locate the money.
    centre, x, y, intercept, slope, fitted, r_squared, weights = fit(pairs)

    if moneyness_band is not None and -slope > 0:
        rough_forward = centre + intercept / (-slope)
        in_band = (
            (pairs["strike_price"] / rough_forward - 1.0).abs() <= moneyness_band
        )
        # Fall back to the full set if the band is too thin to fit.
        if int(in_band.sum()) >= max(3, MIN_PARITY_PAIRS):
            pairs = pairs.loc[in_band].reset_index(drop=True)
            centre, x, y, intercept, slope, fitted, r_squared, weights = fit(pairs)

    discount_factor = -slope

    if not np.isfinite(discount_factor) or discount_factor <= 0:
        raise ChainQualityError(
            f"Invalid estimated discount factor: {discount_factor}"
        )

    forward = centre + intercept / discount_factor
    risk_free_rate = -np.log(discount_factor) / time_to_expiry

    residuals = y - fitted
    if weights is None:
        rmse = float(np.sqrt(np.mean(residuals**2)))
    else:
        rmse = float(np.sqrt(np.average(residuals**2, weights=weights)))

    pairs = pairs.copy()
    pairs["call_minus_put"] = y
    pairs["fitted_call_minus_put"] = fitted
    pairs["parity_residual"] = residuals

    # The scale-free measure of fit: how often the line misses the market
    # by more than the market's own quoted uncertainty. Each mid is known to
    # within its half-spread, so call-minus-put is known to within the SUM
    # of the two half-spreads.
    parity_tolerance = (
        pairs["call_spread"].to_numpy(dtype=float)
        + pairs["put_spread"].to_numpy(dtype=float)
    ) / 2.0
    with np.errstate(invalid="ignore"):
        outside = np.abs(residuals) > parity_tolerance
    fraction_outside_spreads = float(np.mean(outside)) if len(residuals) else np.nan

    estimates = {
        "discount_factor": float(discount_factor),
        "forward": float(forward),
        "risk_free_rate": float(risk_free_rate),
        "time_to_expiry": time_to_expiry,
        "center_strike": centre,
        "n_pairs": int(len(pairs)),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "median_absolute_residual": float(np.median(np.abs(residuals))),
        "fraction_outside_spreads": fraction_outside_spreads,
        "used_spread_weights": bool(use_spread_weights),
        "moneyness_band": moneyness_band,
    }

    failures = quality_failures(estimates)
    estimates["quality_failures"] = tuple(failures)
    estimates["passes_quality"] = not failures

    if enforce_quality and failures:
        raise ChainQualityError(
            "Parity fit failed: "
            + ", ".join(failures)
            + f" (n_pairs={estimates['n_pairs']}, "
            f"R^2={estimates['r_squared']:.6f}, "
            f"outside spreads={fraction_outside_spreads:.3f}, "
            f"rate={risk_free_rate:.4f})",
            reason=failures[0],
        )

    return estimates, pairs
