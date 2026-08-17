from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def estimate_forward_and_rate(
    chain: pd.DataFrame,
    use_spread_weights: bool = False,
):
    """
    Estimate the discount factor, forward and continuously compounded
    interest rate from matched call-put pairs.

    if use_spread_weights is enabled, we do a weighted regression,
    with more weight given to quotes with a narrower spread,
    as these tend to be more accurate

    Uses the put-call parity relationship

        C(K) - P(K) = D(F - K).

    A separate estimate should be calculated for each exact expiration.
    """

    quotes = chain.copy()

    if "mid_price" not in quotes.columns:
        quotes["mid_price"] = (
            quotes["bid_price"] + quotes["ask_price"]
        ) / 2

    quotes["quote_spread"] = (
        quotes["ask_price"] - quotes["bid_price"]
    )

    # Prefer matched pairs for which both quotes are two-sided.
    if "quote_is_two_sided" in quotes.columns:
        quotes = quotes.loc[
            quotes["quote_is_two_sided"].fillna(False)
        ]

    calls = (
        quotes.loc[
            quotes["put_call"].eq("call"),
            ["strike_price", "mid_price", "quote_spread"],
        ]
        .rename(
            columns={
                "mid_price": "call_mid",
                "quote_spread": "call_spread",
            }
        )
    )

    puts = (
        quotes.loc[
            quotes["put_call"].eq("put"),
            ["strike_price", "mid_price", "quote_spread"],
        ]
        .rename(
            columns={
                "mid_price": "put_mid",
                "quote_spread": "put_spread",
            }
        )
    )

    # This explicitly matches calls and puts by strike.
    # validate="one_to_one" also detects unexpected duplicates.
    pairs = calls.merge(
        puts,
        on="strike_price",
        how="inner",
        validate="one_to_one",
    )

    pairs = pairs.dropna(
        subset=["strike_price", "call_mid", "put_mid"]
    )

    finite = np.isfinite(
        pairs[
            ["strike_price", "call_mid", "put_mid"]
        ].to_numpy(dtype=float)
    ).all(axis=1)

    pairs = pairs.loc[finite].sort_values(
        "strike_price"
    ).reset_index(drop=True)

    if len(pairs) < 3:
        raise ValueError(
            "At least three matched call-put pairs are required."
        )

    time_values = (
        # ((chain["expiration_date"] - chain["data_date"]).dt.days / 365.25)
        chain["time_to_expiry_years"]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(time_values) == 0:
        raise ValueError("Time to expiry is missing.")

    if not np.allclose(time_values, time_values[0]):
        raise ValueError(
            "Expected one time-to-expiry value per chain."
        )

    time_to_expiry = float(time_values[0])

    # Centering improves numerical stability because SPX strikes are large.
    center_strike = float(pairs["strike_price"].median())

    X = (
        pairs["strike_price"].to_numpy(dtype=float)
        - center_strike
    ).reshape(-1, 1)

    y = (
        pairs["call_mid"] - pairs["put_mid"]
    ).to_numpy(dtype=float)

    sample_weight = None

    if use_spread_weights:
        variance_proxy = (
            pairs["call_spread"].to_numpy(dtype=float) ** 2
            + pairs["put_spread"].to_numpy(dtype=float) ** 2
        )

        positive = variance_proxy[variance_proxy > 0]

        if len(positive) > 0:
            floor = max(
                0.01 * np.median(positive),
                1e-12,
            )
            sample_weight = 1 / np.maximum(
                variance_proxy,
                floor,
            )

    model = LinearRegression()
    model.fit(
        X,
        y,
        sample_weight=sample_weight,
    )

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Because strikes were centered:
    # C - P = D(F - K0) - D(K - K0)
    discount_factor = -slope

    if not np.isfinite(discount_factor) or discount_factor <= 0:
        raise ValueError(
            f"Invalid estimated discount factor: {discount_factor}"
        )

    forward = (
        center_strike
        + intercept / discount_factor
    )

    risk_free_rate = (
        -np.log(discount_factor) / time_to_expiry
    )

    fitted = model.predict(X)
    residuals = y - fitted

    if sample_weight is None:
        rmse = np.sqrt(np.mean(residuals**2))
    else:
        rmse = np.sqrt(
            np.average(
                residuals**2,
                weights=sample_weight,
            )
        )

    pairs["call_minus_put"] = y
    pairs["fitted_call_minus_put"] = fitted
    pairs["parity_residual"] = residuals

    estimates = {
        "discount_factor": discount_factor,
        "forward": forward,
        "risk_free_rate": risk_free_rate,
        "time_to_expiry": time_to_expiry,
        "center_strike": center_strike,
        "n_pairs": len(pairs),
        "r_squared": model.score(
            X,
            y,
            sample_weight=sample_weight,
        ),
        "rmse": rmse,
    }

    return estimates, pairs
