"""Inference for forecast comparisons, and calibration diagnostics.

The sample here is far less informative than its row count suggests. Every
expiration is forecast on roughly twenty-one consecutive trading days, and
the thirty-, sixty- and ninety-day chains observed on one date all resolve
against SPX levels weeks apart but from overlapping information. Treating
several thousand chains as independent would make any difference in mean
score look decisive. Two corrections are provided:

* a Diebold-Mariano test on the score differential with a Newey-West
  variance, which is the standard treatment for overlapping forecasts; and
* a block bootstrap that resamples whole expirations, which makes no
  assumption about the correlation structure inside a block.

Two counting traps live in this module, and both changed a headline number.

**The HAC lag is in rows, not days.** The score table has one row per
``(observation date, horizon)``, so with three horizons a lag of sixty-three
*rows* is twenty-one trading days, not sixty-three. The ninety-day arm needs
189 rows. Use ``pooled_hac_lag`` rather than passing a horizon lag straight
through; ``diebold_mariano`` reports the units it was given so the mistake
cannot hide.

**"One observation per expiration and horizon" is still three observations
per outcome.** All three horizons of one expiration are scored against the
same SET print, so their PITs are strongly dependent and a KS or chi-square
p-value computed across them is anti-conservative by roughly ``sqrt(3)``.
The number that governs the whole project is the count of distinct
expirations -- fourteen analysis years at twelve monthly expirations each,
so of order 170. ``one_per_expiration`` builds that panel, and
``effective_sample_size`` reports the count.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import BOOTSTRAP_RANDOM_STATE, BOOTSTRAP_REPLICATES, DM_LAG_BY_HORIZON

__all__ = [
    "newey_west_variance",
    "diebold_mariano",
    "rows_per_observation_date",
    "pooled_hac_lag",
    "block_bootstrap_mean",
    "nearest_to_target",
    "one_per_expiration",
    "effective_sample_size",
    "pit_uniformity",
    "skill_score",
]


def newey_west_variance(values, lag: int) -> float:
    """Long-run variance of the mean of a serially correlated series.

    Bartlett kernel, following Newey and West (1987). ``lag`` counts
    *observations in the series supplied*, not calendar or trading days.
    For a series with one row per day and an h-day forecast horizon that is
    roughly h; for a series pooling three horizons per day it is three
    times that. See ``pooled_hac_lag``.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = values.size
    if n < 2:
        return float("nan")

    lag = int(max(0, min(lag, n - 1)))
    centred = values - values.mean()

    variance = float(centred @ centred) / n

    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        covariance = float(centred[k:] @ centred[:-k]) / n
        variance += 2.0 * weight * covariance

    return variance


def rows_per_observation_date(frame: pd.DataFrame, date_column: str = "data_date") -> float:
    """Average number of rows the score table carries per observation date.

    This is the factor converting a horizon lag in trading days into a lag
    in rows of the pooled series.
    """
    n_dates = int(frame[date_column].nunique())
    if n_dates == 0:
        raise ValueError("No observation dates found.")
    return len(frame) / n_dates


def pooled_hac_lag(
    frame: pd.DataFrame,
    lag_by_horizon: dict | None = None,
    date_column: str = "data_date",
    horizon_column: str = "target_dte_days",
) -> int:
    """Newey-West lag, in rows, for a score table pooled across horizons.

    The longest overlap in the pooled series is the longest horizon's, and
    each trading day contributes as many rows as there are horizons, so the
    lag in rows is the longest horizon lag times the rows per date.

    Passing ``max(DM_LAG_BY_HORIZON.values())`` directly -- as the
    conclusion cell did -- sets the window to about a third of what it
    should be, which understates the standard error on the headline
    comparison.
    """
    lag_by_horizon = lag_by_horizon or DM_LAG_BY_HORIZON

    horizons = frame[horizon_column].dropna().unique()
    horizon_lags = [
        lag_by_horizon[int(h)] for h in horizons if int(h) in lag_by_horizon
    ]
    if not horizon_lags:
        raise ValueError("No horizon in the frame appears in lag_by_horizon.")

    per_date = rows_per_observation_date(frame, date_column)
    return int(np.ceil(max(horizon_lags) * per_date))


def diebold_mariano(
    loss_a,
    loss_b,
    lag: int,
    names=("a", "b"),
    lag_units: str = "rows",
) -> dict:
    """Test whether two forecasts have equal expected loss.

    Positive ``mean_difference`` means ``loss_a`` exceeds ``loss_b``, so
    forecast ``b`` is the better one. The test statistic is the mean of the
    loss differential over its Newey-West standard error, compared against
    a normal.

    ``lag`` is in rows of the series passed in. ``lag_units`` is carried
    into the result purely so that a caller who meant days and passed days
    can see it in the output table.
    """
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)

    if loss_a.shape != loss_b.shape:
        raise ValueError("Loss series must be aligned and the same length.")

    differential = loss_a - loss_b
    differential = differential[np.isfinite(differential)]

    n = differential.size
    if n < 3:
        raise ValueError("Too few paired observations for a test.")

    if lag >= n:
        warnings.warn(
            f"Newey-West lag {lag} is not smaller than the sample size {n}; "
            "the long-run variance is not identified.",
            stacklevel=2,
        )
    elif lag > n // 4:
        warnings.warn(
            f"Newey-West lag {lag} exceeds a quarter of the sample size {n}; "
            "the variance estimate will be noisy.",
            stacklevel=2,
        )

    mean_difference = float(differential.mean())
    variance = newey_west_variance(differential, lag)

    standard_error = float(np.sqrt(max(variance, 0.0) / n))
    statistic = mean_difference / standard_error if standard_error > 0 else np.nan
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(statistic))) if np.isfinite(statistic) else np.nan

    naive_error = float(differential.std(ddof=1) / np.sqrt(n))

    return {
        "names": names,
        "n_observations": int(n),
        "mean_difference": mean_difference,
        "newey_west_standard_error": standard_error,
        "iid_standard_error": naive_error,
        "variance_inflation": (
            standard_error / naive_error if naive_error > 0 else np.nan
        ),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "lag": int(lag),
        "lag_units": lag_units,
        "better": names[1] if mean_difference > 0 else names[0],
    }


def block_bootstrap_mean(
    values,
    blocks,
    replicates: int = BOOTSTRAP_REPLICATES,
    confidence: float = 0.95,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> dict:
    """Bootstrap the mean of ``values`` by resampling whole blocks.

    ``blocks`` labels the group each observation belongs to -- expiration
    month, here -- and blocks are drawn with replacement, so any correlation
    within a block is preserved rather than modelled. This makes no
    assumption about the ordering of rows inside a block, which is the right
    property for a panel that overlaps in two directions at once.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype=float),
                          "block": np.asarray(blocks)})
    frame = frame.loc[np.isfinite(frame["value"])]

    grouped = [group["value"].to_numpy() for _, group in frame.groupby("block")]
    n_blocks = len(grouped)

    if n_blocks < 2:
        raise ValueError("At least two blocks are required.")

    generator = np.random.default_rng(random_state)
    means = np.empty(replicates)

    for replicate in range(replicates):
        picks = generator.integers(0, n_blocks, size=n_blocks)
        means[replicate] = np.concatenate([grouped[i] for i in picks]).mean()

    alpha = 1.0 - confidence
    lower, upper = np.quantile(means, [alpha / 2, 1.0 - alpha / 2])

    return {
        "mean": float(frame["value"].mean()),
        "standard_error": float(means.std(ddof=1)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_blocks": int(n_blocks),
        "n_observations": int(len(frame)),
    }


# --------------------------------------------------------------------------
# Reducing the overlapping panel to something that can be tested
# --------------------------------------------------------------------------

def nearest_to_target(
    frame: pd.DataFrame,
    group_columns=("expiration_date", "target_dte_days"),
    horizon_column: str = "target_dte_days",
    actual_column: str = "actual_dte",
) -> pd.DataFrame:
    """One row per group, the one whose realised DTE is nearest its target.

    The previous selection sorted by observation date and took ``.last()``,
    which systematically picks the *shortest* actual DTE in each bucket:
    the last date on which a given expiration is still the nearest one to
    the thirty-day target is the date closest to expiry. The thirty-day PIT
    sample therefore sat at the twenty-day end of a bucket documented as
    spanning twenty to forty days, and CRPS scales roughly with the square
    root of maturity, so that is not a neutral choice.

    Choosing the observation nearest the target is neutral, and it is what
    the horizon label claims.
    """
    frame = frame.copy()

    if actual_column not in frame.columns:
        frame[actual_column] = (
            pd.to_datetime(frame["expiration_date"])
            - pd.to_datetime(frame["data_date"])
        ).dt.days

    distance = (frame[actual_column] - frame[horizon_column]).abs()
    frame = frame.assign(_distance_to_target=distance)

    selected = (
        frame.sort_values(["_distance_to_target", "data_date"], kind="stable")
        .groupby(list(group_columns), as_index=False, sort=False)
        .first()
        .drop(columns="_distance_to_target")
    )

    return selected.sort_values(list(group_columns)).reset_index(drop=True)


def one_per_expiration(
    frame: pd.DataFrame,
    horizon: int | None = None,
    horizon_column: str = "target_dte_days",
    expiration_column: str = "expiration_date",
    actual_column: str = "actual_dte",
) -> pd.DataFrame:
    """The genuinely non-overlapping panel: one row per distinct expiration.

    Grouping by ``(expiration_date, target_dte_days)`` leaves three rows per
    outcome, because the 30-, 60- and 90-day forecasts of one expiration are
    all scored against the same SET print. Their PITs are strongly dependent
    and a KS or chi-square p-value across them is anti-conservative by
    roughly ``sqrt(3)`` in the standard error.

    Pass ``horizon`` to keep one horizon's forecasts -- the honest way to
    run a per-horizon calibration test, since within a horizon consecutive
    monthly expirations resolve against different prints. Passing ``None``
    keeps the observation nearest its own target across all horizons, which
    is the single-number version.
    """
    frame = frame.copy()

    if actual_column not in frame.columns:
        frame[actual_column] = (
            pd.to_datetime(frame["expiration_date"])
            - pd.to_datetime(frame["data_date"])
        ).dt.days

    if horizon is not None:
        frame = frame.loc[frame[horizon_column].astype(int).eq(int(horizon))]
        if frame.empty:
            raise ValueError(f"No observations at the {horizon}-day horizon.")

    return nearest_to_target(
        frame,
        group_columns=(expiration_column,),
        horizon_column=horizon_column,
        actual_column=actual_column,
    )


def effective_sample_size(
    frame: pd.DataFrame,
    expiration_column: str = "expiration_date",
) -> dict:
    """Row count against the count of distinct outcomes behind it.

    Worth printing early and once: everything downstream is governed by the
    number of distinct expirations, not by the number of chains.
    """
    return {
        "rows": int(len(frame)),
        "distinct_expirations": int(frame[expiration_column].nunique()),
        "rows_per_expiration": (
            len(frame) / frame[expiration_column].nunique()
            if frame[expiration_column].nunique() else float("nan")
        ),
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def pit_uniformity(
    pit_values,
    n_bins: int = 10,
    parameters_estimated: int = 0,
) -> dict:
    """Test whether probability integral transforms are uniform.

    Calibrated forecasts produce uniform PITs. Systematic departures name
    the failure: a hump on the left means the forecasts put too little mass
    low down, a U shape means they are too narrow overall, and an inverted U
    means they are too wide.

    Two conditions have to hold for the p-values to mean anything.

    *Independence.* The Kolmogorov-Smirnov and chi-square p-values assume
    independent observations, which overlapping daily forecasts are not.
    Pass a panel from ``one_per_expiration``.

    *No estimated parameters.* If the distribution being tested was tuned on
    these same observations -- as the CRRA-tilted physical density is, since
    gamma is fitted by minimising the score on them -- the null distribution
    of the KS statistic is not the tabulated one and the p-value is
    optimistic by an unknown amount. Set ``parameters_estimated`` to the
    number fitted in-sample; the p-values are then returned as ``nan`` and
    the flag is carried in the result, because a number that cannot be
    interpreted is worse than no number. The bin counts and the shape of
    the histogram remain informative either way.
    """
    values = np.asarray(pit_values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        raise ValueError("Too few PIT values.")

    ks_statistic, ks_p_value = stats.kstest(values, "uniform")

    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    expected = values.size / n_bins
    chi_square = float(np.sum((counts - expected) ** 2 / expected))
    chi_p_value = float(1.0 - stats.chi2.cdf(chi_square, df=n_bins - 1))

    in_sample = int(parameters_estimated) > 0
    if in_sample:
        ks_p_value = float("nan")
        chi_p_value = float("nan")

    return {
        "n_observations": int(values.size),
        "mean": float(values.mean()),
        "ks_statistic": float(ks_statistic),
        "ks_p_value": float(ks_p_value),
        "chi_square": chi_square,
        "chi_square_p_value": chi_p_value,
        "parameters_estimated": int(parameters_estimated),
        "p_values_valid": not in_sample,
        "bin_counts": counts,
        "bin_edges": edges,
        "expected_per_bin": float(expected),
        "share_below_10pct": float(np.mean(values < 0.1)),
        "share_above_90pct": float(np.mean(values > 0.9)),
    }


def skill_score(score, reference_score) -> float:
    """Fractional improvement over a reference forecast.

    ``1 - score / reference`` : zero means no better than the reference,
    one would mean a perfect forecast, and negative means worse.
    """
    score = float(np.mean(score))
    reference_score = float(np.mean(reference_score))

    if reference_score <= 0:
        return float("nan")

    return 1.0 - score / reference_score
