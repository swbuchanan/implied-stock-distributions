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
so of order 170. ``effective_sample_size`` reports the count.

**One horizon per expiration is still overlapping beyond thirty days.**
The ninety-day forecasts of consecutive monthly expirations share two months
of index path, so their PITs are correlated and a nominal 5% test of a
calibrated forecast rejects about a quarter of the time.
``one_per_expiration`` keeps one horizon's row per expiration;
``non_overlapping_panel`` thins it to every second expiration at sixty days
and every third at ninety, which is the panel ``pit_calibration`` assumes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .config import BOOTSTRAP_RANDOM_STATE, BOOTSTRAP_REPLICATES, DM_LAG_BY_HORIZON

# Average calendar days between consecutive standard monthly expirations.
DAYS_PER_MONTH = 365.25 / 12

__all__ = [
    "newey_west_variance",
    "diebold_mariano",
    "rows_per_observation_date",
    "pooled_hac_lag",
    "block_bootstrap_mean",
    "nearest_to_target",
    "one_per_expiration",
    "months_spanned",
    "non_overlapping_panel",
    "effective_sample_size",
    "pit_calibration",
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


def rows_per_observation_date(
    frame: pd.DataFrame, date_column: str = "data_date"
) -> float:
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
    p_value = (
        2.0 * (1.0 - stats.norm.cdf(abs(statistic)))
        if np.isfinite(statistic) else np.nan
    )

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
    horizon: int,
    horizon_column: str = "target_dte_days",
    expiration_column: str = "expiration_date",
    actual_column: str = "actual_dte",
) -> pd.DataFrame:
    """One horizon's forecasts, one row per distinct expiration.

    Grouping by ``(expiration_date, target_dte_days)`` leaves three rows per
    outcome, because the 30-, 60- and 90-day forecasts of one expiration are
    all scored against the same SET print. Keeping one horizon removes that
    dependence.

    ``horizon`` is required. Without it the old default kept, per
    expiration, the row nearest its own target across all horizons, and
    since the 30- and 60-day chains both sit on their targets and ties went
    to the earlier date, that "pooled" panel was in practice a 60-day panel.

    The panel this returns is still not independent beyond thirty days: the
    ninety-day forecasts of consecutive monthly expirations share two months
    of index path. ``non_overlapping_panel`` thins it to forecasts whose
    windows do not overlap; use that for anything with a p-value.
    """
    frame = frame.copy()

    if actual_column not in frame.columns:
        frame[actual_column] = (
            pd.to_datetime(frame[expiration_column])
            - pd.to_datetime(frame["data_date"])
        ).dt.days

    frame = frame.loc[frame[horizon_column].astype(int).eq(int(horizon))]
    if frame.empty:
        raise ValueError(f"No observations at the {horizon}-day horizon.")

    return nearest_to_target(
        frame,
        group_columns=(expiration_column,),
        horizon_column=horizon_column,
        actual_column=actual_column,
    )


def months_spanned(horizon: int) -> int:
    """Monthly expirations a forecast window covers: 1, 2 and 3 at 30, 60, 90.

    Two forecasts of the same horizon overlap unless their expirations are
    at least this many months apart.
    """
    return max(1, round(int(horizon) / DAYS_PER_MONTH))


def non_overlapping_panel(
    frame: pd.DataFrame,
    horizon: int,
    horizon_column: str = "target_dte_days",
    expiration_column: str = "expiration_date",
    actual_column: str = "actual_dte",
) -> pd.DataFrame:
    """Forecasts of one horizon whose windows share no index path.

    Within a horizon consecutive monthly expirations resolve against
    different prints, but at sixty and ninety days their forecast windows
    overlap: the ninety-day forecasts for March and April share two months
    of path. Their PITs are then positively correlated (lag-one correlation
    near 0.5 at sixty days and 0.65 at ninety), and a nominal 5% test of a
    perfectly calibrated forecast rejects about 16% and 24% of the time.

    This keeps every ``months_spanned(horizon)``-th expiration -- every one
    at thirty days, every second at sixty, every third at ninety -- walking
    forward in calendar months, so a gap in the sample (an excluded year)
    restarts the count rather than shifting it.
    """
    panel = one_per_expiration(
        frame, horizon,
        horizon_column=horizon_column,
        expiration_column=expiration_column,
        actual_column=actual_column,
    )

    step = months_spanned(horizon)
    expirations = pd.to_datetime(panel[expiration_column])
    months = (expirations.dt.year * 12 + expirations.dt.month).to_numpy()

    order = np.argsort(months, kind="stable")
    keep = np.zeros(len(panel), dtype=bool)
    next_allowed = -np.inf
    for index in order:
        if months[index] >= next_allowed:
            keep[index] = True
            next_allowed = months[index] + step

    return panel.loc[keep].reset_index(drop=True)


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

def pit_calibration(
    pit_values,
    confidence: float = 0.95,
    parameters_estimated: int = 0,
) -> dict:
    """Three targeted calibration numbers, each naming an economic quantity.

    Calibrated forecasts produce uniform PITs. An omnibus test (KS or
    chi-square) says only that something is wrong; these three say what,
    and in the direction a risk premium predicts.

    * **Location: the mean PIT.** It is 1/2 for a calibrated forecast.
      Above 1/2 means outcomes land high in the forecast distribution --
      the direction of a positive equity risk premium.
    * **Width: the variance of the PIT about 1/2.** It is 1/12 for a
      calibrated forecast. Below 1/12 means outcomes cluster in the middle,
      so the forecast is too wide -- the direction of a variance risk
      premium. Reported as the ratio to 1/12.
    * **Left tail: the share of outcomes in the lowest decile.** It is 10%
      for a calibrated forecast. Below 10% means the forecast puts too much
      mass on crashes -- the direction of a crash risk premium. Reported
      with an exact (Clopper-Pearson) binomial interval and an exact
      two-sided binomial p-value.

    The location and width tests are t-tests with standard errors estimated
    from the sample rather than taken from the uniform null. That keeps each
    one about its own target: a forecast that is only too narrow has
    PITs with more spread than 1/12, and a null standard error would let
    that leak into the location test. All three assume independent
    observations: pass a panel from ``non_overlapping_panel``.

    If the forecast was tuned on these same observations -- as the
    CRRA-tilted density is, since gamma minimises the score on them -- the
    null distributions do not apply. Set ``parameters_estimated`` to the
    number fitted in-sample; the p-values are then returned as ``nan`` and
    the intervals are omitted, because a number that cannot be interpreted
    is worse than no number. The point estimates remain informative.
    """
    values = np.asarray(pit_values, dtype=float)
    values = values[np.isfinite(values)]

    n = int(values.size)
    if n < 2:
        raise ValueError("Too few PIT values.")

    def two_sided(difference, standard_error):
        # Identical PITs carry no estimate of spread to test against.
        if not standard_error > 0:
            return float("nan")
        t_statistic = difference / standard_error
        return float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))

    mean = float(values.mean())
    mean_se = float(values.std(ddof=1) / np.sqrt(n))
    mean_p = two_sided(mean - 0.5, mean_se)

    squared_deviation = 12.0 * (values - 0.5) ** 2
    variance_ratio = float(squared_deviation.mean())
    variance_ratio_se = float(squared_deviation.std(ddof=1) / np.sqrt(n))
    variance_p = two_sided(variance_ratio - 1.0, variance_ratio_se)

    # Clopper-Pearson: the exact interval, in its closed beta-quantile form.
    in_lowest_decile = int(np.sum(values < 0.1))
    alpha = 1.0 - confidence
    tail_lower = (
        float(stats.beta.ppf(alpha / 2, in_lowest_decile, n - in_lowest_decile + 1))
        if in_lowest_decile > 0 else 0.0
    )
    tail_upper = (
        float(stats.beta.ppf(1 - alpha / 2, in_lowest_decile + 1, n - in_lowest_decile))
        if in_lowest_decile < n else 1.0
    )
    tail_p = float(stats.binomtest(in_lowest_decile, n, 0.1).pvalue)

    in_sample = int(parameters_estimated) > 0
    if in_sample:
        mean_p = variance_p = tail_p = float("nan")
        tail_lower = tail_upper = float("nan")

    return {
        "n_observations": n,
        "mean": mean,
        "mean_standard_error": mean_se,
        "mean_p_value": mean_p,
        "variance_ratio": variance_ratio,
        "variance_ratio_standard_error": variance_ratio_se,
        "variance_ratio_p_value": variance_p,
        "share_below_10pct": in_lowest_decile / n,
        "share_below_10pct_lower": tail_lower,
        "share_below_10pct_upper": tail_upper,
        "share_below_10pct_p_value": tail_p,
        "confidence": confidence,
        "parameters_estimated": int(parameters_estimated),
        "p_values_valid": not in_sample,
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
