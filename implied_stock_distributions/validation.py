"""Checking the pipeline's outputs against published external series.

Three quantities fall out of this project that somebody else already
publishes, which makes them free tests of everything upstream. Each is
currently *asserted* in the notebook rather than shown:

* **The short rate.** The rate implied by the put-call parity regression is
  an input to nothing -- it comes purely from option prices. If it tracks
  the actual Treasury bill rate through the financial crisis and the 2022
  hiking cycle, the forward and discount factor feeding every density can
  be trusted. FRED serves ``DGS1MO`` and ``DGS3MO`` as a CSV with no key.
* **The dividend yield.** ``F / S = exp((r - q) tau)``, so the same
  regression gives ``q``. Shiller's monthly S&P 500 series carries the
  trailing dividend and the index level, so the realised yield is one
  division.
* **Risk-neutral skewness.** Cboe's SKEW index is ``100 - 10 S``, where
  ``S`` is the risk-neutral skewness of thirty-day S&P 500 log returns
  estimated from option prices. The fitted densities give ``S`` directly.
  This is the strongest of the three, because it tests the *shape* of the
  recovered density and not only its first two moments -- and Cboe computes
  it by a completely different method (the Bakshi-Kapadia-Madan portfolio
  of options), so agreement is not circular.

A fourth check lives here because it needs the same machinery: whether
chains that fail the parity quality gates cluster in stressed markets. If
the drop rate is higher when VIX is high, the whole evaluation is
conditioned on quiet markets and every tail conclusion weakens. That is the
first thing an interviewer will ask about, and it is a groupby.

Every fetch caches to Parquet under ``config.EXTERNAL_DIR``, so a re-run
does not hit the network, and every function accepts a pre-loaded frame so
the pipeline can be run entirely offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CBOE_INDEX_HISTORY_URL,
    CBOE_SKEW_INDEX,
    CBOE_VIX_INDEX,
    EXTERNAL_CACHE_PATHS,
    FRED_CSV_URL,
    FRED_SHORT_RATE_SERIES,
    SHILLER_SP500_CSV_URL,
    VIX_BUCKET_EDGES,
    VIX_BUCKET_LABELS,
)

__all__ = [
    "fetch_fred_series",
    "fetch_short_rates",
    "fetch_cboe_index",
    "fetch_skew_index",
    "fetch_vix",
    "fetch_dividend_yield",
    "compare_series",
    "validate_implied_rate",
    "validate_dividend_yield",
    "risk_neutral_skewness",
    "skew_index_equivalent",
    "validate_skew",
    "failure_rate_by_bucket",
]


class ExternalDataUnavailable(RuntimeError):
    """A published series could not be fetched and is not cached."""


def _cached(path: Path | None, build):
    """Return a cached frame, or build it, cache it and return it."""
    if path is not None:
        path = Path(path)
        if path.exists():
            return pd.read_parquet(path)

    frame = build()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    return frame


def _read_csv(url: str, what: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as exc:  # noqa: BLE001 - the cause is reported verbatim
        raise ExternalDataUnavailable(
            f"Could not fetch {what} from {url} ({exc}). Download it by hand "
            f"and pass it in, or place the cached Parquet where "
            f"config.EXTERNAL_CACHE_PATHS expects it."
        ) from exc


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def fetch_fred_series(series: str) -> pd.DataFrame:
    """One FRED series as ``(date, value)``, with '.' treated as missing."""
    frame = _read_csv(FRED_CSV_URL.format(series=series), f"FRED series {series}")

    # FRED has used both DATE and observation_date as the first column name.
    date_column = frame.columns[0]
    value_column = [c for c in frame.columns if c != date_column][0]

    return pd.DataFrame({
        "date": pd.to_datetime(frame[date_column]),
        "value": pd.to_numeric(frame[value_column], errors="coerce") / 100.0,
        "series": series,
    }).dropna(subset=["value"]).reset_index(drop=True)


def fetch_short_rates(
    series=FRED_SHORT_RATE_SERIES,
    cache_path=EXTERNAL_CACHE_PATHS["short_rate"],
) -> pd.DataFrame:
    """One- and three-month Treasury constant-maturity rates, wide by date.

    Quoted rates are converted to continuously compounded ones, because the
    parity regression recovers ``-log(D) / tau`` and comparing that against
    a bond-equivalent yield would build in a spread that is not an error.
    """
    def build():
        parts = [fetch_fred_series(name) for name in series]
        wide = (
            pd.concat(parts, ignore_index=True)
            .pivot(index="date", columns="series", values="value")
            .sort_index()
        )
        # Simple annualised quote -> continuously compounded.
        for column in wide.columns:
            wide[column] = np.log1p(wide[column])
        return wide.reset_index()

    return _cached(cache_path, build)


def fetch_cboe_index(index: str, cache_path=None) -> pd.DataFrame:
    """A Cboe daily index history as ``(date, close)``."""
    def build():
        frame = _read_csv(
            CBOE_INDEX_HISTORY_URL.format(index=index), f"Cboe {index} history"
        )
        columns = {c.strip().upper(): c for c in frame.columns}
        date_column = columns.get("DATE", frame.columns[0])
        close_column = columns.get("CLOSE", columns.get(index.upper()))

        if close_column is None:
            close_column = frame.columns[-1]

        return pd.DataFrame({
            "date": pd.to_datetime(frame[date_column], format="mixed"),
            "close": pd.to_numeric(frame[close_column], errors="coerce"),
        }).dropna().sort_values("date").reset_index(drop=True)

    return _cached(cache_path, build)


def fetch_skew_index(cache_path=EXTERNAL_CACHE_PATHS["skew"]) -> pd.DataFrame:
    """Cboe SKEW, daily since 1990."""
    return fetch_cboe_index(CBOE_SKEW_INDEX, cache_path)


def fetch_vix(cache_path=EXTERNAL_CACHE_PATHS["vix"]) -> pd.DataFrame:
    """Cboe VIX, daily since 1990."""
    return fetch_cboe_index(CBOE_VIX_INDEX, cache_path)


def fetch_dividend_yield(
    cache_path=EXTERNAL_CACHE_PATHS["dividend_yield"],
) -> pd.DataFrame:
    """Monthly S&P 500 trailing dividend yield from Shiller's series."""
    def build():
        frame = _read_csv(SHILLER_SP500_CSV_URL, "Shiller S&P 500 series")

        columns = {c.strip().lower(): c for c in frame.columns}
        out = pd.DataFrame({
            "date": pd.to_datetime(frame[columns["date"]]),
            "index_level": pd.to_numeric(
                frame[columns["sp500"]], errors="coerce"
            ),
            "dividend": pd.to_numeric(
                frame[columns["dividend"]], errors="coerce"
            ),
        }).dropna()

        out["dividend_yield"] = out["dividend"] / out["index_level"]
        return out.sort_values("date").reset_index(drop=True)

    return _cached(cache_path, build)


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------

def compare_series(
    implied: pd.Series,
    reference: pd.Series,
    tolerance_days: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Align two date-indexed series and summarise how closely they track.

    The reference is forward-filled onto the implied series' dates within
    ``tolerance_days``, so a bank holiday does not become a missing
    observation, and anything further away is dropped rather than matched
    to a stale value.

    Returns the aligned frame and a summary with the correlation, the mean
    and mean absolute difference, and the same broken out by year -- which
    is the part that matters, since a correlation over twenty years is
    carried entirely by the level shift between the 2010s and 2022.
    """
    implied = implied.dropna().sort_index()
    reference = reference.dropna().sort_index()

    if implied.empty or reference.empty:
        raise ValueError("Both series must be non-empty.")

    aligned = pd.merge_asof(
        pd.DataFrame({"date": implied.index, "implied": implied.to_numpy()}),
        pd.DataFrame({"date": reference.index, "reference": reference.to_numpy()}),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=tolerance_days),
    ).dropna()

    if aligned.empty:
        raise ValueError("No dates matched within the tolerance.")

    aligned["difference"] = aligned["implied"] - aligned["reference"]
    aligned["year"] = aligned["date"].dt.year

    by_year = (
        aligned.groupby("year")
        .agg(
            n=("difference", "size"),
            mean_implied=("implied", "mean"),
            mean_reference=("reference", "mean"),
            bias=("difference", "mean"),
            mean_absolute=("difference", lambda s: s.abs().mean()),
        )
        .round(6)
    )

    summary = {
        "n_observations": int(len(aligned)),
        "correlation": float(aligned["implied"].corr(aligned["reference"])),
        "bias": float(aligned["difference"].mean()),
        "mean_absolute_difference": float(aligned["difference"].abs().mean()),
        "rmse": float(np.sqrt(np.mean(aligned["difference"] ** 2))),
        "by_year": by_year,
    }

    return aligned, summary


def validate_implied_rate(
    diagnostics: pd.DataFrame,
    short_rates: pd.DataFrame | None = None,
    horizon_column: str = "target_dte_days",
) -> dict:
    """Parity-implied rates against Treasury bills, per horizon.

    The 30-day chains are compared against the one-month bill and the
    60- and 90-day chains against the three-month one, because that is the
    maturity each is actually pricing.
    """
    short_rates = short_rates if short_rates is not None else fetch_short_rates()
    short_rates = short_rates.set_index("date")

    reference_for = {30: "DGS1MO", 60: "DGS3MO", 90: "DGS3MO"}
    results = {}

    for horizon, group in diagnostics.groupby(horizon_column):
        column = reference_for.get(int(horizon))
        if column is None or column not in short_rates.columns:
            continue

        implied = (
            group.groupby("data_date")["risk_free_rate"].median().sort_index()
        )
        aligned, summary = compare_series(implied, short_rates[column])
        results[int(horizon)] = {
            "reference": column, "aligned": aligned, **summary,
        }

    return results


def validate_dividend_yield(
    diagnostics: pd.DataFrame,
    dividend_yield: pd.DataFrame | None = None,
) -> dict:
    """Parity-implied dividend yield against Shiller's realised yield.

    Two things make an exact match the wrong expectation, and saying so is
    part of the check rather than an excuse. The implied yield is
    *forward-looking* over the option's life, while Shiller's is a trailing
    twelve-month figure, so the implied series leads at turning points. And
    the implied yield absorbs the index's borrow cost, so it sits slightly
    above the pure dividend yield when securities lending is expensive.
    Tracking to within a few tens of basis points, with the right shape
    through 2008 and 2020, is what confirms the regression.
    """
    dividend_yield = (
        dividend_yield if dividend_yield is not None else fetch_dividend_yield()
    )

    if "implied_dividend_yield" not in diagnostics.columns:
        diagnostics = diagnostics.assign(
            implied_dividend_yield=(
                diagnostics["risk_free_rate"]
                - np.log(diagnostics["forward_over_spot"])
                / diagnostics["time_to_expiry"]
            )
        )

    implied = (
        diagnostics.groupby("data_date")["implied_dividend_yield"]
        .median()
        .sort_index()
    )

    reference = dividend_yield.set_index("date")["dividend_yield"]

    # Shiller is monthly, so allow a month of tolerance in the alignment.
    aligned, summary = compare_series(implied, reference, tolerance_days=31)
    return {"aligned": aligned, **summary}


# --------------------------------------------------------------------------
# Risk-neutral skewness against Cboe SKEW
# --------------------------------------------------------------------------

def risk_neutral_skewness(strike_grid, density, forward=None) -> float:
    """Skewness of log returns under a tabulated risk-neutral density.

    This is the quantity Cboe's SKEW index is a linear function of. Cboe
    estimates it from a portfolio of options directly (Bakshi, Kapadia and
    Madan, 2003); computing it here from the fitted density instead means
    agreement between the two is a real test of the recovered *shape*,
    since nothing in the two calculations is shared.
    """
    strike_grid = np.asarray(strike_grid, dtype=float)
    density = np.asarray(density, dtype=float)

    positive = strike_grid > 0
    levels = strike_grid[positive]
    weights = density[positive]

    mass = float(np.trapezoid(weights, levels))
    if not np.isfinite(mass) or mass <= 0:
        return float("nan")

    reference = float(forward) if forward else float(
        np.trapezoid(levels * weights, levels) / mass
    )
    log_returns = np.log(levels / reference)

    def moment(order, centre=0.0):
        return float(
            np.trapezoid((log_returns - centre) ** order * weights, levels) / mass
        )

    mean = moment(1)
    variance = moment(2, mean)
    third = moment(3, mean)

    if variance <= 0:
        return float("nan")

    return third / variance**1.5


def skew_index_equivalent(skewness) -> np.ndarray:
    """Cboe's SKEW from a risk-neutral skewness: ``100 - 10 S``."""
    return 100.0 - 10.0 * np.asarray(skewness, dtype=float)


def validate_skew(
    fitted: pd.DataFrame,
    skew_index: pd.DataFrame | None = None,
    horizon: int = 30,
    skewness_column: str = "risk_neutral_skewness",
    horizon_column: str = "target_dte_days",
) -> dict:
    """Fitted risk-neutral skewness against Cboe SKEW, at thirty days.

    SKEW describes a thirty-day horizon, so only the thirty-day chains are
    comparable; the sixty- and ninety-day arms should show the same shape
    at a different level, and that is worth plotting but not testing.
    """
    skew_index = skew_index if skew_index is not None else fetch_skew_index()

    subset = fitted.loc[fitted[horizon_column].astype(int).eq(int(horizon))]
    if subset.empty:
        raise ValueError(f"No fitted skewness at the {horizon}-day horizon.")

    implied = (
        subset.groupby("data_date")[skewness_column].median().sort_index()
    )
    implied_index = pd.Series(
        skew_index_equivalent(implied.to_numpy()), index=implied.index
    )

    reference = skew_index.set_index("date")["close"]
    aligned, summary = compare_series(implied_index, reference)

    return {"aligned": aligned, "horizon": int(horizon), **summary}


# --------------------------------------------------------------------------
# Is the sample conditioned on quiet markets?
# --------------------------------------------------------------------------

def failure_rate_by_bucket(
    attempted: pd.DataFrame,
    failed_keys: pd.DataFrame,
    vix: pd.DataFrame | None = None,
    key_columns=("data_date", "expiration_date", "target_dte_days"),
    edges=VIX_BUCKET_EDGES,
    labels=VIX_BUCKET_LABELS,
) -> dict:
    """Where the dropped chains sit, by year and by volatility regime.

    Chains failing the parity gates will not fail at random: wide spreads
    and stale quotes cluster exactly where the tails are interesting --
    October 2008, March 2020, 2022. If the drop rate rises with VIX then
    every tail conclusion in the project is conditioned on quiet markets,
    and the honest thing is to say so with a number attached.

    ``attempted`` is every chain the sweep tried; ``failed_keys`` is the
    subset that failed, keyed the same way.
    """
    vix = vix if vix is not None else fetch_vix()

    key_columns = list(key_columns)
    attempted = attempted[key_columns].drop_duplicates().copy()
    attempted["data_date"] = pd.to_datetime(attempted["data_date"])

    failed = failed_keys[key_columns].drop_duplicates().copy()
    failed["data_date"] = pd.to_datetime(failed["data_date"])
    failed["failed"] = True

    merged = attempted.merge(failed, on=key_columns, how="left")
    merged["failed"] = merged["failed"].fillna(False)

    merged = pd.merge_asof(
        merged.sort_values("data_date"),
        vix.rename(columns={"date": "data_date", "close": "vix"}).sort_values(
            "data_date"
        ),
        on="data_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    )

    merged["vix_bucket"] = pd.cut(
        merged["vix"], bins=list(edges), labels=list(labels)
    )
    merged["year"] = merged["data_date"].dt.year

    def rate(frame, column):
        return (
            frame.groupby(column, observed=True)["failed"]
            .agg(attempted="size", failed="sum")
            .assign(failure_rate=lambda t: t["failed"] / t["attempted"])
            .round(4)
        )

    return {
        "chains": merged,
        "by_year": rate(merged, "year"),
        "by_vix_bucket": rate(merged, "vix_bucket"),
        "overall_failure_rate": float(merged["failed"].mean()),
    }
