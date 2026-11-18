"""Calibration tests on overlapping and non-overlapping forecast panels.

A perfectly calibrated forecast of a random walk is simulated at 30, 60 and
90 days for consecutive monthly expirations. At 60 and 90 days consecutive
forecast windows share one or two months of path, so their PITs are
correlated, and a nominal 5% test on every expiration rejects far more
often than 5%. Thinned to every second (60 days) or third (90 days)
expiration, the windows no longer overlap and each test rejects at about
its nominal rate.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from implied_stock_distributions import stats as isd_stats

TESTS = ("mean", "variance_ratio", "share_below_10pct")
N_EXPIRATIONS = 162     # monthly expirations in the real sample's panel
REPLICATES = 300


def _calibrated_panel(generator, n_expirations=N_EXPIRATIONS):
    """Monthly expirations; each horizon's forecast is the true distribution."""
    increments = generator.normal(size=n_expirations + 2)
    expirations = pd.date_range("2005-01-01", periods=n_expirations, freq="MS")
    expirations = expirations + pd.Timedelta(days=20)

    # The path behind expiration i at m months is increments i+3-m .. i+2.
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    end = np.arange(n_expirations) + 3

    frames = []
    for months in (1, 2, 3):
        horizon = 30 * months
        path = cumulative[end] - cumulative[end - months]
        frames.append(pd.DataFrame({
            "data_date": expirations - pd.Timedelta(days=horizon),
            "expiration_date": expirations,
            "target_dte_days": horizon,
            "actual_dte": horizon,
            "pit_constrained": stats.norm.cdf(path / np.sqrt(months)),
        }))
    return pd.concat(frames, ignore_index=True)


def _rejection_rates(panel_builder, seed):
    generator = np.random.default_rng(seed)
    rejected = {horizon: np.zeros(len(TESTS)) for horizon in (30, 60, 90)}

    for _ in range(REPLICATES):
        frame = _calibrated_panel(generator)
        for horizon in rejected:
            panel = panel_builder(frame, horizon)
            result = isd_stats.pit_calibration(panel["pit_constrained"])
            rejected[horizon] += [
                result[f"{name}_p_value"] < 0.05 for name in TESTS
            ]

    return {horizon: counts / REPLICATES for horizon, counts in rejected.items()}


def test_months_spanned_by_each_horizon():
    assert [isd_stats.months_spanned(h) for h in (30, 60, 90)] == [1, 2, 3]


def test_non_overlapping_panel_thins_by_the_horizon():
    frame = _calibrated_panel(np.random.default_rng(0), n_expirations=24)

    for horizon, expected in ((30, 24), (60, 12), (90, 8)):
        panel = isd_stats.non_overlapping_panel(frame, horizon)
        assert len(panel) == expected
        months = (
            panel["expiration_date"].dt.year * 12
            + panel["expiration_date"].dt.month
        )
        assert months.diff().dropna().eq(isd_stats.months_spanned(horizon)).all()


def test_non_overlapping_panel_restarts_after_a_gap():
    """An excluded year must not shift the phase of the thinning."""
    frame = _calibrated_panel(np.random.default_rng(0), n_expirations=36)
    years = frame["expiration_date"].dt.year
    frame = frame.loc[years != 2006]          # 2005 to 2007, minus 2006

    panel = isd_stats.non_overlapping_panel(frame, 90)
    kept = panel["expiration_date"].dt.strftime("%Y-%m").tolist()

    assert kept[:4] == ["2005-01", "2005-04", "2005-07", "2005-10"]
    assert kept[4] == "2007-01"               # first month after the gap


@pytest.fixture(scope="module")
def thinned_rates():
    return _rejection_rates(isd_stats.non_overlapping_panel, seed=0)


@pytest.mark.parametrize("horizon", [30, 60, 90])
def test_a_calibrated_forecast_is_rejected_at_the_nominal_rate_when_thinned(
    thinned_rates, horizon,
):
    rates = thinned_rates[horizon]

    # 300 replicates: the standard error of a 5% rate is about 1.3 points,
    # so this bound is about four of them, and well below the 16% and 24%
    # that the overlapping 60- and 90-day panels give.
    for name, rate in zip(TESTS, rates, strict=True):
        assert rate < 0.10, (horizon, name, rate)


def test_overlapping_panels_reject_a_calibrated_forecast_too_often():
    """Why the thinning is needed: every expiration at 90 days over-rejects."""
    rates = _rejection_rates(isd_stats.one_per_expiration, seed=1)

    mean_test = TESTS.index("mean")
    assert rates[30][mean_test] < 0.10
    assert rates[90][mean_test] > 0.15
    assert rates[90][mean_test] > rates[60][mean_test] > rates[30][mean_test]
