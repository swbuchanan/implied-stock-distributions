"""One chain end to end, the synthetic dataset, and the skewness check."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from implied_stock_distributions.data_access import (
    catalog_path,
    iter_chains,
    resolve_parquet_path,
)
from implied_stock_distributions.pipeline import score_chain
from implied_stock_distributions.synthetic import (
    SyntheticMarket,
    build_chain,
    write_synthetic_dataset,
)
from implied_stock_distributions.validation import risk_neutral_skewness

FORWARD = 4000.0


def _lognormal_density(grid, volatility=0.16, tau=30 / 365):
    scale = volatility * np.sqrt(tau)
    return stats.lognorm.pdf(grid, s=scale, scale=FORWARD * np.exp(-0.5 * scale**2))


def test_full_support_skewness_is_dominated_by_extrapolated_mass():
    """The documented limitation of the full-support log-skewness.

    The log-skewness of a lognormal is zero. Add one per cent of mass between
    0.1 F and 0.3 F -- far below any traded strike -- and the log-skewness
    becomes strongly negative: the tail, not the traded prices, decides it.
    """
    grid = np.linspace(1.0, 2.5 * FORWARD, 20001)
    clean = _lognormal_density(grid)

    sliver = ((grid > 0.1 * FORWARD) & (grid < 0.3 * FORWARD)).astype(float)
    sliver /= np.trapezoid(sliver, grid)
    contaminated = 0.99 * clean + 0.01 * sliver

    assert risk_neutral_skewness(grid, clean) == pytest.approx(0.0, abs=0.01)
    assert risk_neutral_skewness(grid, contaminated) < -5.0


def _synthetic_chain(seed=1):
    market = SyntheticMarket()
    data_date = pd.Timestamp("2019-05-17")
    expiration = pd.Timestamp("2019-06-21")
    return build_chain(
        data_date, expiration, 30, 2850.0, 0.15, market,
        np.random.default_rng(seed),
    )


def test_score_chain_runs_end_to_end_and_honours_the_identities():
    chain, model = _synthetic_chain()
    result = score_chain(chain, realized_value=2900.0, roughness_penalty=100.0)
    scores = result["scores"]

    # The parity forward recovers the model's forward from noisy quotes.
    assert scores["forward"] == pytest.approx(model.forward, rel=2e-3)

    # The constrained density is a distribution with mean F, by construction.
    assert result["constrained"]["mass"] == pytest.approx(1.0, abs=1e-4)
    assert result["constrained"]["mean"] == pytest.approx(scores["forward"], rel=1e-4)

    for column in (
        "relative_crps_constrained", "relative_crps_iv",
        "relative_crps_iv_recentred", "relative_crps_lognormal",
        "pit_constrained", "risk_neutral_skewness",
    ):
        assert np.isfinite(scores[column]), column

    # The cached density is what Part 3 re-tilts; it must be the scored one.
    assert np.array_equal(result["cached"]["fQ"], result["constrained"]["density"])


def test_constrained_fit_scores_close_to_the_truth():
    """Against a known density the fit gives up almost nothing in CRPS."""
    from implied_stock_distributions.evaluation import crps_from_cdf

    chain, model = _synthetic_chain(seed=4)
    differences = []
    for outcome in (2500.0, 2800.0, 2900.0, 3100.0):
        result = score_chain(chain, realized_value=outcome, roughness_penalty=100.0)
        grid = result["constrained"]["strike_grid"]
        true_cdf = model.cdf(grid)
        true_cdf = (true_cdf - true_cdf[0]) / (true_cdf[-1] - true_cdf[0])
        truth = crps_from_cdf(grid, true_cdf, outcome)
        differences.append(abs(result["scores"]["crps_constrained"] - truth) / truth)

    assert max(differences) < 0.05


def test_synthetic_dataset_round_trips_through_data_access(tmp_path):
    market = SyntheticMarket(n_months=2, seed=3)
    written = write_synthetic_dataset(tmp_path / "data", market, verbose=False)
    catalog = written["catalog"]

    # Outside the project root the catalogue stores absolute paths, so the
    # loader finds the files from anywhere.
    first_key, first_chain = next(iter_chains(catalog))
    assert len(first_chain) > 0
    assert first_chain["data_date"].iloc[0] == first_key[0]

    settlement = written["settlement_values"]
    assert {"expiration_year", "expiration_month", "set_value"} <= set(
        settlement.columns
    )


def test_a_relative_path_outside_the_project_is_stored_absolute(tmp_path, monkeypatch):
    """A relative path is relative to the working directory it was given in,
    so the catalogue must not hand it to a loader that resolves against the
    project root."""
    monkeypatch.chdir(tmp_path)
    stored = catalog_path("alt/processed/SPX/chain.parquet")
    expected = tmp_path.resolve() / "alt" / "processed" / "SPX" / "chain.parquet"
    assert stored == str(expected)
    assert resolve_parquet_path(stored) == expected
