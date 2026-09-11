"""Put-call parity estimation, checked against a chain with a known answer."""

import numpy as np
import pytest

from implied_stock_distributions.errors import ChainQualityError
from implied_stock_distributions.forwards import estimate_forward_and_rate

from .synthetic import synthetic_chain


def test_recovers_the_forward_and_rate_exactly():
    """On an arbitrage-free chain the regression is exact."""
    chain, truth = synthetic_chain(strike_step=25.0)
    estimates, pairs = estimate_forward_and_rate(chain)

    assert estimates["forward"] == pytest.approx(truth["forward"], rel=1e-10)
    assert estimates["risk_free_rate"] == pytest.approx(truth["rate"], abs=1e-10)
    assert estimates["discount_factor"] == pytest.approx(
        truth["discount_factor"], rel=1e-10
    )
    assert estimates["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(pairs["parity_residual"], 0.0, atol=1e-8)


def test_moneyness_band_restricts_the_fitting_window():
    chain, _ = synthetic_chain(strike_step=25.0)

    wide, _ = estimate_forward_and_rate(chain, moneyness_band=None)
    narrow, _ = estimate_forward_and_rate(chain, moneyness_band=0.10)

    assert narrow["n_pairs"] < wide["n_pairs"]
    # Both are exact on noiseless data; the band must not bias the estimate.
    assert narrow["forward"] == pytest.approx(wide["forward"], rel=1e-9)


def test_survives_quote_noise_with_a_small_error():
    chain, truth = synthetic_chain(strike_step=25.0, noise=0.15, seed=3)
    estimates, _ = estimate_forward_and_rate(chain)

    assert estimates["forward"] == pytest.approx(truth["forward"], rel=5e-4)
    assert estimates["risk_free_rate"] == pytest.approx(truth["rate"], abs=5e-3)


def test_rejects_an_implausible_rate():
    """A chain implying a 40% rate is a data problem, and must not pass."""
    chain, _ = synthetic_chain(rate=0.40, strike_step=25.0)

    with pytest.raises(ChainQualityError):
        estimate_forward_and_rate(chain)

    # It still parses when the caller asks to see the bad fit.
    estimates, _ = estimate_forward_and_rate(chain, enforce_quality=False)
    assert estimates["risk_free_rate"] == pytest.approx(0.40, abs=1e-8)


def test_spread_weighting_does_not_change_a_clean_fit():
    chain, truth = synthetic_chain(strike_step=25.0)

    weighted, _ = estimate_forward_and_rate(chain, use_spread_weights=True)
    unweighted, _ = estimate_forward_and_rate(chain, use_spread_weights=False)

    assert weighted["forward"] == pytest.approx(unweighted["forward"], rel=1e-9)
    assert weighted["forward"] == pytest.approx(truth["forward"], rel=1e-9)
