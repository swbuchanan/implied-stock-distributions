# Implied stock distributions

**What probability distribution for the future S&P 500 is embedded in SPX
option prices, how well does it describe what actually happened, and does a
simple pricing-kernel model close the gap?**

Fourteen years of daily SPX option chains at maturities near 30, 60 and 90
days. Forwards and discount factors come from put–call parity; densities come
from the Breeden–Litzenberger relation applied to a shape-constrained
(Fengler-style) call-price spline; forecasts are scored with the continuous
ranked probability score against realised Cboe SET settlement values; and the
risk-neutral densities are tilted toward physical ones under a CRRA pricing
kernel.

```
implied_stock_distributions/
    config.py        paths, sample definition, every hyperparameter
    pricing.py       Black's formula on a forward, and its inversion
    forwards.py      forward and discount factor from put-call parity
    fitting.py       the shape-constrained call-price spline
    smoothing.py     the unconstrained implied-volatility baseline
    selection.py     cross-validated choice of the roughness penalty
    evaluation.py    CRPS, PIT, and the risk-neutral to physical tilt
    benchmarks.py    reference forecasts to score against
    stats.py         Diebold-Mariano, block bootstrap, calibration tests
    data_access.py   locating, cataloguing and loading processed chains
    plotting.py      the figures used in the notebook
notebooks/
    01_processing.ipynb   raw CSVs to monthly Parquet, plus settlement values
    02_project.ipynb      the analysis
tests/                    pytest suite, all against exact answers
```

## Results

Run `notebooks/02_project.ipynb`; its conclusion renders the headline numbers
from the results it just computed, so they cannot drift away from the code.
Fill in the table below from that output once the pipeline has been run
end to end.

|                              | Mean relative CRPS | Skill vs. lognormal benchmark |
| ---------------------------- | ------------------ | ----------------------------- |
| Shape-constrained            |                    |                               |
| IV smoothing                 |                    |                               |
| Lognormal at ATM implied vol |                    |                               |

## What is worth looking at

**The fit is a probability distribution by construction, and says so.** The
boundary conditions `g(0) = 1`, `g'(0) = -1`, `g(x_max) = 0`, `g'(x_max) = 0`
imply two exact identities: the fitted density integrates to one, and its
mean equals the forward implied by put–call parity. Both are reported on
every chain as correctness checks rather than imposed by renormalising
afterwards.

**The baseline is a fair one.** Constraints can only be shown to matter if
the unconstrained comparison is given the same information. The
implied-volatility baseline uses the same out-of-the-money quote selection,
the same support (volatility extrapolated flat beyond the traded strikes),
and a smoothing bound set to the expected weighted residual sum given each
quote's estimated noise, rather than a number chosen by eye.

**The free parameter is selected, not chosen.** The roughness penalty is set
by leave-out-every-fifth-strike cross-validation on out-of-sample pricing
error, measured in half-spreads. Nothing about the realised outcome enters,
so the selection is independent of the evaluation.

**The comparisons carry standard errors.** Each expiration is forecast on
about twenty-one consecutive trading days, so several thousand chains are far
fewer independent observations. Differences in mean score are tested with
Diebold–Mariano using a Newey–West variance, and with a bootstrap over whole
expiration months.

**Calibration is reported, not just accuracy.** The probability integral
transform histogram shows *how* the risk-neutral densities fail against
outcomes, which is the direct evidence that they price risk as well as
describing it — and it is shown again after the CRRA tilt.

## Reproducing

```bash
pip install -e ".[notebooks,dev]"
pytest
```

The tests need no data: they build Black–Scholes chains with a known
volatility and check that the pipeline recovers the analytic lognormal
density, that CRPS matches its Gaussian closed form, and that the parity
regression recovers the forward and rate it was given.

To run the analysis you need daily SPX option-chain CSVs at
`data/raw/SPX/<year>/*.csv`, with columns `OptionKey`, `Symbol`, `DataDate`,
`ExpirationDate`, `PutCall`, `StrikePrice`, `BidPrice`, `AskPrice`,
`BidSize`, `AskSize`, `LastPrice`, `Volume`, `OpenInterest` and
`UnderlyingPrice`. **The raw data are licensed and are not redistributed
here.** Settlement values are downloaded from Cboe by the processing
notebook and cached.

Then run `notebooks/01_processing.ipynb` followed by
`notebooks/02_project.ipynb`.

Without the licensed data, `python -m tests.make_fixture` writes a small
synthetic dataset in the same layout — Black–Scholes chains and a catalogue —
which is enough to execute `02_project.ipynb` end to end and see every figure.
The numbers it produces are meaningless; the point is that a reader can run
the analysis without asking for the data.

`cvxpy` with CLARABEL solves the convex program. If it is unavailable the
package falls back to a small ADMM solver that needs only NumPy and SciPy —
slower, and only intended to keep the pipeline and its tests runnable in a
bare environment.

## Sample

`config.ANALYSIS_YEARS` is deliberately not contiguous: years whose vendor
files carry too many indistinguishable duplicate option keys are excluded,
and `config.EXCLUDED_YEARS` lists them. The coverage table at the end of
`01_processing.ipynb` is the evidence. Excluding 2020 removes the most
informative regime in the sample for tail behaviour, and recovering it would
be worth more than any modelling change.

Only standard monthly, AM-settled expirations are used, because those are the
ones that settle against SET — the realised value everything is scored
against. Time to expiry is measured to the Thursday close, since the
settlement print comes from Friday's opening auction.

## Limitations

- The tails beyond the traded strike range are extrapolation governed by the
  roughness penalty, not by prices. The notebook reports how much mass lives
  there; a proper tail model (Figlewski-style GEV tails) would separate the
  smoothing question from the extrapolation question.
- Only static strike arbitrage is removed. Fengler's backward pass across
  maturities, which would make the three horizons mutually consistent, is not
  implemented.
- The CRRA parameter is estimated on the same observations used to measure
  the improvement, so the Part 3 result is an in-sample fit. Estimating it on
  a training period and evaluating out of sample is the obvious next step.

## References

- Fengler, M. R. (2009). *Arbitrage-Free Smoothing of the Implied Volatility
  Surface.* Quantitative Finance 9(4), 417–428.
- Breeden, D. T. and Litzenberger, R. H. (1978). *Prices of State-Contingent
  Claims Implicit in Option Prices.* Journal of Business 51(4), 621–651.
- Gneiting, T. and Raftery, A. E. (2007). *Strictly Proper Scoring Rules,
  Prediction, and Estimation.* JASA 102(477), 359–378.
- Diebold, F. X. and Mariano, R. S. (1995). *Comparing Predictive Accuracy.*
  Journal of Business & Economic Statistics 13(3), 253–263.
- Taylor, S. J. *Asset Price Dynamics, Volatility, and Prediction*, chapter 16.
