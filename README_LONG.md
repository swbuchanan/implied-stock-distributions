# Recovering Real-World Probability Distributions from SPX Options

This project studies what the cross-section of option prices can tell us about the market's probability distribution for future equity-index levels.

I build an end-to-end Python pipeline that:

1. recovers **risk-neutral probability densities** from historical SPX option chains;
2. evaluates those density forecasts against realized SPX settlements;
3. transforms risk-neutral densities into candidate **real-world (physical) densities** using a pricing-kernel model; and
4. estimates the model's risk-aversion parameter from historical data and evaluates whether the transformed distributions are better calibrated to realized outcomes.

The dataset spans historical SPX option chains from 2005–2024, focusing on expirations approximately 30, 60, and 90 days from the observation date.

## Motivation

Option prices contain much more information than a single implied-volatility number. A cross-section of options across strikes embeds a market-implied distribution over the value of the underlying at expiration.

However, this distribution is a **risk-neutral distribution**, not a direct forecast of real-world probabilities. States in which a dollar payoff is particularly valuable—such as large market drawdowns—receive more weight in option prices.

This project therefore considers two related questions:

> **What probability distribution over future SPX levels is implied by current option prices?**

and

> **Can the risk-neutral distribution be transformed into a better estimate of the real-world distribution of future SPX outcomes?**

## 1. Estimating the forward price and discount rate

For calls and puts with the same strike and expiration, put-call parity gives

\[
C(K)-P(K)=e^{-rT}(F_{0,T}-K).
\]

For a given option chain, I regress \(C(K)-P(K)\) against strike \(K\). The slope and intercept identify the discount factor and forward price.

Estimating the forward directly from option prices avoids having to separately model expected dividends.

I also use put-call-parity residuals and bid-ask spreads as diagnostics for noisy or inconsistent option quotes.

## 2. Recovering the risk-neutral density

For a European call option,

\[
C(K,T)
=
e^{-rT}
\mathbb E_Q[(S_T-K)^+],
\]

where \(Q\) denotes the risk-neutral probability measure.

The Breeden-Litzenberger result implies

\[
f_Q(K)
=
e^{rT}
\frac{\partial^2 C(K,T)}{\partial K^2}.
\]

Thus, if a sufficiently smooth call-price curve \(C(K,T)\) can be recovered across strikes, its second derivative gives the option-implied risk-neutral density.

### Numerical challenge

Direct numerical differentiation of observed option prices is unstable: taking a second derivative greatly amplifies quote noise and interpolation artifacts.

I therefore fit a smooth, shape-constrained call-price curve before differentiating it. The constraints are motivated by no-arbitrage properties of European calls, particularly monotonicity and convexity with respect to strike.

The resulting density is numerically checked for properties including:

- non-negative probability mass;
- integration to approximately one;
- consistency of its mean with the estimated forward; and
- sensible CDF endpoints.

An earlier implied-volatility-smoothing approach is retained as a benchmark for comparing alternative density-construction methods.

## 3. Evaluating probabilistic forecasts

A probability density should be evaluated as a **distributional forecast**, rather than simply by comparing its mean or mode with the realized value.

I use the **Continuous Ranked Probability Score (CRPS)**:

\[
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left(
F(x)-\mathbf 1_{\{x\geq y\}}
\right)^2 dx,
\]

where \(F\) is the forecast CDF and \(y\) is the subsequently realized SPX settlement value.

Lower CRPS indicates a better probabilistic forecast. Unlike a point-error metric, CRPS rewards both calibration and concentration of the forecast distribution.

Forecasts are evaluated against historical SPX settlement values at expiration.

## 4. From risk-neutral to real-world probabilities

The risk-neutral density \(f_Q\) is not, in general, the same as the real-world density \(f_P\).

The two are connected through a pricing kernel or stochastic discount factor. For states indexed by the terminal SPX value \(s\),

\[
m(s)
=
e^{-rT}
\frac{f_Q(s)}{f_P(s)}.
\]

To identify \(f_P\), an additional assumption about the pricing kernel is required.

I study a representative-agent model with constant relative risk aversion (CRRA). Under power utility,

\[
u(W)
=
\frac{W^{1-\gamma}}{1-\gamma},
\]

marginal utility is

\[
u'(W)=W^{-\gamma}.
\]

Under the simplifying assumption that terminal aggregate wealth is proportional to the market index,

\[
m(S_T)\propto S_T^{-\gamma}.
\]

It follows that

\[
f_P(s)
=
\frac{
s^\gamma f_Q(s)
}{
\int_0^\infty y^\gamma f_Q(y)\,dy
}.
\]

The parameter \(\gamma\) controls how strongly probability mass is shifted away from low-index states and toward high-index states when moving from the risk-neutral to the physical distribution.

Economically, \(Q\) overweights bad states because a payoff received during a market downturn is especially valuable to a risk-averse investor. The transformation attempts to remove this state-price effect in order to recover real-world probabilities.

## 5. Estimating the pricing-kernel parameter

Rather than selecting an arbitrary literature value for \(\gamma\), the final stage of the project estimates it empirically from historical option-implied distributions and subsequent realized settlements.

For each observation date \(t\), I first construct and cache the risk-neutral density

\[
f_{Q,t}.
\]

For a candidate value of \(\gamma\), I then construct

\[
f_{P,t}(s;\gamma)
=
\frac{s^\gamma f_{Q,t}(s)}
{\int y^\gamma f_{Q,t}(y)\,dy}.
\]

I choose \(\gamma\) according to the historical predictive performance of these transformed densities, using proper scoring rules such as CRPS.

The analysis also investigates whether the inferred parameter is stable across approximately

\[
30,\quad60,\quad90
\]

day horizons.

For final out-of-sample evaluation, the data are split chronologically rather than randomly, with overlapping forecast horizons handled carefully to avoid look-ahead contamination.

## 6. Empirical questions

The completed analysis is designed to answer several questions:

- How accurately can a risk-neutral density be recovered from noisy historical option-chain data?
- How well calibrated are risk-neutral SPX distributions to subsequent realized settlements?
- Does a CRRA pricing-kernel transformation improve probabilistic forecasts relative to \(f_Q\)?
- What value of \(\gamma\) is preferred by the historical data?
- Is the estimated pricing kernel stable across 30-, 60-, and 90-day horizons?
- Does an estimated pricing kernel continue to improve forecasts out of sample?

## Results

_Final numerical results will be reported here after the parameter-estimation and out-of-sample evaluation stages are complete._

The final comparison will report, by maturity:

- average CRPS for the original risk-neutral densities;
- average CRPS for transformed physical densities;
- estimated \(\gamma\);
- out-of-sample improvements relative to the risk-neutral benchmark; and
- distribution-calibration diagnostics.

## Implementation

The project is written in Python. The core implementation is separated into reusable modules for:

- loading and iterating through large historical option datasets;
- Black/Black-Scholes pricing and implied-volatility calculations;
- estimating forwards and discount rates from put-call parity;
- smoothing and shape-constrained option-price fitting;
- recovering risk-neutral distributions; and
- evaluating probabilistic forecasts.

The analysis notebook provides the mathematical derivations, visualizations, diagnostics, and empirical experiments.

### Main libraries

- NumPy
- pandas
- SciPy
- Matplotlib

## Project structure

implied_stock_distributions/
├── black_scholes.py       # option pricing / implied volatility
├── data_access.py         # option-chain catalogue and iteration
├── evaluation.py          # forecast scoring
├── fitting.py             # shape-constrained curve fitting
├── forwards.py            # put-call-parity estimation
└── smoothing.py           # alternative smoothing methods

project_v2.ipynb           # theory, experiments, and empirical analysis


## Limitations and extensions

Recovering a probability density from option prices is an ill-conditioned numerical problem, particularly in the tails where strikes are sparse. Possible extensions include:

- alternative arbitrage-free volatility-surface parameterizations;
- improved treatment of distribution tails;
- alternative pricing-kernel specifications;
- time-varying rather than constant risk-aversion parameters;
- additional proper scoring rules and calibration tests;
- bootstrap or HAC inference for overlapping forecast horizons; and
- comparison against statistical or econometric return-density forecasts.

More generally, the project illustrates an important distinction in derivatives pricing:

option prices reveal state prices, not physical probabilities directly
	​


and investigates how much of the gap between the two can be explained by a simple economic model of risk preferences.

References

The project draws primarily on work concerning:

- the Breeden-Litzenberger recovery of state-price densities from option prices;
- option-implied risk-neutral distributions;
- stochastic discount factors and pricing kernels;
- CRRA/power utility;
- transformations from risk-neutral to physical densities; and
- proper scoring rules for density forecasts.

See the analysis notebook for the full bibliography and mathematical exposition.
