# Market-implied stock price PDFs: risk-neutral and physical

Option prices embed information about the markets expectation of future performance of the underlying asset.
The set of European option prices across strikes for a given maturity $T$ implies a risk-neutral probability density function of the price $S_T$ of the underlying asset at the maturity.

This project uses several years of daily SPX option chain data to extract the market-implied risk-neutral pdfs for options with 1 day maturities, 7 day maturities, and 28 day maturities.
Using the Breeden-Litzenberger relation, which says that the implied pdf is given by the second partial derivative of price at maturity with respect to strike:
$f_Q(K) = e^{rT} \frac{\partial^2 C(K, T)}{\partial K^2},$
we numerically recover the pdf.

For the full rendered notebook (with tables and plots), see:

[My project on nbviewer](https://nbviewer.org/github/swbuchanan/implied-stock-distributions/blob/main/project.ipynb)

or click project.ipynb in the github repository.




# Option-Implied Probability Distributions

Extracting risk-neutral and physical probability distributions from SPX option prices.

## Overview

Option prices contain information about the market's expectations for the future distribution of an underlying asset. Rather than using a single quantity such as implied volatility, this project recovers the **entire market-implied probability distribution of the S&P 500 at option expiration**.

Using SPX option-chain data, I:

1. construct and smooth the implied volatility smile across strikes,
2. recover the risk-neutral density using the **Breeden–Litzenberger formula**,
3. validate the fitted option surface using no-arbitrage relationships such as **put–call parity**,
4. compare the implied distributions with subsequent market outcomes, and
5. investigate transformations from risk-neutral probabilities to estimates of the physical, or real-world, distribution.

The analysis is performed across multiple time horizons, including approximately **1-, 7-, and 28-day expirations**, using several years of SPX option data.

For the full analysis, see [`project.ipynb`](project.ipynb) or the rendered [`project.html`](project.html).

---

## 1. Recovering the Risk-Neutral Distribution

For European call options, Breeden and Litzenberger showed that the risk-neutral probability density of the terminal underlying price can be recovered from the curvature of option prices with respect to strike:

\[
f_Q(K)
=
\frac{1}{D(T)}
\frac{\partial^2 C(K,T)}{\partial K^2},
\]

where

- \(C(K,T)\) is the call price,
- \(K\) is the strike,
- \(T\) is the expiration,
- \(D(T)\) is the discount factor, and
- \(f_Q\) is the risk-neutral density.

Intuitively, the shape of option prices across strikes contains information about how the market prices different possible terminal values of the index.

The difficulty is that market option prices are observed only at discrete strikes and contain bid-ask noise. Since numerical second derivatives amplify this noise, directly differentiating observed option prices produces unstable probability densities.

The option surface therefore needs to be smoothed before applying Breeden–Litzenberger.

---

## 2. Smoothing the Option Surface

Rather than differentiating raw option prices, the project first converts prices to implied volatilities and fits a smooth volatility smile across log-moneyness.

The fitted volatility curve is then converted back into a dense set of option prices, from which the second derivative with respect to strike can be estimated numerically.

The pipeline is approximately

\[
\text{Option Quotes}
\rightarrow
\text{Implied Volatility}
\rightarrow
\text{Smoothed Volatility Smile}
\rightarrow
\text{Smoothed Option Prices}
\rightarrow
f_Q(S_T).
\]

This step is particularly important because the final density depends on the **second derivative** of the fitted price curve. A surface that appears visually reasonable can still produce economically unreasonable densities if its local curvature is poorly behaved.

---

## 3. Put–Call Parity and Surface Validation

Calls and puts with the same strike and expiration are linked by the no-arbitrage relationship

\[
C(K,T)-P(K,T)
=
D(T)(F_{0,T}-K),
\]

where \(F_{0,T}\) is the forward price of the S&P 500.

For a fixed expiration,

\[
C(K)-P(K)=DF-DK,
\]

so call-minus-put prices should be approximately linear in strike.

This relationship provides two useful checks on the smoothing procedure.

First, the forward price and discount factor can be estimated directly from option prices. If

\[
C(K)-P(K)=a+bK,
\]

then

\[
D=-b,
\qquad
F=-\frac{a}{b}.
\]

Second, fitted calls and puts should continue to satisfy put–call parity after smoothing. The residual

\[
\varepsilon(K)
=
\hat C(K)-\hat P(K)-D(F-K)
\]

should therefore remain small across strikes.

This matters because calls and puts ultimately encode the same terminal risk-neutral distribution. Large discrepancies between the two sides of the market can indicate noisy quotes, poor smoothing, or violations of arbitrage constraints.

---

## 4. Risk-Neutral vs. Physical Probabilities

The distribution obtained from option prices is a **risk-neutral distribution**, not necessarily the market's literal forecast of future outcomes.

Under the risk-neutral measure \(Q\), assets are priced as though their expected return is the risk-free rate. Real-world outcomes instead occur under the physical probability measure \(P\).

In general,

\[
f_Q(S_T) \neq f_P(S_T).
\]

The difference reflects investors' preferences toward risk and the prices they are willing to pay to insure against different states of the world.

This project explores utility-based transformations of the extracted risk-neutral density to investigate whether an estimate of the physical distribution can better describe subsequent SPX outcomes.

---

## 5. Evaluating the Distributions

A probabilistic forecast should not be evaluated solely by comparing its mean or mode with the eventual index level. The full probability distribution must be evaluated.

I therefore compare the extracted distributions with subsequent market outcomes using the **Continuous Ranked Probability Score (CRPS)**.

For a forecast CDF \(F\) and realized outcome \(y\),

\[
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left(F(x)-\mathbf{1}\{x\ge y\}\right)^2dx.
\]

Lower CRPS values indicate that more probability mass was assigned near the outcome that ultimately occurred.

By repeating this calculation across many option expirations, the project evaluates whether the extracted distributions systematically provide useful information about future market outcomes rather than judging individual forecasts in isolation.

---

## Questions Explored

The project is motivated by several questions:

- What probability distribution is implied by the cross-section of SPX option prices?
- How sensitive is the recovered density to the method used to smooth option prices?
- Do calls and puts imply consistent terminal distributions?
- How does the shape of the implied distribution change with time to expiration?
- How well do risk-neutral distributions describe subsequent market outcomes?
- Can a transformation from the risk-neutral measure \(Q\) to the physical measure \(P\) improve distributional forecasts?

---

## Repository Structure

```text
.
├── project.ipynb        # Main analysis and results
├── project.html         # Rendered notebook
├── project.pdf          # PDF version of analysis
├── black_scholes.py     # Black-Scholes pricing and implied-volatility functions
├── data/                # SPX option-chain data
└── README.md
