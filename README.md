````markdown
# SPX Option-Implied Probability Distributions

This project studies how probability distributions for future S&P 500 levels can be extracted from SPX option prices and how well they match realized outcomes.

From historical option chains, the project:

- estimates forward prices and discount rates via **put-call parity**,
- recovers **risk-neutral densities** using the Breeden–Litzenberger relationship,
- compares smoothing methods for noisy option data,
- evaluates forecasts using the **Continuous Ranked Probability Score (CRPS)**, and
- explores a simple transformation from risk-neutral to **physical probabilities** using a CRRA pricing kernel.

The implementation is in Python, with analysis presented in Jupyter notebooks.

---

## Motivation

Under the risk-neutral measure $Q$, a European call price is

$$
C(K,\tau) = e^{-r\tau}\mathbb{E}_Q[(S_T-K)^+].
$$

Differentiating twice with respect to strike yields the Breeden–Litzenberger result:

$$
f_Q(K) = e^{r\tau}\frac{\partial^2 C(K,\tau)}{\partial K^2},
$$

which links option prices to the risk-neutral density.

In practice, this is challenging because option quotes are discrete and noisy, and second derivatives amplify irregularities. A key issue is therefore constructing smooth call-price curves that remain arbitrage-consistent.

---

## Methodology

### 1. Forward and discount rate estimation

Put-call parity implies

$$
C(K)-P(K)=D(F-K),
$$

so if

$$
C(K)-P(K)=a+bK,
$$

then

$$
D=-b, \qquad F=\frac{a}{D}, \qquad r=-\frac{\log D}{\tau}.
$$

This allows forward prices and discount rates to be inferred directly from option data.

---

### 2. Risk-neutral density estimation

#### Implied-volatility smoothing

Prices are converted to implied volatilities and smoothed as a function of log-forward-moneyness:

$$
k=\log\left(\frac{K}{F}\right).
$$

The smoothed surface is converted back to prices and differentiated to obtain a density. This is flexible but does not guarantee arbitrage-free prices.

#### Shape-constrained smoothing (Fengler)

A valid call-price curve must satisfy:

$$
\frac{\partial C}{\partial K}\leq 0, \qquad \frac{\partial^2 C}{\partial K^2}\geq 0.
$$

These conditions ensure that the implied density

$$
f_Q(K)\propto C''(K)
$$

is non-negative. The Fengler approach enforces these constraints directly while fitting a smooth curve, producing an arbitrage-consistent density.

---

## Forecast evaluation

Each implied distribution is compared to the realized SPX level using the CRPS:

$$
\int_{\mathbb{R}}\left(F_D(x)-\mathbf{1}_{x\geq y}\right)^2 dx.
$$

Lower values indicate better probabilistic forecasts. The analysis considers 30-, 60-, and 90-day horizons.

---

## From risk-neutral to physical probabilities

Risk-neutral densities are pricing objects, not real-world forecasts. To explore this gap, I use a CRRA-based transformation:

$$
f_P(s)\propto s^\gamma f_Q(s),
$$

where $\gamma$ controls the pricing-kernel adjustment. Different values of $\gamma$ are evaluated using CRPS.

---

## Repository structure

```text
project/
│
├── README.md
├── data/
│   ├── raw/
│   └── processed/
│
├── implied_stock_distributions/
│   └── ...   # core modeling code
│
├── notebooks/
│   ├── 01_processing.ipynb
│   └── 02_project.ipynb
│
└── reports/
    └── ...
````

---

## Key tools

* Python (NumPy, pandas, SciPy, Matplotlib)
* Black–Scholes implied volatility
* put-call parity
* Breeden–Litzenberger density extraction
* constrained spline fitting
* CRPS for probabilistic evaluation

---

## Key ideas

* option-implied state prices
* risk-neutral vs physical measures
* volatility surface smoothing
* static arbitrage constraints
* probabilistic forecasting

---

## Limitations and extensions

Results depend on smoothing choices, grid resolution, and tail treatment. The CRRA transformation is also intentionally simple and not calibrated out-of-sample.

Possible extensions include:

* out-of-sample evaluation of $\gamma$
* richer pricing-kernel models
* joint strike–maturity surface fitting
* improved tail modeling
* comparison with ML-based forecasts

---

## References

Breeden & Litzenberger (1978) — *Prices of State-Contingent Claims Implicit in Option Prices*

Fengler — *Arbitrage-Free Smoothing of the Implied Volatility Surface*

---

## About

This project explores how option prices encode full probability distributions of future market outcomes, and how these distributions compare to realized returns.

```
```
