* Built a Python pipeline to recover **risk-neutral probability densities from historical SPX option chains (2005–2024)**, estimating forwards and discount rates via put-call parity and applying the Breeden–Litzenberger relation to shape-constrained option-price curves.

* Developed and evaluated **probabilistic forecasts of SPX settlement prices** across ~30/60/90-day horizons using CRPS; implemented a CRRA pricing-kernel transformation from risk-neutral to physical densities and an empirical procedure for estimating the risk-aversion parameter (\gamma).

**Once the final out-of-sample analysis is complete, strengthen the second bullet to:**

* Estimated a CRRA pricing kernel from historical option-implied distributions and tested transformed physical densities out of sample across 30/60/90-day horizons, achieving **[X%] improvement in CRPS** versus the raw risk-neutral benchmark.


---



Options-Implied Probability Distributions — Python, NumPy, SciPy, pandas
• Recovered risk-neutral SPX terminal-price densities from 2005–2024 option chains using put-call parity, no-arbitrage shape constraints, and Breeden–Litzenberger differentiation.
• Built a probabilistic forecasting framework using CRPS and pricing-kernel/CRRA transformations to estimate physical return distributions and empirically infer market risk aversion.
