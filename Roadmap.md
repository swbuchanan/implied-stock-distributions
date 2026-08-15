# Future Extension: SPX Put-Spread Trading Signal

> **Status:** Parked until the core density-extraction and forecasting project is improved.

## Research Question

Do differences between an estimated real-world distribution \(P\) and the option-implied distribution \(Q\) identify mispriced SPX put spreads after bid–ask costs?

## Prerequisites

- [ ] Use the correct SPX/SPXW settlement values.
- [ ] Estimate the market-implied forward using put–call parity.
- [ ] Reconcile call and put data into a consistent option-price curve.
- [ ] Preferably construct an arbitrage-free implied distribution.

## Initial Strategy

- Use options with approximately 28 days to expiration.
- Enter positions on one predetermined day each week.
- Select put strikes near the 25th and 10th percentiles of \(Q\).
- Trade defined-risk vertical put spreads.
- Hold each position to settlement.
- Size positions to have the same maximum possible loss.

## Estimate the Real-World Distribution

- Begin with the CRRA-adjusted density from the existing project.
- Estimate or select the risk-aversion parameter \(\gamma\) using past data only.
- Later compare the CRRA density with:
  - A historical-return distribution.
  - A Student-\(t\) distribution.
  - A GARCH-based forecast.

## Calculate the Model Value

For strikes \(K_L < K_H\), define the payoff of a long put spread as

\[
g(S_T) = (K_H-S_T)^+ - (K_L-S_T)^+.
\]

Calculate its estimated value under \(P\):

\[
V_t^P = e^{-rT}\int g(s)\widehat{p}_t(s)\,ds.
\]

Use executable market prices:

\[
\text{Long cost}
=
\operatorname{ask}(K_H)-\operatorname{bid}(K_L),
\]

\[
\text{Short credit}
=
\operatorname{bid}(K_H)-\operatorname{ask}(K_L).
\]

Include the SPX contract multiplier when calculating P&L.

## Trading Rule

- Buy the put spread when its model value exceeds its executable purchase price by a preset buffer.
- Sell the put spread when its executable credit exceeds its model value by the buffer.
- Otherwise, do not trade.
- Fix the buffer before evaluating the final test period.

## Chronological Backtest

- **2020–2021:** Estimate the model.
- **2022:** Select parameters and the trading threshold.
- **2023:** Run the untouched final test.
- Account for overlapping 28-day positions when calculating uncertainty and performance statistics.

## Benchmarks

Compare the signal with:

- Always selling the same put spread.
- Always buying the same put spread.
- Never trading.
- Trading from a simple historical-return forecast instead of the CRRA model.

The always-sell benchmark is especially important because the strategy might otherwise merely reproduce the historical premium from selling crash insurance.

## Performance Measures

Report:

- Net P&L using bid and ask prices.
- Number of trades.
- Average market exposure.
- Sharpe ratio or another risk-adjusted measure.
- Maximum drawdown.
- Worst individual trade.
- Expected shortfall.
- Performance in rising, falling, and high-volatility markets.
- Block-bootstrap confidence intervals.

## Robustness Checks

Test:

- Different values of \(\gamma\).
- Nearby strike pairs.
- Nearby signal thresholds.
- Non-overlapping entry dates.
- Results after excluding the strongest-performing period.
- Whether the CRRA and historical-return forecasts agree on the direction of the signal.

## Main Interpretation Risk

Determine whether the signal genuinely identifies unusually priced options or merely earns the usual premium from continually selling downside insurance.

A successful signal should improve on the always-sell benchmark through better risk-adjusted returns, smaller drawdowns, or reduced tail losses—not simply positive average P&L.

## Data Limitation

The current snapshots can support an entry-at-quoted-price, hold-to-expiration backtest.

The following extensions would require additional option-chain data:

- Next-session execution.
- Daily mark-to-market valuation.
- Dynamic position exits.
- Delta hedging.
- Delta-hedged volatility strategies.

## Project Organization

Implement this extension in a separate notebook or module so the current project remains focused on option-implied density extraction and probabilistic forecast evaluation.
