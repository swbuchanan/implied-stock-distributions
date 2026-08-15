# Research Improvement Roadmap

## Status and purpose

This is the roadmap for turning the certification project into a defensible research project for quantitative-trading applications.

The central question is:

> **Does a parsimonious pricing-kernel adjustment, estimated ex ante from past SPX risk-neutral densities and subsequent realized returns, improve the out-of-sample calibration and proper-score performance of physical return-density forecasts? Are the adjustment and its forecast gains stable across horizons, volatility regimes, and time?**

The existing [`Roadmap.md`](Roadmap.md) describes a possible put-spread strategy. Keep that extension parked until this roadmap is complete. A reliable negative result from the density-forecasting study is more valuable than a trading backtest built on an unreliable density.

This document is intentionally detailed. Each phase explains:

- what to build,
- why it matters,
- what concepts to learn,
- what evidence to save, and
- what must be true before moving on.

## Recommended scope

To keep the project achievable:

- Make approximately **28-day forecasts** the primary experiment.
- Use approximately **7-day forecasts** as the main robustness check.
- Treat **1-day forecasts** as optional. Settlement timing, quote noise, and small time-value make this the hardest horizon rather than the easiest.
- Use 2010–2023 to study different market regimes, but only after the data and density pipeline is correct.
- Start with one learned CRRA parameter per horizon.
- If time permits, add one small, interpretable machine-learning model only after the CRRA experiment works end to end.
- Do not add a trading strategy until the forecasting results are complete.

The project does not need to prove that a stable kernel exists. These are all useful conclusions:

1. A stable adjustment improves forecasts.
2. The adjustment works only in particular horizons or regimes.
3. A flexible statistical recalibration works but the economic CRRA model does not.
4. No adjustment reliably improves out-of-sample forecasts.

A carefully established negative result demonstrates good research judgment.

## Roadmap at a glance

| Phase | Main outcome | Priority | Dependency |
|---|---|---:|---|
| 0. Freeze the research design | Written hypotheses and rules fixed before final testing | Essential | None |
| 1. Build trustworthy market data | Correct contracts, times, quotes, forwards, and settlements | Essential | Phase 0 |
| 2. Recover valid risk-neutral distributions | One arbitrage-consistent distribution per forecast | Essential | Phase 1 |
| 3. Build the forecast panel | A leak-free table linking each forecast to its later outcome | Essential | Phase 2 |
| 4. Implement honest benchmarks | Raw option and simple historical competitors | Essential | Phase 3 |
| 5. Learn the CRRA adjustment | First direct answer to the research question | Essential | Phase 4 |
| 6. Add a small ML extension | Interpretable nonlinear or regime-dependent adjustment | Differentiator | Phase 5 |
| 7. Evaluate calibration and skill | Scores, coverage, PITs, and uncertainty | Essential | Phases 4–6 |
| 8. Test stability | Rolling parameters and regime-specific performance | Essential | Phase 7 |
| 9. Harden the repository | Tests, deterministic pipeline, clean artifacts, documentation | Essential | Start early; finish here |
| 10. Communicate the result | Results-led README, report, figures, and resume bullet | Essential | Phases 0–9 |

Do not treat the phases as a race. Each phase has a **release gate**. If a gate fails, diagnose the problem there instead of allowing it to contaminate every later result.

### Practical build sequence

Treat the phases as seven work blocks rather than trying to rewrite everything simultaneously:

1. Archive the course version, safely preserve the current work in progress, audit what data may be published, freeze the research protocol, and create a tested package skeleton.
2. Resolve contract identity and settlement values; then implement quote filters and parity-implied forwards.
3. Build and validate one arbitrage-consistent risk-neutral estimator on a small set of difficult dates.
4. Extend the validated data and density pipeline to 2010–2023; then create the normalized-return forecast panel and simple benchmarks.
5. Fit the learned walk-forward CRRA model and complete proper-score, calibration, uncertainty, and stability analysis. After applying the Phase 9 and 10 release gates to this core, this is the first **application-ready checkpoint**.
6. Add the regularized density-ratio spline only if the core pipeline is reliable and the effective training sample is adequate.
7. Run the final robustness checks and rebuild every public artifact from one frozen configuration.

If application deadlines arrive after block 5, publish the rigorous core and describe the ML model as future work. A smaller completed study is stronger than a larger partly validated one.

---

## The statistical object being studied

Let $X_{t,h}$ be a future SPX outcome over horizon $h$. For example, it can be the forward-normalized log return defined later in this roadmap. Let

- $q_{t,h}(x)$ be its risk-neutral density inferred from option prices,
- $p_{t,h}(x)$ be its unknown physical, or real-world, density,
- $D_{t,h}$ be the discount factor, and
- $\bar m_{t,h}(x)$ be the stochastic discount factor projected onto the terminal SPX outcome.

Their relationship is

\[
\bar m_{t,h}(x)
=
D_{t,h}\frac{q_{t,h}(x)}{p_{t,h}(x)}.
\]

Ignoring a scale factor that disappears during normalization, a proposed kernel shape $m_\theta(x)$ therefore creates a candidate physical forecast

\[
p_{\theta,t,h}(x)
=
\frac{q_{t,h}(x)/m_\theta(x)}
{\int q_{t,h}(u)/m_\theta(u)\,du}.
\]

Under the strong CRRA assumption $m(R)\propto R^{-\gamma}$, where $R$ is a positive gross market return,

\[
p_{\gamma,t,h}(R)
\propto
q_{t,h}(R)R^\gamma.
\]

This is best described as an **SPX index-state pricing-kernel adjustment**. Options identify $q$, not $p$ or the full economy-wide stochastic discount factor. The later realized returns are also required to estimate the adjustment. The fitted value of $\gamma$ should not automatically be interpreted as the literal risk aversion of a representative investor.

### Important vocabulary

**Calibration** asks whether events happen with the advertised frequencies. If an 80% interval is well calibrated, about 80% of outcomes should fall inside it.

**Sharpness** asks whether a forecast is concentrated rather than needlessly wide. A distribution can obtain good coverage simply by being extremely broad, so calibration alone is not enough.

**A proper score** rewards both honest calibration and useful concentration. CRPS and the logarithmic score are proper scores.

**Out of sample** means the observation being predicted was not used to choose parameters, features, filters, or hyperparameters.

**Stable** should mean more than a visually smooth parameter plot. This project will examine both:

- **parameter stability:** whether fitted kernel parameters change materially over time; and
- **transport stability:** whether a rule fitted in earlier data continues to improve forecasts in later regimes.

---

# Phase 0: Freeze the question and experimental rules

## Goal

Write down what will be predicted, what will be compared, and what would count as supporting or rejecting the hypotheses before looking at the final results.

## Why this comes first

It is easy to try many filters, horizons, parameters, and score definitions and then report only the combination that worked. Even without dishonest intent, that process overfits the research design to historical noise. A short research protocol makes the final result more credible.

## Tasks

- [ ] Inventory the current Git state before restructuring. Preserve both the last committed course version and the modified or untracked work in progress; review and stage files individually rather than using bulk add, clean, reset, or checkout commands.
- [ ] Preserve the original certification notebook as a clearly labeled archived artifact.
- [ ] Determine the vendor data’s redistribution terms before committing additional option files or creating a public release. If already-tracked data cannot legally remain in public history, make a safe private backup and plan a clean public repository rather than casually rewriting the working repository.
- [ ] Establish the minimum reproducibility scaffold now: a package skeleton, supported Python version and lock file, versioned experiment configuration, data-manifest format, a few mathematical invariant tests, and read-only CI running an offline synthetic smoke test.
- [ ] Separate data acquisition from analysis. The frozen experiment should read checksum-verified local inputs and run without live downloads.
- [ ] State that 28 days is the primary horizon, seven days is the principal robustness horizon, and one day is optional.
- [ ] Choose a deterministic forecast schedule after inspecting historical coverage. A sensible candidate is one forecast origin per week rather than every daily chain.
- [ ] Define the exact option-maturity selection rule, such as the closest eligible expiration to 28 calendar days within a fixed tolerance.
- [ ] Freeze at most three quote-time regime features before the final experiment if the optional conditional model will be attempted. Do not search feature subsets after seeing test results.
- [ ] Define the target as the correctly settled, forward-normalized log return

  \[
  X_{t,h}=\log\left(\frac{S_T^{\text{settle}}}{F_{t,T}}\right).
  \]

- [ ] Define the primary competitors before fitting anything:

  1. unadjusted risk-neutral distribution,
  2. fixed $\gamma=4$ CRRA transformation as a literature-inspired reference,
  3. learned constant CRRA transformation,
  4. simple historical physical-density forecast,
  5. EWMA or Student-$t$ physical-density forecast.

  If the optional ML phase will be completed, add exactly one regularized density-ratio spline to this frozen comparison set before inspecting its final results.

- [ ] Define primary evaluation measures:

  - mean CRPS and CRPS skill,
  - mean logarithmic score if tail handling makes it reliable,
  - PIT calibration,
  - 50%, 80%, and 95% interval coverage, and
  - left-tail coverage as a prespecified secondary diagnostic.

- [ ] State that all parameter estimation and feature scaling will be chronological.
- [ ] State that, at forecast time $t$, only training examples whose settlement has already occurred by $t$ are available. Merely having an earlier quote date is not sufficient.
- [ ] Prespecify a minimum initial training span or number of completed forecasts and a refitting cadence, such as monthly or quarterly.
- [ ] Decide which subperiods or regimes will be reported before viewing their results. Examples are pre-COVID, the COVID shock, and post-2020, or low/middle/high implied-volatility regimes based on thresholds calculated from training data.
- [ ] Record that 2020–2023 has already influenced the project. Results for those years can be described as walk-forward or pseudo-out-of-sample, but not as a pristine untouched holdout.
- [ ] If newer option data later become available, seal it as a genuinely untouched final test before downloading outcomes.

## What to learn

- The difference between an exploratory analysis and a confirmatory test.
- Look-ahead bias, data leakage, and researcher degrees of freedom.
- Why ordinary shuffled train/test splits are inappropriate for market time series.

## Deliverable

Create a short `research_protocol.md` or configuration file containing every decision above. Changes made later should be logged with a reason.

## Release gate

Do not begin final model comparisons until another person could read the protocol and reproduce which forecasts, models, and metrics will be reported without asking you to make new choices.

---

# Phase 1: Build a trustworthy market-data and settlement layer

## Goal

Create one clean record for every option quote and one correct realized settlement value for every expiration being studied.

## Why this matters

A sophisticated model cannot repair a wrong outcome. Standard SPX and SPX Weeklys can have different settlement conventions. Comparing an AM-settled option distribution with the ordinary SPX closing price changes the event that was supposedly forecast.

## 1.1 Preserve contract identity

- [ ] Inspect the original data source for fields identifying the option root, series, expiration style, and settlement type.
- [ ] Keep these identifiers during preprocessing rather than retaining only price and strike columns.
- [ ] Add explicit fields such as:

  - `option_root`,
  - `expiration`,
  - `settlement_type`,
  - `quote_timestamp`,
  - `settlement_timestamp`, and
  - `contract_key`.

- [ ] Confirm how OptionsDX represents cases where traditional SPX and SPXW contracts share the same expiration date.
- [ ] If the source cannot distinguish them, do not guess. Either obtain better identifiers or restrict the main sample to expirations whose settlement convention is unambiguous. Report the excluded observations.
- [ ] Make this a hard feasibility decision before modeling: verify the vendor’s treatment of coincident SPX/SPXW expirations, acquire symbol-level data, or restrict the main study to unambiguous expirations.
- [ ] Do not pool AM- and PM-settled series in the headline model merely because they share a date. A changing mix of mechanically different targets could masquerade as kernel instability.

## 1.2 Obtain the correct realized value

- [ ] Use the official special opening quotation for AM-settled standard SPX options.
- [ ] Use the appropriate official closing settlement for PM-settled SPXW options.
- [ ] Save both the value and its source in a settlement table.
- [ ] Add tests around third-Friday expirations, holidays, end-of-month expirations, daylight-saving changes, and missing values.
- [ ] Never download settlement data dynamically inside the final report. Cache a versioned input so the same commit produces the same result.

## 1.3 Measure time correctly

- [ ] Calculate time to settlement from actual timestamps, not simply `DTE / 365`.
- [ ] Document the day-count convention.
- [ ] Verify that quote timestamps have the intended timezone.
- [ ] Use the true time remaining to AM or PM settlement.

## 1.4 Replace blanket missing-value removal with explicit quote filters

For each option side, retain a row because its required fields are valid, not because every unrelated column happens to be present.

- [ ] Require finite strike, bid, ask, underlying, expiration, and quote timestamps.
- [ ] Require `ask >= bid >= 0`.
- [ ] Decide and document how zero bids are handled.
- [ ] Calculate absolute and relative bid–ask spreads.
- [ ] Remove or downweight quotes with extremely wide spreads using a rule fixed from market-quality considerations, not later forecast results.
- [ ] Check elementary option-price bounds.
- [ ] Prefer liquid out-of-the-money puts below the forward and calls above the forward. They generally contain less intrinsic value and are often the more informative side of the market.
- [ ] Keep a reason code for every rejected quote.

Midpoints are reasonable observations for extracting a market distribution, but they are measured with uncertainty. A quote with a five-point spread should influence the fit less than a quote with a five-cent spread.

## 1.5 Infer the discount factor and forward from put–call parity

For matched calls and puts,

\[
C(K)-P(K)=D(F-K)=a+bK,
\]

so

\[
D=-b,\qquad F=-\frac{a}{b}.
\]

- [ ] Match calls and puts by strike and expiration.
- [ ] Begin with relatively liquid strikes near the money.
- [ ] Estimate the line with spread-aware weights and a robust regression method so one bad quote cannot determine the forward.
- [ ] Check that the fitted discount factor and forward are economically plausible.
- [ ] Compare the implied rate with an external point-in-time short-rate curve.
- [ ] At very short horizons, recognize that $D$ is extremely close to one and its regression slope can be dominated by bid–ask noise. If joint estimation is unstable, fix $D$ from the external curve and estimate

  \[
  F=\operatorname{weighted\ median}_K
  \left[K+\frac{C(K)-P(K)}{D}\right].
  \]

  Record this as a prespecified fallback and compare it with the joint parity regression as a diagnostic.
- [ ] Save parity residuals across strikes.
- [ ] Investigate large residuals rather than silently averaging them away.

## 1.6 Produce a data-quality report

Report by year and horizon:

- number of usable forecast dates,
- number of quotes per chain,
- strike range relative to the forward,
- percentage of zero bids,
- median relative spread,
- rejected-quote counts by reason,
- forward and discount-factor diagnostics,
- AM/PM settlement composition, and
- gaps or changes in expiration availability.

## What to learn

- SPX versus SPXW contract specifications.
- Put–call parity and synthetic options.
- Forward price versus spot price.
- Discount factors and continuous rates.
- Market microstructure noise and why midpoint quality varies.
- Robust regression and weighted least squares.

## Deliverables

- A documented data dictionary.
- A deterministic preprocessing script rather than preprocessing hidden only in a notebook.
- A settlement table with provenance.
- A data-quality report with tables and figures.

## Release gate

For every forecast included in the research sample, the project must know what contract was quoted, when it settled, what value it settled at, and why each quote was accepted. Ambiguous settlement cases must be resolved or excluded.

---

# Phase 2: Recover an arbitrage-consistent risk-neutral distribution

## Goal

Produce one valid risk-neutral distribution $q_{t,h}$ for each forecast date and expiration, together with diagnostics showing how well it represents the option market.

## Why the current clipping approach should be replaced

The second derivative in Breeden–Litzenberger amplifies small quote and smoothing errors. Clipping negative curvature to zero creates a nonnegative curve, but it does not explain why the fitted option curve violated convexity. Renormalizing a truncated curve can also assign all probability to the observed strike range even when meaningful tail mass lies outside it.

The economic constraints should be imposed while fitting the option curve or probability distribution, not repaired afterward.

## 2.1 Combine puts and calls into one market view

- [ ] Use the parity-implied $D$ and $F$.
- [ ] Convert out-of-the-money puts into synthetic calls when useful:

  \[
  C(K)=P(K)+D(F-K).
  \]

- [ ] Fit one curve or distribution to both sides rather than treating call- and put-implied densities as unrelated forecasts.
- [ ] Weight observations using bid–ask uncertainty.

## 2.2 Implement a constrained price-curve baseline

For fixed maturity, an arbitrage-free European call-price curve should be:

- nonnegative,
- decreasing as strike increases,
- convex as strike increases,
- have vertical-spread slopes between $-D$ and zero,
- bounded by basic no-arbitrage limits, and
- satisfy the finite-grid counterparts of $C(0)=DF$ and $C(K)\to0$ as $K\to\infty$.

On a strike grid, convexity means that successive slopes cannot decrease. Fit call values using constrained least squares or a constrained spline with those restrictions built into the optimization.

The slope bound is essential because it keeps the implied tail probability between zero and one. These conditions preserve the Breeden–Litzenberger interpretation:

\[
Q(S_T>K)=-\frac{1}{D}\frac{\partial C}{\partial K},
\qquad
q(K)=\frac{1}{D}\frac{\partial^2 C}{\partial K^2}.
\]

A useful numerical lesson is that CRPS and coverage mainly require a CDF, not a visually smooth PDF. The first derivative of a valid convex call curve can provide that CDF more robustly than taking a noisy second derivative solely for plotting.

## 2.3 Optional robustness estimator: discrete state prices

As a transparent alternative, choose terminal-price grid points $s_j$ and nonnegative risk-neutral probabilities $\pi_j$. Fit them directly to calls and puts:

\[
\widehat C(K_i)=D\sum_j \pi_j(s_j-K_i)^+,
\]

\[
\widehat P(K_i)=D\sum_j \pi_j(K_i-s_j)^+.
\]

Use constraints

\[
\pi_j\geq0,\qquad \sum_j\pi_j=1,\qquad \sum_j\pi_js_j=F.
\]

This is the discrete counterpart of recovering state prices. It automatically creates a valid probability distribution and forward moment. Add a modest smoothness or entropy penalty because many probability vectors can price a finite set of options nearly equally well.

Use this method as either the production estimator or, if time permits, an independent check on the constrained call-curve estimator. The essential requirement is one estimator that passes synthetic-recovery, quote-repricing, mass, moment, and boundary tests. A second estimator is a research differentiator, not a prerequisite for the core study.

If the estimator returns genuine point masses, CRPS and interval probabilities remain well defined, but an ordinary continuous log density at the realized value does not. Either convert the probabilities into prespecified finite-width bins or use a continuous production estimator for log scoring. Genuine atoms also require a randomized PIT; do not expect the ordinary $F(Y)$ to be uniform.

## 2.4 Treat tails as a model choice

Options only identify the distribution well where useful strikes are quoted. The tails beyond the last liquid strikes are not observed directly.

- [ ] Add an explicit tail rule rather than setting all outside probability to zero.
- [ ] Consider a simple parametric tail joined smoothly to the fitted central distribution or a broad state grid with a regularizing prior.
- [ ] Choose tail hyperparameters using quote reconstruction and stability within training data, never using final realized outcomes.
- [ ] Re-run the main results under at least one reasonable alternative tail rule.
- [ ] Report how much mass lies outside the observed liquid-strike interval.

## 2.5 Required diagnostics for every density

Save at least:

- probability mass error,
- nonnegativity violations,
- forward-moment error,
- option-price reconstruction error,
- percentage of fitted prices inside their bid–ask intervals,
- put–call-parity residuals,
- boundary density or boundary probability mass,
- observed strike coverage relative to $F$,
- tail mass,
- optimizer status, and
- reason for excluding a failed chain.

Create a small set of deliberately difficult example dates: quiet markets, the COVID crash, very short maturity, sparse strikes, and very wide spreads. Use them as regression fixtures so later code changes cannot silently degrade the estimator.

## What to learn

- Static arbitrage across strikes.
- Convex functions and constrained optimization.
- The inverse nature of option-density extraction.
- Regularization: accepting a slightly worse in-sample fit to obtain a more stable solution.
- Why tails are partly an assumption rather than an observed fact.
- The distinction between density, CDF, probability mass, and state price.

## Deliverables

- A reusable density-estimation module.
- Unit tests using distributions with known option prices.
- A diagnostics table with one row per forecast.
- Optionally, a comparison of the constrained curve and discrete state-price methods on representative chains.

## Release gate

The accepted densities must have nonnegative mass, unit total mass, a mean matching the inferred forward within a documented tolerance, small quote reconstruction error relative to spreads, valid vertical-spread probabilities, and no unexplained concentration at the boundaries. The main method must recover known synthetic examples. If a second estimator is implemented, the two should tell a broadly consistent story.

---

# Phase 3: Express every forecast on a comparable return scale

## Goal

Build a modeling table in which forecasts from SPX near 1,100 and SPX near 4,700 can be compared fairly.

## Why raw index levels are unsuitable

A CRPS of 50 index points has a very different economic scale at SPX 1,100 than at SPX 4,700. Extending the sample to 2010 makes raw-level average scores especially misleading.

Use

\[
X=\log\left(\frac{S_T}{F_{t,T}}\right).
\]

Here $X=0$ means that the terminal index equals today’s forward price. This removes the changing index level and incorporates the term’s financing and dividend carry through $F$.

If $S=Fe^X$, transform the density with the Jacobian:

\[
q_X(x)=Fe^x q_S(Fe^x).
\]

For a discrete distribution, simply transform every state $s_j$ to $x_j=\log(s_j/F)$ while retaining its probability mass.

## Tasks

- [ ] Assign every forecast a unique `forecast_id`.
- [ ] Store quote date, expiration, horizon, $D$, $F$, settlement value, and realized $X$.
- [ ] Store the risk-neutral distribution on a common $X$ grid or as discrete masses.
- [ ] Store only features that were observable at the quote timestamp.
- [ ] Store quality diagnostics from Phase 2.
- [ ] Apply a quality threshold fixed without looking at forecast performance.
- [ ] Log every excluded forecast and its reason.
- [ ] Choose a deterministic primary sampling schedule after the coverage audit.
- [ ] Keep a denser schedule as a robustness sample if desired, but handle its additional dependence during inference.
- [ ] Check how often different forecasts share the same expiration outcome.

Possible contemporaneous features for later stability models include:

- at-the-money implied volatility,
- risk-neutral skew or a simple put-skew measure,
- seven-versus-28-day volatility term structure,
- recent realized volatility,
- recent SPX drawdown,
- the implied-forward rate, and
- implied volatility minus **trailing** realized volatility measured at the quote timestamp.

Start with no more than three regime features. Every extra feature makes overfitting easier.

## What to learn

- Change of variables for probability densities.
- Forward-normalized returns.
- Feature availability and point-in-time data.
- Dependence caused by shared and overlapping outcomes.

## Deliverable

A versioned forecast panel with a documented schema. A long-form table with `forecast_id`, grid coordinate, and probability mass is often easier to audit than opaque objects stored inside a notebook.

## Release gate

Select random forecast rows and reconstruct the original terminal-price distribution, the transformed return distribution, and the realized outcome. Probability must be preserved under the change of variables, and no feature may contain information published after its quote timestamp.

---

# Phase 4: Implement benchmarks before learning a kernel

## Goal

Determine what “better” means by comparing the pricing-kernel models with simple alternatives that could plausibly solve the same forecasting problem.

## Why benchmarks matter

Beating raw $q$ only shows that a risk-neutral distribution is not a perfect physical forecast, which theory already suggests. The stronger question is whether the proposed adjustment adds value beyond simple physical-return models.

## Required benchmarks

### A. Raw option-implied distribution

Use $p_t=q_t$. In the CRRA family this is equivalent to $\gamma=0$. It measures how much the transformation changes the option-implied forecast.

### B. ATM lognormal option benchmark

Construct a simple distribution using the inferred forward and at-the-money implied volatility. Comparing it with the full option-implied distribution helps show whether the smile’s skew and tails add information.

### C. Historical Student-$t$ distribution

Estimate location and scale from trailing returns at the same horizon and use a Student-$t$ shape. This is a simple physical-density model with heavier tails than a normal distribution. Define the window, location rule, and treatment of overlapping $h$-day returns in the protocol.

### D. EWMA volatility distribution

Use exponentially declining weights so recent squared returns influence the volatility forecast more than old returns. Specify a normal or Student-$t$ innovation law, a drift/location rule, and how one-day volatility is aggregated to the target horizon. This teaches a core idea behind volatility forecasting without requiring a large model.

### E. Optional GARCH-$t$ benchmark

Add this only after the simpler historical benchmarks work. Its purpose is comparison, not to turn the project into a GARCH survey.

## Fair-comparison rules

- [ ] Give every benchmark the same forecast target and settlement outcome.
- [ ] Express every historical benchmark in the same $X=\log(S_T/F_{t,T})$ coordinates. If it begins with spot returns, shift it by the current forward carry and make the AM/PM endpoint convention consistent.
- [ ] Use the same forecast dates.
- [ ] Estimate every historical model using past observations only.
- [ ] Fit scalers and hyperparameters inside the training window.
- [ ] Use the same evaluation code for every model.
- [ ] Record model failures rather than silently changing the comparison sample.

## What to learn

- Normal versus Student-$t$ tails.
- Historical and exponentially weighted volatility.
- The difference between risk-neutral and statistical forecasting baselines.
- Why a sophisticated model must beat simple alternatives, not just a straw man.

## Deliverable

One function or class per benchmark returning a valid CDF and probability mass or density on the common return grid.

## Release gate

All benchmarks must pass common normalization tests and generate forecasts in a chronological dry run before fitting the pricing-kernel parameters.

---

# Phase 5: Learn the simplest pricing-kernel adjustment

## Goal

Estimate one parsimonious CRRA adjustment per horizon using past option-implied distributions and their subsequent realized returns.

## 5.1 Work in log-return coordinates

Because $R=(F_{t,T}/S_t)e^X$, and the factor $F_{t,T}/S_t$ is constant within a forecast and disappears during normalization, the CRRA transformation becomes an exponential tilt:

\[
p_{\gamma,t,h}(x)
=
\frac{q_{t,h}(x)e^{\gamma_h x}}
{\int q_{t,h}(u)e^{\gamma_h u}\,du}.
\]

This formula is convenient: it preserves nonnegativity, normalization is explicit, and $\gamma=0$ reproduces the raw risk-neutral forecast.

## 5.2 Estimate $\gamma$ chronologically

At each forecast origin $t$:

1. Find all older forecasts whose terminal outcomes were already known before $t$.
2. Estimate $\gamma_h$ on those completed cases only.
3. Generate $p_{\gamma,t,h}$.
4. Save the forecast before adding the new outcome to future training data.

Use an expanding window first. A rolling fixed-length window is a later stability experiment.

Estimate $\gamma_h$ separately for each horizon. A seven-day and a 28-day stochastic discount factor are different multi-period objects, so equality should be tested rather than assumed.

Do not issue a learned forecast until the protocol’s minimum training requirement is met. Re-estimating monthly or quarterly is easier to audit than changing the parameter after every overlapping observation.

## 5.3 Choose an estimation loss

Two defensible choices are:

- **negative logarithmic score:** maximum likelihood under the candidate physical density; and
- **mean CRPS:** less sensitive to a single extremely low fitted density at an outcome.

Choose one as primary in the protocol and use the other as sensitivity analysis. If the log score is used, reliable full-support tails are essential; otherwise one boundary event can dominate the fit. Do not simply floor the density at whichever outcome occurred. If numerical protection is unavoidable, prespecify the same normalized contamination model for every forecast,

\[
p_\epsilon(x)=(1-\epsilon)p(x)+\epsilon r(x),
\]

where $r$ is a broad fixed reference density and $\epsilon$ is small and fixed without using test results.

A one-dimensional grid search over a prespecified range of $\gamma$ is transparent and sufficient. There is no benefit in using a complex optimizer for one parameter.

Because the factor $e^{\gamma x}$ can magnify a weakly identified right tail, repeat the estimate under the prespecified alternative tail rule and wider return-grid endpoint. Treat a large change in $\gamma$ as a failed stability diagnostic, not as a harmless numerical detail.

## 5.4 Compare fixed and learned versions

Compare:

- $\gamma=0$: raw risk-neutral density,
- $\gamma=4$: the current literature-inspired choice,
- one expanding-window learned $\gamma_h$, and
- optionally, a rolling-window learned $\gamma_{t,h}$.

## What to learn

- Exponential tilting and normalization constants.
- Maximum likelihood and proper-score estimation.
- Expanding versus rolling estimation windows.
- Parameter uncertainty.
- Why a convenient representative-agent model need not identify literal preferences.

## Deliverables

- Tested CRRA transformation code.
- A walk-forward table showing the parameter available at each forecast origin.
- A plot of rolling or expanding $\gamma_h$ estimates with uncertainty bands.
- A comparison with $\gamma=0$ and $\gamma=4$.

## Release gate

Re-running the walk-forward process must reproduce the same forecast for every date. A test should prove that changing a future outcome cannot alter an earlier parameter estimate or forecast.

---

# Phase 6: Add a small, useful machine-learning extension

## Guiding principle

The dataset may contain hundreds of strikes per chain, but those strikes do **not** provide hundreds of independent physical outcomes. Each forecast date supplies only one later realized return, and overlapping horizons make the effective sample still smaller.

This is a small-data problem. A neural network, random forest on hundreds of density bins, or flexible normalizing flow is likely to memorize regimes and produce a complicated story without reliable evidence. The useful form of machine learning here is **regularized, chronological, and constrained to output a valid density**.

## 6.1 Recommended pricing-kernel ML model: a regularized density-ratio spline

CRRA assumes that the logarithm of the inverse-kernel weight is a straight line in the return. A small spline lets the data ask whether that line should bend. Here “stable” means one shape in the chosen forward-normalized log-return coordinate $x$, fitted separately by horizon:

\[
p_{\theta,t,h}(x)
=
\frac{q_{t,h}(x)\exp(f_{\theta,h}(x))}
{\int q_{t,h}(u)\exp(f_{\theta,h}(u))\,du},
\]

with

\[
f_{\theta,h}(x)
=
\gamma_h x+\sum_{j=1}^{J}\theta_{h,j}B_j(x).
\]

The functions $B_j(x)$ are smooth basis functions. The linear part $\gamma_hx$ is the CRRA model; the spline terms represent a small, smooth departure from it. If the base distribution covers the modeled support and the normalizing integral is finite and positive, exponentiation plus normalization produces a nonnegative distribution with total mass one.

Keep this genuinely small:

- [ ] Use roughly four to six **total** effective degrees of freedom, including the linear CRRA term, not four to six extra nonlinear terms.
- [ ] Remove the spline basis’s constant and linear components, or orthogonalize it against $\{1,x\}$, so that it cannot duplicate $\gamma_hx$. An additive constant has no effect after normalization.
- [ ] Leave $\gamma_h$ unpenalized and penalize nonlinear curvature. With weak evidence, the model should fall back toward learned CRRA.
- [ ] In the primary version, require $f_{\theta,h}(x)$ to be nondecreasing. This corresponds to the standard decreasing pricing-kernel shape. Treat a nonmonotone version as an explicitly exploratory pricing-kernel-puzzle test.
- [ ] Control extrapolation outside the well-observed return region; do not allow a fitted tail weight to explode simply because no outcome constrained it.
- [ ] Numerically test that $0<\int q(u)e^{f(u)}du<\infty$ under every modeled tail, not merely on a conveniently truncated grid.
- [ ] Select the penalty and effective degrees of freedom inside chronological validation data only.
- [ ] Fit a separate model for each horizon.

Fit the parameters by penalized log likelihood on completed past forecasts. In simple terms, the model is rewarded when it placed high probability near outcomes that later occurred, while the penalty charges it for unnecessary bends. Repeat the walk-forward fit with CRPS as a sensitivity check if computation permits.

This is the most useful ML addition because it directly tests the research question: is there a **stable but not necessarily CRRA-shaped** reweighting of $q$ that transports to future forecasts? It is more informative here than a generic classifier or neural network and remains easy to plot and explain.

## 6.2 Optional statistical benchmark: two-parameter PIT recalibration

For each completed raw risk-neutral forecast, calculate its probability integral transform:

\[
u_t=F^Q_t(x_t).
\]

If the raw risk-neutral forecasts were calibrated physical forecasts, the historical $u_t$ values would be approximately uniform on $[0,1]$. Let $G$ be the CDF learned from past PIT values. Define

\[
F^*_t(x)=G\!\left(F^Q_t(x)\right).
\]

If $G$ is monotone and maps zero to zero and one to one, $F^*_t$ is automatically a valid CDF. Its density, when derivatives exist, is

\[
p^*_t(x)=g\!\left(F^Q_t(x)\right)q_t(x).
\]

This is a learned density-ratio adjustment. It targets **marginal PIT calibration** while retaining the information in the option-implied ranking of outcomes. Uniform pooled PITs do not by themselves establish conditional calibration, and serial dependence must still be checked.

### Implementation sequence

- [ ] Begin with the empirical CDF of past PIT values as a diagnostic, not as the final model.
- [ ] If this optional benchmark is included, use a two-parameter beta-CDF map first. Do not add a separately tuned flexible spline unless the primary analysis is already complete.
- [ ] Shrink the map toward the identity, $G(u)=u$. When data are weak, the model should make a small adjustment rather than an extreme one.
- [ ] Select smoothing or regularization using inner walk-forward validation.
- [ ] Fit a separate map for each horizon.
- [ ] Compare this statistical recalibration with CRRA. Do not call the monotone map a structural market pricing kernel.

The comparison among CRRA, the stable spline, and PIT recalibration is informative:

- If all three help similarly, a simple economic adjustment may capture most of the stable distortion.
- If the stable spline wins, a low-dimensional kernel may transport, but the CRRA line is too restrictive.
- If only PIT recalibration helps, miscalibration may be repeatable without supporting a structural pricing-kernel interpretation.
- If none helps, historical distortions may not transport into the future.

## 6.3 Optional stretch: a regularized conditional exponential tilt

Only after the stable spline is complete, test whether a few observable market states add value without giving up density validity:

\[
p_{\beta,\delta,t,h}(x)
=
\frac{q_{t,h}(x)\exp(f_h(x,z_t))}
{\int q_{t,h}(u)\exp(f_h(u,z_t))\,du},
\]

where $z_t$ contains a few market-regime features known at forecast time.

A deliberately small state-dependent CRRA model is

\[
f_h(x,z_t)
=
\left(\beta_{0,h}+\delta_h^\top z_t\right)x.
\]

Interpretation:

- $\beta_{0,h}$ is the constant CRRA-like tilt.
- $\delta_h^\top z_t$ lets that tilt change modestly with volatility, skew, or recent realized volatility.
- Exponentiation guarantees positive weights.
- A finite, positive normalizer guarantees total probability one.

Fit the model by regularized negative log likelihood:

\[
\mathcal L(\beta_0,\delta)
=
-\sum_t\log p_{\beta,\delta,t,h}(x_t)
+\lambda\lVert\delta_h\rVert_2^2.
\]

The ridge penalty controlled by $\lambda$ shrinks only the state interactions toward zero, so weak evidence returns the model to constant CRRA rather than raw $q$. Fit the model separately by horizon.

### ML safeguards

- [ ] Standardize features using training data only.
- [ ] Use an inner chronological validation loop to select only $\lambda$ from a small prespecified grid.
- [ ] Use a gap or embargo so validation training never contains an outcome that had not settled at the validation forecast date.
- [ ] Use the at-most-three regime variables frozen in Phase 0; do not search subsets on the final evaluation.
- [ ] Compare with the nested constant-CRRA model, not merely raw $q$.
- [ ] Report coefficient paths or partial effects so the model remains interpretable.
- [ ] Do not promote the ML model based on in-sample likelihood.
- [ ] Do not repeatedly add features after seeing the final test result.

## What to learn

- Density-ratio estimation.
- Basis functions, smooth splines, and curvature penalties.
- PIT-based forecast recalibration.
- Monotone regression or splines.
- Exponential-family density tilting.
- Ridge regularization and the bias–variance tradeoff.
- Nested time-series cross-validation.
- Why model constraints can be more valuable than model complexity.

## Deliverables

- One regularized density-ratio spline that nests the CRRA model.
- Optionally, one two-parameter PIT recalibration benchmark.
- Optionally, one regularized conditional-tilt model if the effective sample supports it.
- A diagram showing how $q_t$ is reweighted and renormalized.
- A small hyperparameter table containing only choices made inside chronological training data.

## Release gate

The ML output must always be a valid distribution, and the entire training procedure must pass the same no-future-information test as the CRRA model. It earns a place in the final headline only if paired out-of-sample evidence shows a repeatable improvement over both raw $q$ and learned constant CRRA. Its performance against the historical benchmarks should still be reported.

---

# Phase 7: Evaluate calibration, sharpness, and forecast skill

## Goal

Answer whether each model produces better physical-density forecasts and quantify how uncertain that answer is.

## 7.1 Use rolling forecast origins

For each chronological forecast date:

1. form the information set available at that date,
2. train or update each model,
3. save its forecast,
4. move forward in time, and
5. score the forecast only after its settlement becomes known.

This is also called walk-forward or rolling-origin evaluation. Never shuffle the observations.

If model hyperparameters require tuning, use a nested version: an inner set of older rolling origins selects hyperparameters, and the outer origin measures performance.

## 7.2 Proper scores

### CRPS

For forecast CDF $F$ and outcome $y$,

\[
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left(F(x)-\mathbf 1\{x\geq y\}\right)^2dx.
\]

Lower is better. Integrate over a domain that includes the full modeled tails and the outcome. Do not stop the integral at the last quoted strike.

Report skill relative to a benchmark:

\[
\text{CRPS skill}
=
1-\frac{\overline{\text{CRPS}}_{\text{model}}}
{\overline{\text{CRPS}}_{\text{benchmark}}}.
\]

Positive skill is improvement.

### Logarithmic score

The negative log score is

\[
-\log p_t(y_t).
\]

It strongly penalizes assigning almost no density to the realized outcome. That makes it useful but highly sensitive to tail construction and numerical zeros. It also requires a continuous density or a prespecified bin-density convention; do not evaluate genuine point masses with an ordinary density log score. Treat disagreements between log score and CRPS as information rather than choosing whichever looks better.

## 7.3 Calibration diagnostics

- [ ] Plot PIT histograms with uncertainty bands appropriate for dependent data.
- [ ] Plot the empirical PIT CDF against the 45-degree line.
- [ ] Test whether PITs are approximately uniform.
- [ ] Test or diagnose serial dependence in PITs.
- [ ] Use randomized PITs if any forecast contains genuine atoms.
- [ ] Report 50%, 80%, and 95% interval coverage.
- [ ] Report average interval width as a sharpness measure.
- [ ] Check left-tail quantiles separately because downside states are especially relevant to index options.

Do not say “better calibrated” because CRPS decreased. Use PIT and coverage evidence for calibration; use CRPS and log score for overall probabilistic quality.

## 7.4 Paired uncertainty

Scores from competing models occur on the same dates, so compare paired differences:

\[
d_t=\text{score}_{A,t}-\text{score}_{B,t}.
\]

Overlapping seven- and 28-day outcomes are dependent. Use a time-block bootstrap or HAC-style standard errors. Blocks should be long enough to preserve the main overlap, and a non-overlapping forecast sample should be a robustness check.

Report:

- mean score difference,
- confidence interval,
- proportion of dates on which each model wins,
- cumulative score difference through time, and
- results with the strongest period removed.

## 7.5 Avoid a model tournament

Keep the number of headline comparisons small. If many models, features, horizons, tail rules, and regimes are tested, some will win by chance. Label unplanned analyses exploratory and consider multiple-comparison adjustments for large model families.

## What to learn

- Proper scoring rules.
- PIT calibration and interval coverage.
- Paired model comparison.
- Serial dependence, overlapping returns, block bootstrap, and HAC uncertainty.
- Statistical significance versus practical improvement.

## Deliverables

- One tidy score table with one row per model and horizon.
- Calibration plots.
- Paired score-difference plots and confidence intervals.
- A list of prespecified and exploratory analyses.

## Release gate

Every headline statement must identify the comparator, horizon, evaluation period, metric, and uncertainty. The result must not depend on one crisis window, one tail rule, or an invalid independence assumption without that limitation being stated.

---

# Phase 8: Test what “stable” actually means

## Goal

Determine whether a common pricing-kernel adjustment transports across time or whether it changes predictably with market conditions.

## 8.1 Parameter stability

- [ ] Plot expanding-window and rolling-window $\gamma_h$ estimates.
- [ ] Add block-bootstrap uncertainty bands.
- [ ] Compare estimates across horizons.
- [ ] Compare estimates across prespecified regimes.
- [ ] Run a simple structural-break diagnostic only after understanding the plots and data coverage.

Do not interpret every visible movement as a structural change. Parameter estimates are noisy, especially when the effective sample is small.

## 8.2 Transport stability

Report score differences separately for:

- time subperiods,
- low/middle/high volatility defined from quote-time information,
- quote-time drawdown or recent-trend regimes,
- crisis and ordinary periods, and
- each horizon.

The constant-kernel model is supported most strongly when it improves out-of-sample forecasts broadly. If it wins only in one regime, describe it as regime-dependent rather than stable.

A split by the **subsequent** rising or falling market may be shown as an ex-post error diagnostic, but it is not evidence that a regime-dependent forecasting rule was knowable at the quote date.

## 8.3 Compare nested explanations

The model sequence creates a useful interpretation:

1. **Raw $q$:** no physical adjustment.
2. **Constant CRRA:** one stable economic tilt.
3. **Stable density-ratio spline:** one regularized nonlinear economic tilt.
4. **Rolling CRRA:** the same simple shape with a time-varying parameter.
5. **Conditional exponential tilt:** modest state dependence.
6. **Optional beta-PIT recalibration:** low-dimensional marginal calibration correction without a strong structural interpretation.

Possible conclusions:

- Constant CRRA wins: evidence for a stable low-dimensional adjustment.
- Stable spline wins: a low-dimensional nonlinear adjustment transports better than the CRRA restriction.
- Conditional tilt wins: adjustment varies with observable regimes.
- PIT recalibration wins: marginal miscalibration is repeatable, but the structural kernel models are misspecified.
- Historical benchmark wins: option-implied $q$ may not add physical-forecast skill after a simple model.
- No model wins reliably: the mapping is unstable or too weak to estimate from this sample.

## What to learn

- Structural stability versus parameter-estimation noise.
- Conditional versus unconditional models.
- Regime dependence.
- The empirical pricing-kernel puzzle and why a nonmonotone estimated kernel need not literally imply negative investor risk aversion.

## Deliverables

- Rolling parameter figures.
- Regime-specific score table.
- A concise decision tree connecting the observed results to defensible interpretations.

## Release gate

Use the word “stable” only if both parameter and performance evidence support it. Otherwise state exactly how and where the adjustment changes.

---

# Phase 9: Make the code reproducible and reviewable

## Goal

Turn a large exploratory notebook into a small research codebase whose critical assumptions can be tested independently.

## Recommended structure

```text
.
├── pyproject.toml
├── README.md
├── Research_Roadmap.md
├── configs/
│   └── research.yaml
├── src/
│   └── implied_distributions/
│       ├── data.py
│       ├── contracts.py
│       ├── parity.py
│       ├── risk_neutral.py
│       ├── transforms.py
│       ├── baselines.py
│       ├── kernel_models.py
│       ├── evaluation.py
│       └── pipeline.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_density_examples.ipynb
│   └── 03_final_report.ipynb
├── tests/
│   ├── test_parity.py
│   ├── test_risk_neutral.py
│   ├── test_transforms.py
│   ├── test_scoring.py
│   └── test_no_leakage.py
├── results/
│   ├── tables/
│   └── figures/
└── data/
    ├── README.md
    └── manifest.csv
```

The exact names can change. The important separation is:

- modules perform calculations,
- tests verify invariants,
- configurations record research choices,
- notebooks explain and visualize results, and
- generated artifacts are reproducible from a specific configuration.

## Engineering tasks

- [ ] Preserve the original project before restructuring.
- [ ] Add a `pyproject.toml`, choose one supported Python version, and commit a lock file. A project file declares dependencies; the lock file records the exact resolved versions.
- [ ] Move reusable Black–Scholes, parity, density, kernel, and scoring logic into modules.
- [ ] Replace wildcard imports.
- [ ] Add type hints and concise docstrings for mathematical inputs and units.
- [ ] Seed any random diagnostic sampling or remove randomness from the final report.
- [ ] Put data downloads or updates behind a separate acquisition command. Freeze and checksum every external input—including option files, settlements, index outcomes, and diagnostic rate data—before running the experiment.
- [ ] Make the full research run work offline from those frozen inputs.
- [ ] Save large intermediate calculations so the final notebook renders quickly.
- [ ] Make the final report read versioned results rather than download live data or refit the full history.
- [ ] Remove notebook tracebacks, stale outputs, TODO headings, and inconsistent execution counts before release.
- [ ] Generate HTML and PDF from the same fresh-kernel notebook execution and commit them together.
- [ ] Add an automated release check that fails on notebook error outputs, unexecuted code cells, or results referring to different run identifiers.
- [ ] Do not let CI automatically commit randomly executed notebooks.
- [ ] Make CI read-only, install from the lock file, run lint and tests plus an offline synthetic end-to-end smoke case, and upload rendered artifacts without committing them.
- [ ] Add raw data, generated caches, notebook checkpoints, and `__pycache__` to `.gitignore` as appropriate.
- [ ] Document what data can legally be redistributed. If raw market data cannot be published, provide acquisition instructions, a schema, checksums, and a small permissible synthetic fixture.
- [ ] Add a data manifest recording source, acquisition time, year, row count, checksum, schema version, processing code/config version, and redistribution status.
- [ ] Make each full run write a run manifest containing the Git commit, environment-lock hash, data-manifest hashes, exact configuration, seed, training cutoffs, run identifier, and output hashes.

Document three distinct reproduction paths:

1. a public offline smoke run using legal sample or synthetic data,
2. a full headline-results rebuild using user-supplied licensed files matching documented checksums, and
3. a fast report render from compact saved result tables.

This distinction matters: a public clone cannot promise to rebuild private-data results unless the necessary inputs are legally available to the reviewer.

## Minimum test suite

### Pricing identities

- Black–Scholes call–put parity.
- Implied-volatility inversion recovers known volatility.
- Price bounds reject impossible inputs.

### Risk-neutral distribution

- Nonnegative probability.
- Unit mass.
- Mean equals forward.
- Fitted call curve is decreasing and convex, with vertical-spread slopes between $-D$ and zero.
- Finite-grid boundary conditions approximate $C(0)=DF$ and $C(K)\to0$.
- Synthetic option prices reproduce known prices within tolerance.
- Call and put inputs produce one consistent distribution.

### Transformations

- $\gamma=0$ returns $q$.
- Every kernel model remains normalized and nonnegative.
- Every density-ratio model has a finite, positive normalizer under the full tail rule.
- Change of variables preserves probability.
- PIT recalibration returns the identity when $G(u)=u$.

### Evaluation

- CRPS matches known analytical or trusted numerical examples.
- An outcome outside the observed strike range still incurs the full tail penalty.
- Log scoring rejects unsupported point-mass inputs unless a bin-density convention is explicitly supplied.
- Randomized PIT behavior is correct for a distribution with known atoms.
- Coverage calculations are correct.
- Block-resampling code preserves ordering inside blocks.

### Leakage

- A future outcome cannot change an earlier forecast.
- Training labels all have settlement timestamps earlier than the forecast timestamp.
- Feature scalers are fit only on the training window.

## What to learn

- Unit tests as executable mathematical claims.
- Separation of exploratory and production research code.
- Deterministic builds, dependency pinning, and data provenance.
- Why reproducibility includes market-data versions, not only Python versions.

## Deliverables

- Passing tests and CI.
- One documented command that rebuilds the derived results.
- One documented command that renders the final report.
- A clean clone that can run tests without private raw data.

## Release gate

Do not link the repository in applications until the public branch contains one coherent version of the code, data description, notebook, HTML/PDF, and README, with no contradictory results or unfinished execution state.

---

# Phase 10: Write the final research narrative

## Goal

Make the value of the project understandable within two minutes while preserving enough detail for a technical reviewer to audit it.

## README structure

1. Research question.
2. One-paragraph result, including a negative or mixed result if that is what occurred.
3. Why $Q\neq P$ and what a pricing-kernel adjustment does.
4. Dataset, contracts, settlement handling, sample period, and horizons.
5. Arbitrage-consistent density method and validation results.
6. Walk-forward model design and benchmarks.
7. Main score and calibration table with uncertainty.
8. Stability and regime findings.
9. Key limitations.
10. Repository structure and exact reproduction instructions.

## Recommended figures

- One clean example showing market quotes, fitted arbitrage-consistent prices, $q$, adjusted $p$, and the later outcome.
- Put–call-parity and density-validation diagnostics.
- PIT histograms or empirical PIT CDFs for the main models.
- Paired CRPS differences with confidence intervals.
- Rolling $\gamma$ estimates.
- Regime-specific performance.
- A cumulative out-of-sample score-difference plot showing when any advantage appeared or disappeared.

## Claims to avoid

- “Recovered the true physical distribution.”
- “Identified the market’s pricing kernel from option prices alone.”
- “Estimated investors’ true risk aversion.”
- “Found a stable kernel” without parameter and transport evidence.
- “Better calibrated” based only on lower CRPS.
- “Found mispricing” or “found alpha” without an executable, costed trading test.
- “Untouched test set” for years already examined while developing the project.

## Strong result language

Prefer statements such as:

> A horizon-specific CRRA adjustment estimated using completed past forecasts improved 28-day CRPS by X% relative to raw risk-neutral densities, with a block-bootstrap 95% interval of [A, B]. PIT and interval-coverage diagnostics [did/did not] show corresponding calibration improvement. The gain [persisted/did not persist] across prespecified volatility regimes.

Or, for a negative result:

> Neither a constant CRRA tilt nor a regularized statistical recalibration produced a stable out-of-sample improvement over raw risk-neutral and historical benchmarks. Estimated transformations varied materially across regimes, showing that the apparent in-sample $Q\rightarrow P$ improvement was not transportable.

Both are credible research conclusions.

## Release gate

Every number in the README should be generated by the same saved results used in the report. Every headline claim should be traceable to a table, figure, configuration, and testable piece of code.

---

# Priority tiers

## Tier A: Application-ready core

Complete these before adding more models:

- correct contract and settlement handling,
- explicit quote filtering,
- a parity-based forward and validated discount factor,
- arbitrage-consistent $q$,
- the complete 2010–2023 walk-forward sample if the public question retains its stability claim,
- forward-normalized outcomes,
- raw-$q$ and simple historical benchmarks,
- learned walk-forward CRRA adjustment,
- CRPS, PIT, coverage, and block uncertainty,
- rolling and regime-specific stability evidence,
- tests and a clean deterministic report, and
- a concise results-led README.

This is already a strong resume project.

## Tier B: Research differentiators

- regularized stable density-ratio spline,
- optional two-parameter PIT recalibration,
- comparison of two risk-neutral-density estimators,
- a newly acquired sealed holdout if possible.

## Tier C: Optional extensions

- GARCH-$t$ and additional physical benchmarks,
- regularized conditional exponential tilt,
- richer state-dependent pricing kernels,
- weighted tail scores,
- other equity indices,
- more maturities or a full maturity surface, and
- the put-spread trading experiment in the separate roadmap.

Do not begin Tier C because Tier A results are disappointing. First determine whether the disappointment is a bug, uncertainty, or the genuine research answer.

---

# Suggested learning order

Read narrowly as each concept becomes necessary:

1. **SPX contract mechanics:** Cboe’s [SPX/SPXW specifications](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications) and [AM-settlement explanation](https://cdn.cboe.com/resources/spx/Settlement_of_Standard_AM_Settled_SP_500_Index_Options.pdf).
2. **Risk-neutral density extraction:** Breeden–Litzenberger, followed by shape-constrained option estimation such as [Aït-Sahalia and Duarte](https://www.nber.org/papers/w8944.pdf).
3. **Utility and pricing-kernel transformations:** [Bliss and Panigirtzoglou](https://www.chicagofed.org/~/media/publications/working-papers/2001/wp2001-15r3-pdf.pdf).
4. **Why stability is not guaranteed:** [Rosenberg and Engle](https://archive.nyu.edu/bitstream/2451/26919/2/wpa99014.pdf) and [Jackwerth](https://pages.stern.nyu.edu/~dbackus/Disasters/Jackwerth%202000.pdf).
5. **Density-forecast evaluation:** [Diebold, Gunther, and Tay](https://www.nber.org/papers/t0215) for PIT evaluation and [Gneiting and Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) for proper scoring rules.
6. **Forecast recalibration:** the PIT-based construction in [Recalibrating Probabilistic Forecasts](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1010771).
7. **Chronological validation:** a concise [rolling-origin cross-validation explanation](https://robjhyndman.com/hyndsight/tscv/).

For every paper, answer four questions in your own notes:

1. What is observed?
2. What is assumed?
3. What is estimated?
4. What evidence could show that the method failed?

That habit is directly transferable to quantitative research work.

---

# Final definition of done

The project is ready to feature prominently when:

- [ ] the target settlement value is correct for every evaluated contract,
- [ ] the risk-neutral distributions satisfy mass, moment, parity, and quote-fit diagnostics,
- [ ] all forecasts are expressed on a comparable return scale,
- [ ] every learned forecast is genuinely chronological,
- [ ] simple benchmarks are included,
- [ ] calibration and proper-score performance are reported separately,
- [ ] uncertainty accounts for overlapping observations,
- [ ] stability is tested rather than assumed,
- [ ] if an ML model is included, it is interpretable and demonstrably out-of-sample,
- [ ] negative or mixed findings are reported honestly,
- [ ] the notebook, HTML/PDF, README, code, and saved tables agree, and
- [ ] a reviewer can reproduce the public portion of the analysis from documented inputs and commands.

At that point, the project will demonstrate derivatives knowledge, numerical optimization, probabilistic forecasting, time-series validation, restrained machine learning, software testing, and research judgment—the combination that makes it relevant to prop-trading applications.
