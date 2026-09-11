"""End-to-end demonstration on synthetic chains, with no vendor data.

    python -m implied_stock_distributions.demo

Generates synthetic SPX-like chains whose true density is known in closed
form, runs the whole pipeline over them -- parity fit, both smoothers,
CRPS, the inference, the CRRA tilt -- and prints the numbers plus one
figure. It takes about a minute.

The point is not the demonstration. It is that a reader can check the code
does what the notebook says it does, which is otherwise impossible here
because the raw chains are not redistributable. Two things fall out that
the real sample cannot give at all:

* the recovered density is compared against the *true* one, in CRPS and in
  total variation, so the recovery error of the estimator is measured
  rather than assumed; and
* the outcomes are drawn from a CRRA tilt of the true risk-neutral density
  at a known gamma, so Part 3's estimator has something to be right or
  wrong about.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import lognormal_forecast
from .errors import ChainError, failure_reason
from .evaluation import (
    cdf_from_density,
    common_strike_grid,
    crps_from_cdf,
    evaluate_distribution,
    gamma_plausible_interval,
    recentre_on_forward,
    relative_crps_curve,
)
from .fitting import (
    construct_risk_neutral_distribution_from_spline,
    fit_shape_constrained_call_spline,
    select_call_equivalent_quotes,
)
from .forwards import estimate_forward_and_rate
from .smoothing import create_density_from_chain
from .stats import (
    block_bootstrap_mean,
    diebold_mariano,
    effective_sample_size,
    one_per_expiration,
    pit_uniformity,
    pooled_hac_lag,
    skill_score,
)
from .synthetic import SyntheticMarket, generate_sample
from .validation import risk_neutral_skewness


def score_chains(chains, settlement, models, roughness_penalty=100.0, verbose=True):
    """Fit and score every synthetic chain, both methods plus benchmarks."""
    rows, cached, failures = [], [], []
    started = time.time()

    for index, (key, chain) in enumerate(chains, start=1):
        data_date, expiration, target = key

        if verbose and index % 50 == 0:
            print(f"  {index}/{len(chains)} chains  ({time.time()-started:.0f}s)")

        realized = settlement.get((expiration.year, expiration.month))
        if realized is None:
            failures.append({"key": key, "reason": "no_settlement_value"})
            continue

        try:
            estimates, _ = estimate_forward_and_rate(chain)
            forward = estimates["forward"]
            quotes = select_call_equivalent_quotes(
                chain, forward, estimates["discount_factor"]
            )

            fit = fit_shape_constrained_call_spline(
                chain,
                forward=forward,
                risk_free_rate=estimates["risk_free_rate"],
                roughness_penalty=roughness_penalty,
                quotes=quotes,
            )

            # One grid for every forecast scored on this chain.
            grid = common_strike_grid(
                forward,
                extra_points=np.concatenate([
                    fit["breakpoints"] * forward,
                    quotes["strike_price"].to_numpy(dtype=float),
                ]),
            )

            constrained = construct_risk_neutral_distribution_from_spline(
                fit, strike_grid=grid
            )
            iv_result = create_density_from_chain(
                chain,
                forward=forward,
                risk_free_rate=estimates["risk_free_rate"],
                quotes=quotes,
                strike_grid=grid,
            )

            recentred_grid, recentred_density = recentre_on_forward(
                iv_result["strike_grid"], iv_result["density"], forward
            )
            recentred_grid, recentred_cdf = cdf_from_density(
                recentred_grid, recentred_density
            )

            atm = float(
                np.asarray(iv_result["iv_of_strike"](forward)).ravel()[0]
            )
            benchmark = lognormal_forecast(
                forward, atm, estimates["time_to_expiry"], realized,
                strike_grid=grid,
            )

            # The truth, through the same quadrature. Each method returns a
            # grid that may differ from the shared one by its endpoints, so
            # the truth is tabulated on each method's own grid; now that
            # the quadrature is exact the score does not care which.
            model = models[key]
            constrained_grid = constrained["strike_grid"]
            true_cdf = model.cdf(constrained_grid)
            true_cdf = (true_cdf - true_cdf[0]) / (true_cdf[-1] - true_cdf[0])

        except (ChainError, ValueError) as exc:
            failures.append({"key": key, "reason": failure_reason(exc)})
            continue

        constrained_score = evaluate_distribution(
            constrained["strike_grid"], constrained["cdf"], realized, forward,
            density=constrained["density"],
        )
        iv_score = evaluate_distribution(
            iv_result["strike_grid"], iv_result["cdf"], realized, forward,
            density=iv_result["density"],
        )

        # Recovery error against the true density: the thing only a
        # synthetic study can measure.
        def l1_against_truth(method_grid, method_density):
            truth = model.density(method_grid)
            truth = truth / np.trapezoid(truth, method_grid)
            return float(
                np.trapezoid(np.abs(method_density - truth), method_grid)
            )

        l1_constrained = l1_against_truth(
            constrained_grid, constrained["density"]
        )
        l1_iv = l1_against_truth(
            iv_result["strike_grid"], iv_result["density"]
        )

        true_density = model.density(constrained_grid)
        true_density = true_density / np.trapezoid(true_density, constrained_grid)

        rows.append({
            "data_date": data_date,
            "expiration_date": expiration,
            "target_dte_days": int(target),
            "actual_dte": (expiration - data_date).days,
            "year": data_date.year,
            "forward": forward,
            "realized_value": realized,
            "true_forward": model.forward,

            "relative_crps_constrained": constrained_score["relative_crps"],
            "relative_crps_iv": iv_score["relative_crps"],
            "relative_crps_iv_recentred": (
                crps_from_cdf(recentred_grid, recentred_cdf, realized) / forward
            ),
            "relative_crps_lognormal": benchmark["relative_crps"],
            "relative_crps_truth": (
                crps_from_cdf(constrained_grid, true_cdf, realized) / forward
            ),

            "pit_constrained": constrained_score["pit"],
            "pit_iv": iv_score["pit"],
            "pit_lognormal": benchmark["pit"],

            "mean_over_forward_iv": iv_score["mean_over_forward"],
            "negative_mass_clipped": iv_result["negative_mass_clipped"],
            "n_smile_knots": iv_result["n_smile_knots"],
            "mass_below_traded": fit["mass_below_traded"],
            "mass_above_traded": fit["mass_above_traded"],

            "l1_error_constrained": l1_constrained,
            "l1_error_iv": l1_iv,
            "skewness_constrained": risk_neutral_skewness(
                constrained_grid, constrained["density"], forward
            ),
            "skewness_true": risk_neutral_skewness(constrained_grid, true_density, model.forward),
        })

        cached.append({
            "data_date": data_date,
            "expiration_date": expiration,
            "target_dte": int(target),
            "realized_value": float(realized),
            "forward": float(forward),
            "traded_range": fit["strike_bounds"],
            "strike_grid": constrained["strike_grid"],
            "fQ": constrained["density"],
        })

    return pd.DataFrame(rows), cached, pd.DataFrame(failures)


def run(max_chains: int = 240, months: int = 14, seed: int = 20260903,
        gamma: float = 2.5, figure_path: Path | None = None):
    """Run the whole demonstration and print the results."""
    market = SyntheticMarket(n_months=months, seed=seed, gamma=gamma)

    print("Generating synthetic chains ...")
    chains, settlement_values, truth = generate_sample(market)

    if max_chains and len(chains) > max_chains:
        # Thin by observation DATE, keeping every horizon observed on a kept
        # date. Thinning the flat list instead would leave about one row per
        # date, and the pooled HAC lag -- which is a lag in rows, and whose
        # whole point is that each date contributes several of them -- would
        # then be computed against a panel that no longer has the structure
        # the real one has.
        dates = sorted({key[0] for key, _ in chains})
        step = max(1, len(chains) // max_chains)
        keep = set(dates[::step])
        chains = [(key, chain) for key, chain in chains if key[0] in keep]

    settlement = {
        (int(r.expiration_year), int(r.expiration_month)): float(r.set_value)
        for r in settlement_values.itertuples(index=False)
    }

    print(f"Scoring {len(chains)} chains ...")
    scores, cached, failures = score_chains(
        chains, settlement, truth["models"]
    )

    if scores.empty:
        raise RuntimeError("No chains could be scored.")

    print()
    print("=" * 68)
    print("SAMPLE")
    print("=" * 68)
    size = effective_sample_size(scores)
    print(f"  {size['rows']} chains scored, {len(failures)} dropped")
    print(f"  {size['distinct_expirations']} distinct expirations "
          f"({size['rows_per_expiration']:.1f} rows each)")
    print("  The second number is what governs every standard error below.")

    print()
    print("=" * 68)
    print("SCORES  (mean relative CRPS; lower is better)")
    print("=" * 68)
    methods = {
        "relative_crps_truth": "True density (the floor)",
        "relative_crps_constrained": "Shape-constrained",
        "relative_crps_iv_recentred": "IV smoothing, recentred on F",
        "relative_crps_iv": "IV smoothing",
        "relative_crps_lognormal": "Lognormal at ATM vol",
    }
    for column, label in methods.items():
        skill = skill_score(scores[column], scores["relative_crps_lognormal"])
        print(f"  {label:32s} {scores[column].mean():.5f}   "
              f"skill vs lognormal {skill:+.4f}")

    print()
    print("  Location versus shape (issue 2.2):")
    displacement = scores["mean_over_forward_iv"] - 1.0
    gap_raw = (scores["relative_crps_iv"]
               - scores["relative_crps_constrained"]).mean()
    gap_recentred = (scores["relative_crps_iv_recentred"]
                     - scores["relative_crps_constrained"]).mean()
    print(f"    IV baseline mean / forward: {1 + displacement.mean():.5f} "
          f"(median {1 + displacement.median():.5f})")
    print(f"    IV minus constrained, raw:       {gap_raw:+.5f}")
    print(f"    IV minus constrained, recentred: {gap_recentred:+.5f}")
    if abs(gap_raw) > 0:
        print(f"    Share of the gap that is location: "
              f"{1 - gap_recentred / gap_raw:+.1%}")

    print()
    print("=" * 68)
    print("IS THE DIFFERENCE REAL?")
    print("=" * 68)
    lag = pooled_hac_lag(scores)
    dm = diebold_mariano(
        scores["relative_crps_iv"], scores["relative_crps_constrained"],
        lag=lag, names=("IV smoothing", "Shape-constrained"),
    )
    print(f"  Pooled Diebold-Mariano, HAC lag {lag} ROWS "
          f"(= {lag / (size['rows'] / scores['data_date'].nunique()):.0f} days "
          f"x {size['rows'] / scores['data_date'].nunique():.0f} rows/date)")
    print(f"    mean difference {dm['mean_difference']:+.6f}, "
          f"NW se {dm['newey_west_standard_error']:.6f}, p = {dm['p_value']:.4f}")
    print(f"    variance inflation over naive: {dm['variance_inflation']:.1f}x")

    bootstrap = block_bootstrap_mean(
        scores["relative_crps_iv"] - scores["relative_crps_constrained"],
        scores["expiration_date"].astype(str),
    )
    print(f"  Block bootstrap over {bootstrap['n_blocks']} expirations: "
          f"{bootstrap['mean']:+.6f} "
          f"[{bootstrap['lower']:+.6f}, {bootstrap['upper']:+.6f}]")

    print()
    print("=" * 68)
    print("CALIBRATION  (one observation per expiration -- issue 2.4)")
    print("=" * 68)
    for horizon in sorted(scores["target_dte_days"].unique()):
        panel = one_per_expiration(scores, horizon=int(horizon))
        result = pit_uniformity(panel["pit_constrained"])
        print(f"  {int(horizon):>3}-day: n = {result['n_observations']:>3}, "
              f"KS p = {result['ks_p_value']:.4f}, "
              f"mean PIT = {result['mean']:.3f}, "
              f"share below 10% = {result['share_below_10pct']:.3f}")

    print()
    print("=" * 68)
    print("RECOVERY ERROR AGAINST THE TRUE DENSITY")
    print("=" * 68)
    print(f"  L1 distance, shape-constrained: "
          f"{scores['l1_error_constrained'].mean():.4f} "
          f"(median {scores['l1_error_constrained'].median():.4f})")
    print(f"  L1 distance, IV smoothing:      "
          f"{scores['l1_error_iv'].mean():.4f} "
          f"(median {scores['l1_error_iv'].median():.4f})")
    print(f"  Risk-neutral skewness: fitted "
          f"{scores['skewness_constrained'].mean():+.3f} vs true "
          f"{scores['skewness_true'].mean():+.3f}")

    print()
    print("=" * 68)
    print("PART 3: RECOVERING A KNOWN GAMMA")
    print("=" * 68)
    gamma_grid = np.linspace(0.0, 5.0, 26)
    matrix = relative_crps_curve(gamma_grid, cached)
    blocks = [observation["expiration_date"] for observation in cached]
    interval = gamma_plausible_interval(
        gamma_grid, matrix, blocks, replicates=400
    )
    print(f"  True gamma used to draw the outcomes: {truth['gamma']:.2f}")
    print(f"  Estimated gamma-hat:                  {interval['gamma_hat']:.2f}")
    print(f"  Paired-difference 95% interval:       "
          f"[{interval['plausible_lower']:.2f}, {interval['plausible_upper']:.2f}]")
    print(f"  One-standard-error interval:          "
          f"[{interval['one_se_lower']:.2f}, {interval['one_se_upper']:.2f}]")
    print("  (Issue 2.5: this interval is built from the PAIRED difference")
    print("   L(gamma) - L(gamma-hat), not the standard error of the level.)")

    if figure_path is not None:
        _figure(scores, gamma_grid, interval, truth, Path(figure_path))
        print()
        print(f"Figure written to {figure_path}")

    return scores, cached, interval, truth


def _figure(scores, gamma_grid, interval, truth, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import PALETTE, use_project_style

    use_project_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    columns = [
        ("relative_crps_truth", "True density", PALETTE["reference"]),
        ("relative_crps_constrained", "Shape-constrained", PALETTE["constrained"]),
        ("relative_crps_iv", "IV smoothing", PALETTE["iv"]),
        ("relative_crps_lognormal", "Lognormal", PALETTE["benchmark"]),
    ]
    by_horizon = scores.groupby("target_dte_days")
    positions = np.arange(by_horizon.ngroups)
    width = 0.8 / len(columns)
    for index, (column, label, colour) in enumerate(columns):
        axes[0].bar(positions + index * width, by_horizon[column].mean(),
                    width=width, color=colour, label=label)
    axes[0].set_xticks(positions + width * (len(columns) - 1) / 2)
    axes[0].set_xticklabels([f"{h}d" for h in by_horizon.groups])
    axes[0].set_ylabel("Mean relative CRPS")
    axes[0].set_title("Scores by horizon")
    axes[0].legend(fontsize=8)

    axes[1].hist(scores["l1_error_constrained"], bins=30,
                 color=PALETTE["constrained"], alpha=0.85, label="Shape-constrained")
    axes[1].hist(scores["l1_error_iv"], bins=30,
                 color=PALETTE["iv"], alpha=0.6, label="IV smoothing")
    axes[1].set_xlabel(r"$L^1$ distance from the true density")
    axes[1].set_ylabel("Chains")
    axes[1].set_title("Recovery error against a known truth")
    axes[1].legend(fontsize=8)

    axes[2].plot(gamma_grid, interval["curve"], color=PALETTE["constrained"])
    axes[2].axvline(interval["gamma_hat"], color=PALETTE["iv"],
                    linestyle="--", label=fr"$\hat\gamma$ = {interval['gamma_hat']:.2f}")
    axes[2].axvline(truth["gamma"], color=PALETTE["violation"],
                    linestyle=":", label=fr"true $\gamma$ = {truth['gamma']:.2f}")
    axes[2].axvspan(interval["plausible_lower"], interval["plausible_upper"],
                    color=PALETTE["constrained"], alpha=0.12,
                    label="95% paired interval")
    axes[2].set_xlabel(r"$\gamma$")
    axes[2].set_ylabel("Mean relative CRPS")
    axes[2].set_title("Recovering a known risk aversion")
    axes[2].legend(fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chains", type=int, default=240)
    parser.add_argument("--months", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--gamma", type=float, default=2.5)
    parser.add_argument("--figure", default="figures/synthetic_demo.png")
    args = parser.parse_args(argv)

    run(
        max_chains=args.max_chains,
        months=args.months,
        seed=args.seed,
        gamma=args.gamma,
        figure_path=Path(args.figure) if args.figure else None,
    )


if __name__ == "__main__":
    _main()
