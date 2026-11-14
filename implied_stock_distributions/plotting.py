"""Figures for the analysis notebook.

Kept out of the notebook so that the notebook reads as an argument rather
than as a program. Every function takes data already computed and returns
``(fig, ax)``, so nothing here fits, scores or loads anything.

Colour is assigned by identity and in a fixed order, never cycled: the
shape-constrained method is always slot one and the implied-volatility
baseline always slot two, in every figure. The four categorical hues were
checked for colour-vision separation (all pairs, light surface) rather than
chosen by eye, and red is reserved for the one thing it means here -- a
region where the fitted density is negative, which is a no-arbitrage
violation rather than another series.

Figures belonging to the external validations live in ``validation.py``,
next to the code that builds their inputs.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PALETTE",
    "use_project_style",
    "plot_call_chain",
    "plot_parity_fit",
    "plot_naive_price_smoothing",
    "plot_method_comparison",
    "plot_roughness_cv",
    "plot_pit_histogram",
    "plot_score_by_group",
    "plot_rate_validation",
    "plot_gamma_curve",
    "plot_gamma_interval",
    "plot_extrapolated_mass",
]

# Fixed categorical order. Slot one is the constrained method throughout.
PALETTE = {
    "constrained": "#3373C4",
    "iv": "#D96A0F",
    "benchmark": "#1B9E77",
    "physical": "#8A5A00",
    "reference": "#6B6B6B",
    "violation": "#C0392B",
    "grid": "#DDDDDD",
    "ink": "#333333",
    "muted": "#767676",
}


def use_project_style():
    """Recessive axes, light grid, consistent type sizes."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.constrained_layout.use": True,
    })


def _title(info, extra=""):
    import pandas as pd

    data_date = pd.Timestamp(info["data_date"]).date()
    expiry = pd.Timestamp(info["expiration_date"]).date()
    actual = (pd.Timestamp(info["expiration_date"])
              - pd.Timestamp(info["data_date"])).days
    head = (
        f"{data_date}, expiring {expiry} "
        f"({actual} days; {int(info['target_dte_days'])}-day target)"
    )
    return f"{extra}\n{head}" if extra else head


# --------------------------------------------------------------------------
# Part 1: the data and the parity fit
# --------------------------------------------------------------------------

def plot_call_chain(info, chain, ax=None):
    """Call mid-prices against strike for one chain."""
    calls = (
        chain.loc[chain["put_call"].eq("call")]
        .sort_values("strike_price")
    )
    if calls.empty:
        raise ValueError("The selected chain contains no calls.")

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))

    ax.plot(
        calls["strike_price"], calls["mid_price"],
        color=PALETTE["constrained"], label="Call midpoint",
    )
    ax.axvline(
        calls["underlying_price"].median(),
        color=PALETTE["reference"], linestyle="--", linewidth=1,
        label="SPX spot",
    )

    ax.set_xlabel("Strike price")
    ax.set_ylabel("Mid-price")
    ax.set_title(_title(info, "SPX call prices"))
    ax.legend()
    return fig, ax


def plot_parity_fit(info, estimates, pairs, ax=None, inset=True):
    """Call minus put against strike, with the fitted parity line.

    The optional inset zooms on the intercept, which is the quantity the
    forward is read from and which the full-range view compresses to a
    point.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 4.2))

    ax.scatter(
        pairs["strike_price"], pairs["call_minus_put"],
        s=16, color=PALETTE["constrained"], alpha=0.75, label="Matched pairs",
    )
    ax.plot(
        pairs["strike_price"], pairs["fitted_call_minus_put"],
        color=PALETTE["iv"], label="Parity fit",
    )
    ax.axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=0.8)
    ax.axvline(
        estimates["forward"], color=PALETTE["benchmark"], linestyle="--",
        linewidth=1.2, label=f"Forward = {estimates['forward']:,.1f}",
    )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Call price − put price")
    ax.set_title(_title(
        info,
        f"Put-call parity fit  ·  implied rate {estimates['risk_free_rate']:.2%}"
        f"  ·  R² {estimates['r_squared']:.6f}"
        f"  ·  {estimates.get('fraction_outside_spreads', float('nan')):.1%}"
        " outside spreads",
    ))
    ax.legend()

    if inset:
        # Zoom on the crossing point, where the forward is identified.
        axins = ax.inset_axes([0.62, 0.58, 0.36, 0.38])
        centre = estimates["forward"]
        span = 0.02 * centre
        near = pairs.loc[
            (pairs["strike_price"] > centre - span)
            & (pairs["strike_price"] < centre + span)
        ]
        if len(near) >= 2:
            axins.scatter(near["strike_price"], near["call_minus_put"],
                          s=12, color=PALETTE["constrained"], alpha=0.85)
            axins.plot(near["strike_price"], near["fitted_call_minus_put"],
                       color=PALETTE["iv"], linewidth=1.4)
            axins.axhline(0, color=PALETTE["reference"],
                          linestyle="--", linewidth=0.8)
            axins.axvline(centre, color=PALETTE["benchmark"],
                          linestyle="--", linewidth=1.0)
            axins.set_title("zoom on the crossing", fontsize=8)
            axins.tick_params(labelsize=7)
            axins.grid(alpha=0.4)

    return fig, ax


# --------------------------------------------------------------------------
# Part 1: why constraints are needed
# --------------------------------------------------------------------------

def plot_naive_price_smoothing(info, strikes, prices, grid, curve, first, second):
    """Three panels showing an unconstrained price-space fit failing.

    The bottom panel shades the regions where the fitted second derivative
    is negative, which is where the implied density would be negative.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.5), sharex=True)

    axes[0].scatter(strikes, prices, s=12, alpha=0.6,
                    color=PALETTE["reference"], label="Observed mid-prices")
    axes[0].plot(grid, curve, color=PALETTE["iv"], label="Price-space spline")
    axes[0].set_ylabel("C(K)")
    axes[0].legend()
    axes[0].set_title(_title(info, "Unconstrained smoothing in price space"))

    axes[1].plot(grid, first, color=PALETTE["constrained"])
    axes[1].axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("C′(K)")

    axes[2].plot(grid, second, color=PALETTE["constrained"])
    axes[2].axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=0.8)
    axes[2].fill_between(
        grid, second, 0, where=second < 0,
        color=PALETTE["violation"], alpha=0.30,
        label="Negative density (convexity violated)",
    )
    axes[2].set_ylabel("C″(K)")
    axes[2].set_xlabel("Strike price")
    axes[2].legend()

    return fig, axes


def plot_method_comparison(
    info,
    quotes,
    iv_result,
    constrained_distribution,
    forward,
):
    """Prices, the smile and both densities for one chain."""
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)

    axes[0].scatter(
        quotes["strike_price"], quotes["call_equivalent_price"],
        s=12, alpha=0.6, color=PALETTE["reference"],
        label="Observed (OTM, call-equivalent)",
    )
    axes[0].plot(
        constrained_distribution["strike_grid"],
        constrained_distribution["call_prices"],
        color=PALETTE["constrained"], label="Shape-constrained fit",
    )
    axes[0].plot(
        iv_result["strike_grid"], iv_result["price_smooth"],
        color=PALETTE["iv"], linestyle="--", label="IV-smoothing fit",
    )
    axes[0].set_ylabel("Call price")
    axes[0].set_title(_title(info, "Two fits to the same quotes"))
    axes[0].legend()

    axes[1].scatter(
        iv_result["observed_strikes"], 100 * iv_result["observed_iv"],
        s=12, alpha=0.6, color=PALETTE["reference"], label="Observed IV",
    )
    axes[1].plot(
        iv_result["strike_grid"],
        100 * iv_result["iv_of_strike"](iv_result["strike_grid"]),
        color=PALETTE["iv"],
        label=(
            "Smoothed IV, flat extrapolation "
            f"({iv_result.get('n_smile_knots', '?')} knots)"
        ),
    )
    axes[1].set_ylabel("Implied volatility (%)")
    axes[1].legend()

    axes[2].plot(
        constrained_distribution["strike_grid"],
        constrained_distribution["density"],
        color=PALETTE["constrained"], label="Shape-constrained",
    )
    axes[2].plot(
        iv_result["strike_grid"], iv_result["density"],
        color=PALETTE["iv"], linestyle="--", label="IV smoothing",
    )
    axes[2].set_ylabel("Density")
    axes[2].set_xlabel("SPX level at expiration")
    axes[2].legend()

    for ax in axes:
        ax.axvline(forward, color=PALETTE["reference"],
                   linestyle=":", linewidth=1)

    # Focus on the region where anything is visible.
    lower = constrained_distribution["strike_bounds"][0] * 0.85
    upper = constrained_distribution["strike_bounds"][1] * 1.05
    axes[0].set_xlim(lower, upper)

    return fig, axes


def plot_roughness_cv(summary, chosen, ax=None):
    """Held-out pricing error against the roughness penalty.

    The one-standard-error band is drawn, because the argmin of a noisy CV
    curve is not by itself a defensible choice and the band is what makes
    the selected value defensible.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4))

    ax.plot(
        summary["roughness_penalty"], summary["median_held_out_error"],
        marker="o", markersize=5, color=PALETTE["constrained"],
        label="Median held-out error",
    )

    if "standard_error" in summary.columns:
        best = summary["median_held_out_error"].min()
        threshold = best + float(
            summary.loc[summary["median_held_out_error"].idxmin(), "standard_error"]
        )
        ax.axhline(
            threshold, color=PALETTE["muted"], linestyle=":", linewidth=1.0,
            label="One standard error above the minimum",
        )
        ax.fill_between(
            summary["roughness_penalty"],
            summary["median_held_out_error"] - summary["standard_error"],
            summary["median_held_out_error"] + summary["standard_error"],
            color=PALETTE["constrained"], alpha=0.15,
        )

    ax.axvline(
        chosen, color=PALETTE["reference"], linestyle="--", linewidth=1.2,
        label=f"Selected λ = {chosen:g}",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Roughness penalty λ")
    ax.set_ylabel("Held-out error (half-spreads)")
    ax.set_title("Cross-validated choice of the roughness penalty")
    ax.legend()
    return fig, ax


def plot_extrapolated_mass(scores, ax=None):
    """Distribution of fitted probability mass outside the traded strikes."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4))

    ax.hist(
        100 * scores["mass_below_traded"], bins=60,
        color=PALETTE["constrained"], alpha=0.85, label="Below lowest strike",
    )
    ax.hist(
        100 * scores["mass_above_traded"], bins=60,
        color=PALETTE["iv"], alpha=0.65, label="Above highest strike",
    )
    ax.set_xlabel("Fitted probability mass outside the traded range (%)")
    ax.set_ylabel("Chains")
    ax.set_title(
        "How much of each distribution is extrapolation rather than price"
    )
    ax.legend()
    return fig, ax


# --------------------------------------------------------------------------
# Part 2: calibration and scores
# --------------------------------------------------------------------------

def plot_pit_histogram(series, n_bins=10, ax=None):
    """Probability integral transforms against the uniform they should be.

    ``series`` maps a label to an array of PIT values. A calibrated forecast
    fills every bin equally; the flat line is what to compare against.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4))

    colors = [PALETTE["constrained"], PALETTE["physical"], PALETTE["benchmark"]]
    edges = np.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins / (len(series) + 0.6)

    for index, (label, values) in enumerate(series.items()):
        values = np.asarray(values, dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        share = counts / counts.sum()
        ax.bar(
            edges[:-1] + (index + 0.3) * width, share, width=width,
            align="edge", color=colors[index % len(colors)], label=label,
        )

    ax.axhline(
        1.0 / n_bins, color=PALETTE["reference"], linestyle="--",
        linewidth=1.2, label="Calibrated (uniform)",
    )
    ax.set_xlabel("Probability integral transform  F(realised outcome)")
    ax.set_ylabel("Share of forecasts")
    ax.set_title("Calibration of the forecast distributions")
    ax.legend()
    return fig, ax


def plot_score_by_group(
    table,
    group_column,
    value_columns,
    labels=None,
    ax=None,
    ylabel="Mean relative CRPS",
):
    """Mean score by year or horizon, one bar group per category."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 4))

    grouped = table.groupby(group_column)[list(value_columns)].mean()
    positions = np.arange(len(grouped))
    width = 0.8 / len(value_columns)

    colors = [PALETTE["constrained"], PALETTE["iv"],
              PALETTE["benchmark"], PALETTE["physical"]]

    for index, column in enumerate(value_columns):
        ax.bar(
            positions + index * width, grouped[column], width=width,
            color=colors[index % len(colors)],
            label=(labels or {}).get(column, column),
        )

    ax.set_xticks(positions + width * (len(value_columns) - 1) / 2)
    ax.set_xticklabels(grouped.index, rotation=0)
    ax.set_xlabel(group_column.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig, ax


def plot_rate_validation(diagnostics, reference=None, ax=None):
    """Parity-implied rates over time, optionally against a market series.

    ``reference`` is a date-indexed Series -- see
    ``validation.fetch_short_rates``. Without one this is a plausibility
    check, not a validation, and the title says so.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9, 4))

    daily = (
        diagnostics.groupby("data_date")["risk_free_rate"].median().sort_index()
    )

    ax.plot(
        daily.index, 100 * daily.to_numpy(),
        color=PALETTE["constrained"], linewidth=1.4,
        label="Implied by put-call parity",
    )

    if reference is not None:
        ax.plot(
            reference.index, 100 * reference.to_numpy(),
            color=PALETTE["iv"], linewidth=1.4, linestyle="--",
            label="Treasury bill (FRED)",
        )
        title = "Rates recovered from option prices, against the actual bill rate"
    else:
        title = "Rates recovered from option prices alone (no reference series)"

    ax.axhline(0, color=PALETTE["reference"], linestyle=":", linewidth=0.8)
    ax.set_xlabel("Observation date")
    ax.set_ylabel("Continuously compounded rate (%)")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_gamma_curve(gamma_grid, curves, chosen=None, ax=None):
    """Mean relative CRPS as a function of the risk-aversion parameter."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.5, 4.5))

    colors = [PALETTE["constrained"], PALETTE["iv"],
              PALETTE["benchmark"], PALETTE["physical"]]

    for index, (label, values) in enumerate(curves.items()):
        ax.plot(gamma_grid, values, color=colors[index % len(colors)], label=label)

    if chosen is not None:
        ax.axvline(
            chosen, color=PALETTE["reference"], linestyle="--", linewidth=1.2,
            label=fr"$\hat\gamma$ = {chosen:.2f}",
        )

    ax.set_xlabel(r"Coefficient of relative risk aversion $\gamma$")
    ax.set_ylabel("Mean relative CRPS")
    ax.set_title(r"Score of the tilted density against $\gamma$")
    ax.legend()
    return fig, ax


def plot_gamma_interval(gamma_grid, interval, ax=None):
    """The paired-difference curve that identifies gamma, with its band.

    This is the figure that belongs next to a gamma estimate, rather than
    the raw score curve: the level of the score is dominated by how far
    each outcome happened to land from its forward, and that variation
    cancels in the paired difference. Plotting the level and quoting its
    standard error -- which is what an earlier version did -- makes the
    objective look far flatter than the data actually say it is.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.5, 4.5))

    difference = np.asarray(interval["mean_difference"], dtype=float)
    lower = np.asarray(interval["lower"], dtype=float)
    upper = np.asarray(interval["upper"], dtype=float)

    ax.plot(gamma_grid, difference, color=PALETTE["constrained"],
            label=r"$L(\gamma) - L(\hat\gamma)$")
    ax.fill_between(gamma_grid, lower, upper, color=PALETTE["constrained"],
                    alpha=0.18,
                    label=f"{interval['confidence']:.0%} block bootstrap")
    ax.axhline(0, color=PALETTE["reference"], linestyle="--", linewidth=1.0)
    ax.axvline(interval["gamma_hat"], color=PALETTE["iv"], linestyle="--",
               linewidth=1.2, label=fr"$\hat\gamma$ = {interval['gamma_hat']:.2f}")

    if np.isfinite(interval.get("plausible_lower", np.nan)):
        ax.axvspan(interval["plausible_lower"], interval["plausible_upper"],
                   color=PALETTE["benchmark"], alpha=0.10,
                   label="Not rejected")

    ax.set_xlabel(r"Coefficient of relative risk aversion $\gamma$")
    ax.set_ylabel("Paired score difference")
    ax.set_title(r"What the data actually identify about $\gamma$")
    ax.legend(fontsize=8)
    return fig, ax
