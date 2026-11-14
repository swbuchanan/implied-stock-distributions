"""Choosing the roughness penalty by cross-validation.

The penalty controls the only genuinely free choice in the shape-constrained
fit, and it was previously set by eye -- and set to three different values
in three different places, so that the example figures and the reported
scores did not come from the same model.

The criterion used here is out-of-sample pricing error. Every ``stride``-th
strike is held out, the curve is fitted to the rest, and the fitted price at
the held-out strikes is compared with the quoted one in units of the quote's
own half-spread. A penalty that is too small chases the bid-ask noise and
prices held-out strikes badly; one that is too large cannot follow the
smile. Nothing about the realised outcome enters, so the selection is
independent of the evaluation that follows.

Three things that make the comparison across folds honest
---------------------------------------------------------
**The knot vector is held fixed.** ``build_call_spline_program`` puts a knot
at every strike it is given, so letting the knots follow the training set
removes a fifth of the basis functions along with a fifth of the data. Each
fold would then be a different function space, and the CV curve would be
measuring the interaction of the penalty with the knot count rather than the
penalty. The full strike set is passed as ``knot_strikes`` and only the data
are held out.

**The penalty is rescaled to the fold.** Fengler's data term is a *sum* over
quotes, not an average, so fitting to 80% of them makes a given ``lambda``
act ``1 / 0.8 = 1.25`` times as strongly as it will in production. That is
inside one step of a factor-three grid, but it is free to correct: the
penalty handed to each fold is multiplied by the fraction of quotes retained.

**Selection uses the one-standard-error rule.** Taking the argmin of a noisy
CV curve with nothing attached picks a value the next sample would not
reproduce. The convention -- and the conservative choice here, since it errs
toward the smoother fit -- is the largest penalty whose median error is
within one standard error of the minimum.

A caveat the notebook should carry rather than this module: CV measures
*interpolation* at interior strikes, while CRPS is driven by the tails,
where no held-out strike exists. The selected penalty is therefore an
argument, not a proof, and the full results table across the penalty grid
is what turns the only free parameter into a robustness section.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CV_HOLDOUT_STRIDE,
    MIN_QUOTES_FOR_SPLINE,
    ROUGHNESS_PENALTY_DEFAULT,
    ROUGHNESS_PENALTY_GRID,
    SELECTED_PENALTY_PATH,
    read_selected_penalty,
)
from .errors import ChainError
from .fitting import (
    fit_shape_constrained_call_spline,
    pricing_error_scale,
    select_call_equivalent_quotes,
)
from .forwards import estimate_forward_and_rate

__all__ = [
    "cross_validate_chain",
    "select_roughness_penalty",
    "save_selected_penalty",
    "load_selected_penalty",
]


def cross_validate_chain(
    chain: pd.DataFrame,
    penalties=ROUGHNESS_PENALTY_GRID,
    stride: int = CV_HOLDOUT_STRIDE,
    estimates: dict | None = None,
    rescale_penalty: bool = True,
    **fit_kwargs,
) -> dict:
    """Held-out pricing error for one chain, for each candidate penalty.

    Returns a mapping from penalty to the root mean squared standardised
    error on the held-out strikes, averaged over the ``stride`` folds.
    ``nan`` marks a penalty for which no fold could be fitted.
    """
    if estimates is None:
        estimates, _ = estimate_forward_and_rate(chain)

    forward = estimates["forward"]
    discount_factor = estimates["discount_factor"]
    risk_free_rate = estimates["risk_free_rate"]

    quotes = select_call_equivalent_quotes(chain, forward, discount_factor)

    if len(quotes) < MIN_QUOTES_FOR_SPLINE + stride:
        raise ChainError(
            "Too few quotes to hold any out.", reason="insufficient_quotes"
        )

    # The knot vector is fixed from the full strike set, so every fold fits
    # the same function space and only the data change.
    all_strikes = quotes["strike_price"].to_numpy(dtype=float)

    position = np.arange(len(quotes))
    results = {}

    for penalty in penalties:
        fold_errors = []

        for fold in range(stride):
            held_out = position % stride == fold
            training = quotes.loc[~held_out]
            testing = quotes.loc[held_out]

            if len(training) < MIN_QUOTES_FOR_SPLINE or testing.empty:
                continue

            # Held-out strikes must sit inside the fitted range, otherwise
            # the comparison is about extrapolation rather than smoothing.
            inside = (
                testing["strike_price"].between(
                    training["strike_price"].min(),
                    training["strike_price"].max(),
                )
            )
            testing = testing.loc[inside]
            if testing.empty:
                continue

            # The data term is a sum, so a fold fitted to a fraction of the
            # quotes feels the same lambda more strongly. Scale it back.
            fold_penalty = penalty
            if rescale_penalty:
                fold_penalty = penalty * len(training) / len(quotes)

            try:
                fit = fit_shape_constrained_call_spline(
                    chain,
                    forward=forward,
                    risk_free_rate=risk_free_rate,
                    roughness_penalty=fold_penalty,
                    quotes=training,
                    knot_strikes=all_strikes,
                    **fit_kwargs,
                )
            except ChainError:
                continue

            x = testing["strike_price"].to_numpy(dtype=float) / forward
            predicted = discount_factor * forward * fit["spline"](x)
            actual = testing["call_equivalent_price"].to_numpy(dtype=float)

            scale = pricing_error_scale(
                testing["spread"].to_numpy(dtype=float),
                forward,
                discount_factor,
            ) * discount_factor * forward

            standardized = (predicted - actual) / scale
            standardized = standardized[np.isfinite(standardized)]

            if standardized.size:
                fold_errors.append(float(np.mean(standardized**2)))

        results[penalty] = (
            float(np.sqrt(np.mean(fold_errors))) if fold_errors else np.nan
        )

    return results


def select_roughness_penalty(
    chains,
    penalties=ROUGHNESS_PENALTY_GRID,
    stride: int = CV_HOLDOUT_STRIDE,
    progress_every: int | None = 50,
    one_standard_error: bool = True,
    **fit_kwargs,
):
    """Cross-validate the penalty over a sample of chains.

    Parameters
    ----------
    chains : iterable
        Yields either DataFrames or ``(key, DataFrame)`` pairs, so the
        catalogue iterator can be passed straight in.
    one_standard_error : bool
        Select the largest penalty whose median held-out error is within
        one standard error of the smallest, rather than the bare argmin.
        This is the standard convention and it errs toward the smoother
        fit, which is the right direction to err when the criterion
        measures interpolation and the score is driven by the tails.

    Returns
    -------
    tuple[float, pandas.DataFrame, pandas.DataFrame]
        The selected penalty, a summary of the CV curve with the median,
        mean, standard error and count behind each point, and the raw
        per-chain table so the notebook can report the full distribution
        rather than only its centre.
    """
    rows = []

    for index, item in enumerate(chains, start=1):
        chain = item[1] if isinstance(item, tuple) else item

        if progress_every and index % progress_every == 0:
            print(f"Cross-validated {index} chains")

        try:
            scores = cross_validate_chain(
                chain,
                penalties=penalties,
                stride=stride,
                **fit_kwargs,
            )
        except (ChainError, ValueError):
            continue

        rows.append(scores)

    if not rows:
        raise ValueError("No chains could be cross-validated.")

    table = pd.DataFrame(rows)

    counts = table.notna().sum(axis=0).to_numpy()
    standard_errors = (
        table.std(axis=0, ddof=1).to_numpy() / np.sqrt(np.maximum(counts, 1))
    )

    summary = pd.DataFrame({
        "roughness_penalty": table.columns.astype(float),
        "median_held_out_error": table.median(axis=0).to_numpy(),
        "mean_held_out_error": table.mean(axis=0).to_numpy(),
        "standard_error": standard_errors,
        "n_chains": counts,
    }).sort_values("roughness_penalty").reset_index(drop=True)

    best_index = int(summary["median_held_out_error"].idxmin())
    minimum = float(summary.loc[best_index, "median_held_out_error"])

    if one_standard_error:
        threshold = minimum + float(summary.loc[best_index, "standard_error"])
        within = summary.loc[summary["median_held_out_error"] <= threshold]
        selected = float(within["roughness_penalty"].max())
    else:
        selected = float(summary.loc[best_index, "roughness_penalty"])

    summary["selected"] = np.isclose(summary["roughness_penalty"], selected)
    summary["within_one_se"] = summary["median_held_out_error"] <= (
        minimum + float(summary.loc[best_index, "standard_error"])
    )

    return selected, summary, table


# --------------------------------------------------------------------------
# Persisting the selection, so nothing has to be pasted anywhere
# --------------------------------------------------------------------------

def save_selected_penalty(
    penalty: float,
    summary: pd.DataFrame | None = None,
    path: Path = SELECTED_PENALTY_PATH,
) -> Path:
    """Write the cross-validated penalty where ``config`` will read it.

    The notebook used to print "paste into config.ROUGHNESS_PENALTY" next
    to a value marked PROVISIONAL, which leaves a reader to conclude the
    published results were produced with a placeholder. Writing the number
    to a small JSON file that ``config`` loads on import removes the manual
    step and the ambiguity: after the cross-validation cell has run once,
    ``config.ROUGHNESS_PENALTY`` *is* the cross-validated value, and
    ``config.ROUGHNESS_PENALTY_IS_DEFAULT`` says so.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "roughness_penalty": float(penalty),
        "selected_at": pd.Timestamp.utcnow().isoformat(),
    }

    if summary is not None:
        payload["n_chains"] = int(summary["n_chains"].max())
        payload["curve"] = {
            f"{row.roughness_penalty:g}": float(row.median_held_out_error)
            for row in summary.itertuples(index=False)
        }

    path.write_text(json.dumps(payload, indent=2))
    return path


def load_selected_penalty(
    path: Path = SELECTED_PENALTY_PATH,
    default: float = ROUGHNESS_PENALTY_DEFAULT,
) -> tuple[float, bool]:
    """Read the saved penalty, returning ``(value, is_default)``.

    Delegates to ``config`` so that there is exactly one implementation of
    the read, and ``config.ROUGHNESS_PENALTY`` cannot disagree with what
    this returns.
    """
    return read_selected_penalty(path, default)
