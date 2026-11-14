"""Synthetic SPX-like option chains with a known true density.

The raw vendor data are not redistributable, which is fine, but the
consequence is that a reader sees code and prose and has no way to check
either. This module removes that: it generates option chains from a model
whose density, CDF and option prices are all available in closed form, in
the same shape the processing notebook produces, so that

    python -m implied_stock_distributions.synthetic

writes a complete synthetic dataset that ``data_access`` loads unchanged
and the analysis notebook runs against end to end in about a minute.

Three things follow from having a known truth, and they are worth more than
the reproducibility alone:

* the test suite gets a fixture with an analytic answer, so the recovery of
  a density can be *measured* rather than eyeballed;
* the roughness penalty can be chosen by minimising error against the true
  density rather than against held-out mid-quotes, which is the only way to
  see what the penalty does in the extrapolated tails where no held-out
  strike exists; and
* the realised outcomes are drawn from a physical measure that is a CRRA
  tilt, at a *known* gamma, of the risk-neutral measure of one chain per
  expiration. Only that chain's density is an exact tilt of the outcome
  distribution, so a gamma estimate or PIT check pooled over every chain
  is confounded by forecast misspecification.

The model
---------
Under ``Q``, ``S_T`` is a two-component lognormal mixture,

    f_Q(s) = sum_j w_j LogNormal(s; m_j, v_j),

with the component means constrained so that ``E_Q[S_T] = F`` exactly. A
mixture prices European options in closed form as the same mixture of Black
prices, one per component at its own forward, which is what makes the
quotes exact rather than simulated. Two components with a heavier, wider
down-component reproduce the feature that matters here: a left-skewed
risk-neutral density with a fatter left tail than a lognormal, which is
what SPX smiles imply and what drives the score.

Tilting by ``s^gamma`` maps the mixture to another mixture -- component
``j`` becomes ``LogNormal(m_j + gamma v_j, v_j)`` and its weight is
multiplied by ``exp(gamma m_j + gamma^2 v_j / 2)`` before renormalising --
so the physical measure the outcomes are drawn from is exact too.

Market microstructure
---------------------
Quotes are not the model prices. Strikes are spaced five points near the
money and twenty-five in the wings, as SPX is; spreads widen with distance
from the money and are floored at a tick; prices are rounded to the tick;
and a configurable share of quotes are stale, carrying a price from an
earlier index level. These are the features that make the cleaning code in
``01_processing.ipynb`` and the quote-weighting in the fit do something,
rather than passing trivially.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    AM_SETTLEMENT_DAY_ADJUSTMENT,
    DATA_DIR,
    DAY_COUNT_DAYS_PER_YEAR,
    TARGET_DTES,
)
from .data_access import catalog_path

__all__ = [
    "MixtureModel",
    "SyntheticMarket",
    "build_chain",
    "generate_sample",
    "write_synthetic_dataset",
]

TICK = 0.05


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MixtureModel:
    """A two-component lognormal mixture for ``S_T`` with mean ``forward``.

    Parameters are given as component volatilities and a shift of the
    down-component, and the log-means are then solved so that the mixture
    mean is exactly the forward. That keeps the generated chains consistent
    with put-call parity by construction, which is what the pipeline's
    forward estimator is supposed to recover.
    """

    forward: float
    time_to_expiry: float
    weights: tuple[float, ...] = (0.80, 0.20)
    volatilities: tuple[float, ...] = (0.15, 0.28)
    # Multiplicative displacement of each component's own forward. The
    # weighted average is renormalised to one, so these set the skew.
    displacements: tuple[float, ...] = (1.02, 0.92)

    def __post_init__(self):
        n = len(self.weights)
        if not (len(self.volatilities) == len(self.displacements) == n):
            raise ValueError("weights, volatilities and displacements must align.")
        if abs(sum(self.weights) - 1.0) > 1e-12:
            raise ValueError("weights must sum to one.")
        if self.forward <= 0 or self.time_to_expiry <= 0:
            raise ValueError("forward and time_to_expiry must be positive.")

    @property
    def component_forwards(self) -> np.ndarray:
        """Component means, scaled so the mixture mean is the forward."""
        weights = np.asarray(self.weights, dtype=float)
        raw = np.asarray(self.displacements, dtype=float)
        return self.forward * raw / float(weights @ raw)

    @property
    def log_scales(self) -> np.ndarray:
        """``s_j = sigma_j sqrt(tau)``, the log standard deviations."""
        return np.asarray(self.volatilities, dtype=float) * np.sqrt(
            self.time_to_expiry
        )

    @property
    def log_means(self) -> np.ndarray:
        """``m_j``, so that ``exp(m_j + s_j^2 / 2)`` is the component mean."""
        return np.log(self.component_forwards) - 0.5 * self.log_scales**2

    # -- closed forms ------------------------------------------------------

    def density(self, levels) -> np.ndarray:
        """True risk-neutral density on a grid."""
        levels = np.asarray(levels, dtype=float)
        out = np.zeros_like(levels)
        positive = levels > 0

        for weight, mean_log, scale in zip(
            self.weights, self.log_means, self.log_scales, strict=True
        ):
            out[positive] += weight * stats.lognorm.pdf(
                levels[positive], s=scale, scale=np.exp(mean_log)
            )
        return out

    def cdf(self, levels) -> np.ndarray:
        """True risk-neutral CDF on a grid."""
        levels = np.asarray(levels, dtype=float)
        out = np.zeros_like(levels)
        positive = levels > 0

        for weight, mean_log, scale in zip(
            self.weights, self.log_means, self.log_scales, strict=True
        ):
            out[positive] += weight * stats.lognorm.cdf(
                levels[positive], s=scale, scale=np.exp(mean_log)
            )
        return out

    def call_price(self, strike, risk_free_rate: float) -> np.ndarray:
        """Exact European call price: the mixture of Black prices."""
        from .pricing import black_call_price

        strike = np.atleast_1d(np.asarray(strike, dtype=float))
        total = np.zeros_like(strike)

        for weight, component_forward, scale in zip(
            self.weights, self.component_forwards, self.log_scales, strict=True
        ):
            total += weight * black_call_price(
                component_forward,
                strike,
                scale / np.sqrt(self.time_to_expiry),
                self.time_to_expiry,
                risk_free_rate,
            )
        return total

    def put_price(self, strike, risk_free_rate: float) -> np.ndarray:
        """Exact European put price, by parity from the call."""
        strike = np.atleast_1d(np.asarray(strike, dtype=float))
        discount = np.exp(-risk_free_rate * self.time_to_expiry)
        return self.call_price(strike, risk_free_rate) - discount * (
            self.forward - strike
        )

    def tilted(self, gamma: float) -> MixtureModel:
        """The CRRA-tilted mixture: still a mixture, in closed form.

        ``s^gamma LogNormal(m, v)`` is ``LogNormal(m + gamma v, v)`` up to a
        factor ``exp(gamma m + gamma^2 v / 2)``, so the physical measure is
        a mixture with shifted components and reweighted components.
        """
        log_means = self.log_means
        scales = self.log_scales
        variances = scales**2

        log_weights = (
            np.log(np.asarray(self.weights, dtype=float))
            + gamma * log_means
            + 0.5 * gamma**2 * variances
        )
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights = weights / weights.sum()

        new_log_means = log_means + gamma * variances
        new_forwards = np.exp(new_log_means + 0.5 * variances)
        mean = float(weights @ new_forwards)

        return MixtureModel(
            forward=mean,
            time_to_expiry=self.time_to_expiry,
            weights=tuple(weights),
            volatilities=tuple(self.volatilities),
            displacements=tuple(new_forwards / mean),
        )

    def sample(self, size: int, generator: np.random.Generator) -> np.ndarray:
        """Draw outcomes from the mixture."""
        weights = np.asarray(self.weights, dtype=float)
        which = generator.choice(len(weights), size=size, p=weights)
        normals = generator.standard_normal(size)
        return np.exp(self.log_means[which] + self.log_scales[which] * normals)


# --------------------------------------------------------------------------
# The market
# --------------------------------------------------------------------------

@dataclass
class SyntheticMarket:
    """Settings for a synthetic sample: index path, rates and quote noise."""

    start: str = "2018-01-02"
    n_months: int = 36
    initial_level: float = 2700.0
    drift: float = 0.06
    realised_volatility: float = 0.16
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.019
    # Risk aversion of the representative investor generating the outcomes.
    # Part 3's estimator should recover roughly this.
    gamma: float = 2.5
    # Base volatility level and how much it moves month to month.
    base_volatility: float = 0.15
    volatility_of_volatility: float = 0.25
    stale_quote_share: float = 0.04
    wide_quote_share: float = 0.02
    seed: int = 20260903
    target_dtes: tuple[int, ...] = field(default_factory=lambda: tuple(TARGET_DTES))


def _third_friday(year: int, month: int) -> pd.Timestamp:
    """Standard monthly SPX expiration."""
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.dayofweek) % 7
    return first + pd.Timedelta(days=offset + 14)


def _strike_ladder(forward: float) -> np.ndarray:
    """SPX-like strikes: five points near the money, twenty-five in the wings.

    Real chains are not a uniform grid, and a uniform grid would hide the
    thing the roughness penalty has to cope with -- a basis whose knots
    bunch in the middle and spread out in the tails.
    """
    near_low, near_high = 0.90 * forward, 1.10 * forward
    far_low, far_high = 0.50 * forward, 1.30 * forward

    near = np.arange(np.ceil(near_low / 5) * 5, near_high, 5.0)
    lower = np.arange(np.ceil(far_low / 25) * 25, near_low, 25.0)
    upper = np.arange(np.ceil(near_high / 25) * 25, far_high, 25.0)

    return np.unique(np.concatenate([lower, near, upper]))


def _quote_spread(price, strike, forward, generator):
    """Half-spread model: wider in the wings and on cheap options.

    SPX spreads scale roughly with the option's own value near the money
    and hit a floor of a few ticks in the wings, where the quote is a fixed
    number of ticks wide regardless of price.
    """
    price = np.asarray(price, dtype=float)
    moneyness = np.abs(np.log(np.asarray(strike, dtype=float) / forward))

    proportional = 0.02 * price * (1.0 + 4.0 * moneyness)
    floor = TICK * (2.0 + 30.0 * moneyness)

    spread = np.maximum(proportional, floor)
    noise = np.exp(generator.normal(0.0, 0.15, size=spread.shape))
    return spread * noise


def _round_to_tick(values):
    return np.round(np.asarray(values, dtype=float) / TICK) * TICK


def build_chain(
    data_date: pd.Timestamp,
    expiration_date: pd.Timestamp,
    target_dte: int,
    spot: float,
    volatility: float,
    market: SyntheticMarket,
    generator: np.random.Generator,
) -> tuple[pd.DataFrame, MixtureModel]:
    """One synthetic chain, plus the model that generated it."""
    data_date = pd.Timestamp(data_date)
    expiration_date = pd.Timestamp(expiration_date)

    dte = int((expiration_date - data_date).days)
    time_to_expiry = (dte - AM_SETTLEMENT_DAY_ADJUSTMENT) / DAY_COUNT_DAYS_PER_YEAR

    if time_to_expiry <= 0:
        raise ValueError("Expiration must be after the data date.")

    rate = market.risk_free_rate
    forward = spot * np.exp((rate - market.dividend_yield) * time_to_expiry)

    model = MixtureModel(
        forward=float(forward),
        time_to_expiry=float(time_to_expiry),
        volatilities=(volatility, volatility * 1.85),
    )

    strikes = _strike_ladder(forward)

    call_mid = model.call_price(strikes, rate)
    put_mid = model.put_price(strikes, rate)

    rows = []
    for label, mid in (("call", call_mid), ("put", put_mid)):
        mid = np.asarray(mid, dtype=float)

        # Stale quotes: a price left over from a different index level.
        stale = generator.random(mid.shape) < market.stale_quote_share
        displaced = model.call_price(
            strikes * np.exp(generator.normal(0.0, 0.004, size=mid.shape)), rate
        ) if label == "call" else model.put_price(
            strikes * np.exp(generator.normal(0.0, 0.004, size=mid.shape)), rate
        )
        mid = np.where(stale, displaced, mid)

        spread = _quote_spread(mid, strikes, forward, generator)
        # A minority of quotes are much wider than the model says, which is
        # what the spread weighting in the parity fit is there to handle.
        wide = generator.random(mid.shape) < market.wide_quote_share
        spread = np.where(wide, spread * 6.0, spread)

        bid = _round_to_tick(np.maximum(mid - spread / 2.0, 0.0))
        ask = _round_to_tick(np.maximum(mid + spread / 2.0, bid + TICK))

        rows.append(pd.DataFrame({
            "strike_price": strikes,
            "put_call": label,
            "bid_price": bid,
            "ask_price": ask,
        }))

    chain = pd.concat(rows, ignore_index=True)

    # Deep out-of-the-money options that would quote below a tick are not
    # quoted two-sided in a real chain.
    chain["quote_is_valid"] = (chain["ask_price"] > 0) & (
        chain["ask_price"] >= chain["bid_price"]
    )
    chain["quote_is_two_sided"] = chain["quote_is_valid"] & (chain["bid_price"] > 0)
    chain["mid_price"] = (
        (chain["bid_price"] + chain["ask_price"]) / 2
    ).where(chain["quote_is_valid"])

    chain["symbol"] = "SPX"
    chain["data_date"] = data_date
    chain["expiration_date"] = expiration_date
    chain["settlement_date"] = expiration_date
    chain["target_dte_days"] = np.int16(target_dte)
    chain["dte_calendar_days"] = np.int32(dte)
    chain["time_to_expiry_years"] = time_to_expiry
    chain["underlying_price"] = spot
    chain["is_standard_monthly"] = True
    chain["bid_size"] = 50
    chain["ask_size"] = 50
    chain["last_price"] = chain["mid_price"]
    chain["volume"] = 100
    chain["open_interest"] = 1000
    chain["source_file"] = "synthetic"
    chain["option_key"] = (
        data_date.strftime("%Y%m%d")
        + "_" + expiration_date.strftime("%Y%m%d")
        + "_" + chain["put_call"].str[0].str.upper()
        + "_" + chain["strike_price"].astype(int).astype(str)
    )

    return chain.sort_values(["put_call", "strike_price"], ignore_index=True), model


def generate_sample(market: SyntheticMarket | None = None):
    """Generate a whole synthetic sample: chains, outcomes and the truth.

    Returns
    -------
    tuple[list[tuple[tuple, pandas.DataFrame]], pandas.DataFrame, dict]
        ``(chains, settlement_values, truth)`` where ``chains`` is what
        ``data_access.iter_chains`` yields, ``settlement_values`` is what
        ``load_settlement_values`` returns, and ``truth`` carries the
        generating model per chain plus the gamma the outcomes were drawn
        with.
    """
    market = market or SyntheticMarket()
    generator = np.random.default_rng(market.seed)

    start = pd.Timestamp(market.start)
    expirations = [
        _third_friday(
            (start + pd.DateOffset(months=offset)).year,
            (start + pd.DateOffset(months=offset)).month,
        )
        for offset in range(market.n_months + 4)
    ]
    expirations = sorted({e for e in expirations if e > start})

    # One index path on business days, so every chain on a date sees the
    # same spot and the panel overlaps exactly as the real one does.
    dates = pd.bdate_range(start, expirations[-1])
    steps = generator.normal(
        (market.drift - 0.5 * market.realised_volatility**2) / 252.0,
        market.realised_volatility / np.sqrt(252.0),
        size=len(dates),
    )
    path = pd.Series(
        market.initial_level * np.exp(np.cumsum(steps)), index=dates
    )

    # A slow-moving volatility level, so the smile is not identical every day.
    volatility = pd.Series(
        market.base_volatility * np.exp(
            market.volatility_of_volatility * np.cumsum(
                generator.normal(0.0, 0.05, size=len(dates))
            )
        ),
        index=dates,
    ).clip(0.08, 0.55)

    chains = []
    models = {}

    for expiration in expirations:
        for target in market.target_dtes:
            # Every business day on which this expiration is the nearest
            # one to this target, as the real selection produces.
            window = dates[
                (dates >= expiration - pd.Timedelta(days=target + 10))
                & (dates <= expiration - pd.Timedelta(days=max(target - 10, 7)))
            ]

            for data_date in window:
                chain, model = build_chain(
                    data_date, expiration, target,
                    float(path.loc[data_date]),
                    float(volatility.loc[data_date]),
                    market, generator,
                )
                key = (data_date, expiration, int(target))
                chains.append((key, chain))
                models[key] = model

    # Outcomes are drawn ONCE per expiration, from the physical measure of
    # the middle model observed for it (roughly a mid-window 60-day chain)
    # -- which is what makes all three horizons of one expiration score
    # against the same print, the dependence the whole inference section is
    # about. Only that source chain's density is an exact tilt of the
    # outcome distribution; every other chain of the expiration has its own
    # forward, volatility and maturity.
    settlement_rows = []
    for expiration in expirations:
        candidates = [
            models[key] for key in models if key[1] == expiration
        ]
        if not candidates:
            continue
        reference = candidates[len(candidates) // 2]
        outcome = float(reference.tilted(market.gamma).sample(1, generator)[0])
        settlement_rows.append({
            "expiration_year": expiration.year,
            "expiration_month": expiration.month,
            "settlement_date": expiration,
            "set_value": outcome,
        })

    settlement_values = pd.DataFrame(settlement_rows)

    truth = {
        "models": models,
        "gamma": market.gamma,
        "market": market,
        "index_path": path,
        "volatility_path": volatility,
    }

    return chains, settlement_values, truth


def write_synthetic_dataset(
    root: Path,
    market: SyntheticMarket | None = None,
    verbose: bool = True,
) -> dict:
    """Write a synthetic dataset in the layout the real pipeline produces.

    After this, ``data_access.load_chain_catalog()`` and
    ``load_settlement_values()`` work unchanged and ``02_project.ipynb``
    runs end to end without any vendor data.
    """
    root = Path(root)
    processed = root / "processed"
    spx = processed / "SPX"

    chains, settlement_values, truth = generate_sample(market)

    frame = pd.concat(
        [chain for _, chain in chains], ignore_index=True
    )

    catalog_rows = []
    for (year, month), group in frame.groupby(
        [frame["data_date"].dt.year, frame["data_date"].dt.month], sort=True
    ):
        year_dir = spx / str(int(year))
        year_dir.mkdir(parents=True, exist_ok=True)
        path = year_dir / f"spx_options_{int(year)}_{int(month):02d}.parquet"
        group.to_parquet(path, index=False)

        keys = group[
            ["data_date", "expiration_date", "target_dte_days"]
        ].drop_duplicates()
        keys["parquet_path"] = catalog_path(path)
        catalog_rows.append(keys)

    catalog = pd.concat(catalog_rows, ignore_index=True).sort_values(
        ["data_date", "expiration_date", "target_dte_days"], ignore_index=True
    )

    spx.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(spx / "chain_catalog.parquet", index=False)

    processed.mkdir(parents=True, exist_ok=True)
    settlement_values.to_parquet(
        processed / "cboe_spx_set_values.parquet", index=False
    )

    if verbose:
        print(
            f"{len(catalog):,} synthetic chains across "
            f"{catalog['expiration_date'].nunique()} expirations "
            f"written under {spx}"
        )
        print(f"True gamma used for the outcomes: {truth['gamma']}")

    return {
        "catalog": catalog,
        "settlement_values": settlement_values,
        "truth": truth,
        "root": root,
    }


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SPX dataset the pipeline can run on."
    )
    parser.add_argument(
        "--out", default=str(DATA_DIR),
        help="Data directory to write under (default: the project's data/).",
    )
    parser.add_argument(
        "--months", type=int, default=36, help="Number of expirations."
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--gamma", type=float, default=2.5)
    args = parser.parse_args(argv)

    market = SyntheticMarket(
        n_months=args.months, seed=args.seed, gamma=args.gamma
    )
    write_synthetic_dataset(Path(args.out), market)


if __name__ == "__main__":
    _main()
