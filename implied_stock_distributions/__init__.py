"""Risk-neutral and physical densities for the S&P 500 from SPX option prices.

Modules
-------
``config``      paths, sample definition and every model hyperparameter
``pricing``     Black's formula on a forward, and its inversion
``forwards``    forward and discount factor from put-call parity
``fitting``     shape-constrained (Fengler-style) call-price spline
``smoothing``   the unconstrained implied-volatility baseline
``selection``   cross-validated choice of the roughness penalty
``evaluation``  CRPS, PIT, and the CRRA tilt from risk-neutral to physical
``pipeline``    one chain end to end: both fits, a benchmark and the scores
``benchmarks``  reference forecasts to score against
``stats``       Diebold-Mariano, block bootstrap, calibration tests
``validation``  the pipeline's outputs against published external series
``synthetic``   synthetic chains with a known true density, for the demo
                and the test suite
``demo``        the whole pipeline on synthetic data, with no vendor files
``data_access`` locating, cataloguing and loading processed chains
``plotting``    the figures used in the analysis notebook
"""

from .config import ANALYSIS_YEARS, TARGET_DTES
from .errors import (
    ChainError,
    ChainQualityError,
    InsufficientQuotesError,
    SolverError,
    failure_reason,
)
from .evaluation import (
    common_strike_grid,
    crps_from_cdf,
    crps_from_density_grid,
    crps_location_sensitivity,
    distribution_moments,
    gamma_plausible_interval,
    implied_equity_risk_premium,
    mean_relative_crps_for_gamma,
    physical_density_from_rn_grid,
    pit_for_gamma,
    probability_integral_transform,
    recentre_on_forward,
    relative_crps_curve,
    relative_crps_for_gamma,
    tilt_moment_summary,
    tilted_distribution,
)
from .fitting import (
    construct_risk_neutral_distribution_from_spline,
    fit_shape_constrained_call_spline,
    select_call_equivalent_quotes,
)
from .forwards import estimate_forward_and_rate, quality_failures
from .pipeline import score_chain
from .pricing import (
    black_call_price,
    black_put_price,
    black_vega,
    implied_volatility_call,
)
from .smoothing import create_density_from_chain
from .stats import (
    block_bootstrap_mean,
    diebold_mariano,
    effective_sample_size,
    nearest_to_target,
    one_per_expiration,
    pit_uniformity,
    pooled_hac_lag,
    skill_score,
)

__all__ = [
    "ANALYSIS_YEARS",
    "TARGET_DTES",
    "ChainError",
    "ChainQualityError",
    "InsufficientQuotesError",
    "SolverError",
    "failure_reason",
    "black_call_price",
    "black_put_price",
    "black_vega",
    "implied_volatility_call",
    "estimate_forward_and_rate",
    "quality_failures",
    "select_call_equivalent_quotes",
    "fit_shape_constrained_call_spline",
    "construct_risk_neutral_distribution_from_spline",
    "create_density_from_chain",
    "common_strike_grid",
    "crps_from_cdf",
    "crps_from_density_grid",
    "probability_integral_transform",
    "distribution_moments",
    "recentre_on_forward",
    "score_chain",
    "physical_density_from_rn_grid",
    "mean_relative_crps_for_gamma",
    "relative_crps_for_gamma",
    "relative_crps_curve",
    "gamma_plausible_interval",
    "pit_for_gamma",
    "tilted_distribution",
    "implied_equity_risk_premium",
    "crps_location_sensitivity",
    "tilt_moment_summary",
    "block_bootstrap_mean",
    "diebold_mariano",
    "pooled_hac_lag",
    "nearest_to_target",
    "one_per_expiration",
    "effective_sample_size",
    "pit_uniformity",
    "skill_score",
]

