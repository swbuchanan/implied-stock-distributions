"""Paths, sample definition and every tunable parameter used in the project.

All model hyperparameters live here rather than in call sites, so that the
figures in the notebooks and the numbers in the results tables are
guaranteed to come from the same configuration.

Importing this module reads nothing from disk. The one parameter that is
estimated rather than set -- the roughness penalty -- is chosen by
``selection.select_roughness_penalty`` and passed explicitly to every fit,
so no function's default can depend on a file outside the repository.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Directory structure
# --------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
SPX_RAW_DIR = DATA_DIR / "raw" / "SPX"
SPX_PROCESSED_DIR = DATA_DIR / "processed" / "SPX"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

CHAIN_CATALOG_PATH = SPX_PROCESSED_DIR / "chain_catalog.parquet"
SET_VALUES_PATH = PROCESSED_DIR / "cboe_spx_set_values.parquet"
CACHED_DENSITY_DIR = PROCESSED_DIR / "cached_densities"
SELECTED_PENALTY_PATH = PROCESSED_DIR / "selected_roughness_penalty.json"

CHAIN_COLUMNS = [
    "data_date",
    "expiration_date",
    "target_dte_days",
]

# --------------------------------------------------------------------------
# Analysis sample
# --------------------------------------------------------------------------

# Years for which the vendor files pass the duplicate-key and coverage
# checks in 01_processing.ipynb. The excluded years are listed explicitly
# rather than implied, so that the gaps in the sample are visible here and
# can be reported in the notebook.
ANALYSIS_YEARS: tuple[int, ...] = (
    2005, 2006, 2007, 2008, 2009, 2010,
    2014, 2015, 2016,
    2018,
    2021, 2022, 2023, 2024,
)

EXCLUDED_YEARS: tuple[int, ...] = (
    2011, 2012, 2013, 2017, 2019, 2020,
)

SET_DOWNLOAD_YEARS: tuple[int, ...] = tuple(
    range(min(ANALYSIS_YEARS), max(ANALYSIS_YEARS) + 1)
)

# Above this share of a year's rows sharing an option_key with a differing
# quote, the chains built from what remains cannot be attributed to a
# single settlement convention and the year is dropped. Stated as a number
# so the exclusion rule is reproducible from the coverage table rather than
# being a judgement call recorded only in prose.
MAX_DUPLICATE_KEY_SHARE = 0.02

# --------------------------------------------------------------------------
# Option-chain selection (used by 01_processing.ipynb)
# --------------------------------------------------------------------------

TARGET_DTES: tuple[int, ...] = (30, 60, 90)
TARGET_DTE_TOLERANCE_DAYS = 10
MINIMUM_DTE_DAYS = 7
STANDARD_MONTHLY_ONLY = True

# Day-count convention for time to expiry. SPX standard monthlies are
# AM-settled: the settlement print is taken from Friday's opening auction,
# so the option effectively stops trading at Thursday's close. Subtracting
# one calendar day is closer to the truth than measuring to the Friday.
DAY_COUNT_DAYS_PER_YEAR = 365.0
AM_SETTLEMENT_DAY_ADJUSTMENT = 1.0

# The same calendar clock is used for discounting and for volatility. That
# is a defensible default, but variance accumulates on trading days rather
# than calendar days, and around Thanksgiving and Christmas the difference
# moves thirty-day implied volatility by roughly a point. This supports the
# trading-clock robustness check in the processing notebook; nothing in the
# pipeline uses it.
TRADING_DAYS_PER_YEAR = 252.0

# --------------------------------------------------------------------------
# Forward and discount-factor estimation (put-call parity)
# --------------------------------------------------------------------------

# Parity holds at every strike in theory. In practice deep in- and
# out-of-the-money pairs are stale and wide, and they lever the regression
# slope, which is the discount factor itself. Restricting to a band around
# the spot keeps the estimate on the quotes that are actually traded.
PARITY_MONEYNESS_BAND = 0.10

MIN_PARITY_PAIRS = 10

# R^2 is a weak gate on a relationship this close to deterministic: a fit
# can clear 0.999 while sitting outside the quoted spreads at half the
# strikes, because the spread of the *data* in the numerator is enormous
# compared with the residual. The share of pairs whose parity residual
# exceeds the call plus put half-spread is the gate that means something, and it
# is the one to quote to a market maker. Both are enforced.
MIN_PARITY_R_SQUARED = 0.999
MAX_FRACTION_OUTSIDE_SPREADS = 0.25

# Plausible range for a continuously compounded annualised rate implied by
# a chain. Anything outside this is a data problem, not a rate.
IMPLIED_RATE_BOUNDS = (-0.02, 0.12)

PARITY_USE_SPREAD_WEIGHTS = True

# --------------------------------------------------------------------------
# Shape-constrained call-price spline (Fengler)
# --------------------------------------------------------------------------

# Weight on Fengler's roughness penalty. The fit minimises the *sum* of
# squared spread-standardised pricing errors plus this times the integrated
# squared curvature, so the meaningful range is set by the number of quotes:
# with a few hundred strikes the penalty only begins to bite above one, and
# dominates above ten thousand.
#
# The value is chosen by cross-validation (`selection.select_roughness_penalty`
# over ROUGHNESS_PENALTY_GRID below) and passed explicitly to every fit; the
# notebook and the demo both do this. `selection.save_selected_penalty`
# records the choice in SELECTED_PENALTY_PATH, and
# `selection.load_selected_penalty` reads it back, falling back to this
# default only when nothing has been saved.
ROUGHNESS_PENALTY_DEFAULT = 100.0

UPPER_SUPPORT_MULTIPLE = 2.5
N_TAIL_KNOTS = 3

# Floor on the assumed pricing-error scale, expressed as a fraction of the
# forward rather than in index points, so that it means the same thing at
# SPX 1200 and SPX 6000. The absolute floor covers the minimum tick.
MIN_ERROR_FRACTION_OF_FORWARD = 1e-5
MIN_ERROR_ABSOLUTE = 0.025

MIN_QUOTES_FOR_SPLINE = 8
DENSITY_GRID_POINTS = 1000

# Grid of candidate penalties for cross-validation, spanning from "no
# effect" to "heavily over-smoothed" on a typical chain.
ROUGHNESS_PENALTY_GRID: tuple[float, ...] = (
    1e-1, 1.0, 1e1, 1e2, 3e2, 1e3, 3e3, 1e4, 1e5,
)
CV_HOLDOUT_STRIDE = 5
CV_SAMPLE_SIZE = 300
CV_RANDOM_STATE = 20260830

# --------------------------------------------------------------------------
# Implied-volatility smoothing baseline
# --------------------------------------------------------------------------

# `make_splrep`'s smoothing factor is an upper bound on the weighted sum of
# squared residuals. With weights set to the reciprocal of the estimated
# log-IV quote noise, the statistically motivated bound is the number of
# observations, which is the expectation of that sum when the fit is
# correct. Using a fixed number instead (0.05 * n, say) leaves the bound
# slack at realistic noise levels and collapses the fit to a single cubic.
IV_SMOOTHING_CHI_SQUARE_MULTIPLE = 1.0
IV_MIN_LOG_NOISE = 5e-4
IV_MAX_LOG_NOISE = 0.25
IV_GRID_POINTS = 1000

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

CRPS_MASS_TOLERANCE = 1e-3
GAMMA_BOUNDS = (0.0, 5.0)
GAMMA_GRID_POINTS = 61

# Newey-West lag PER HORIZON, in trading days. Overlapping daily
# observations of the same expiration induce serial correlation out to
# roughly the forecast horizon.
#
# These are in DAYS. The score table carries one row per (date, horizon),
# so a pooled series needs the lag in ROWS: use `stats.pooled_hac_lag`,
# which multiplies by the rows per observation date. Passing
# max(DM_LAG_BY_HORIZON.values()) straight into `stats.diebold_mariano` on
# the pooled panel sets the window to a third of what it should be.
DM_LAG_BY_HORIZON = {30: 21, 60: 42, 90: 63}

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_RANDOM_STATE = 20260830

RANDOM_SEED = 20260830

# --------------------------------------------------------------------------
# External series used to validate the pipeline (see validation.py)
# --------------------------------------------------------------------------

# The implied short rate, the implied dividend yield and the risk-neutral
# skewness of the fitted densities are all outputs of this pipeline that a
# published series can be compared against. All three sources are free and
# fetchable without a key.
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SHORT_RATE_SERIES: tuple[str, ...] = ("DGS1MO", "DGS3MO")

CBOE_INDEX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"
)
CBOE_SKEW_INDEX = "SKEW"
CBOE_VIX_INDEX = "VIX"

# Shiller's monthly S&P 500 series, mirrored as clean CSV. `Dividend` is
# the trailing twelve-month dividend per index point, so the yield is
# `Dividend / SP500`.
SHILLER_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
)

EXTERNAL_CACHE_PATHS = {
    "short_rate": EXTERNAL_DIR / "fred_short_rates.parquet",
    "skew": EXTERNAL_DIR / "cboe_skew.parquet",
    "vix": EXTERNAL_DIR / "cboe_vix.parquet",
    "dividend_yield": EXTERNAL_DIR / "shiller_sp500.parquet",
}

# Buckets for the conditioning check: if chains fail the parity gates more
# often when volatility is high, the whole evaluation is conditioned on
# quiet markets and every tail conclusion weakens.
VIX_BUCKET_EDGES: tuple[float, ...] = (0.0, 15.0, 20.0, 30.0, 100.0)
VIX_BUCKET_LABELS: tuple[str, ...] = (
    "VIX < 15", "15-20", "20-30", "VIX > 30",
)
