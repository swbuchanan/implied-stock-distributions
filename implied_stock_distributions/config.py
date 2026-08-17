from pathlib import Path

# Directory structure
PACKAGE_DIR = Path(__file__).resolve().parent
print(PACKAGE_DIR)
PROJECT_ROOT = PACKAGE_DIR.parent
print(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
SPX_RAW_DIR = DATA_DIR / "raw" / "SPX"
SPX_PROCESSED_DIR = DATA_DIR / "processed" / "SPX"

CHAIN_CATALOG_PATH = (
    SPX_PROCESSED_DIR / "chain_catalog.parquet"
)

CHAIN_COLUMNS = [
    "data_date",
    "expiration_date",
    "target_dte_days",
]

# Analysis sample
ANALYSIS_YEARS: tuple[int, ...] = (
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2014,
    2015,
    2016,
    2018,
    2021,
    2022,
    2023,
    2024,
)

# Option-chain selection
TARGET_DTES: tuple[int, ...] = (30, 60, 90)
TARGET_DTE_TOLERANCE_DAYS = 10
