from pathlib import Path

import pandas as pd

from .config import (
    ANALYSIS_YEARS,
    CHAIN_CATALOG_PATH,
    PROJECT_ROOT,
    SPX_PROCESSED_DIR,
)


CHAIN_COLUMNS = [
    "data_date",
    "expiration_date",
    "target_dte_days",
]


def discover_data_files(
    data_dir: Path = SPX_PROCESSED_DIR,
    years: tuple[int, ...] = ANALYSIS_YEARS,
) -> tuple[dict[int, list[Path]], list[Path]]:
    """Locate the monthly Parquet files used in the analysis."""

    files_by_year = {
        year: sorted(
            (data_dir / str(year)).glob(
                f"spx_options_{year}_[0-9][0-9].parquet"
            )
        )
        for year in years
    }

    files = [
        path
        for year_files in files_by_year.values()
        for path in year_files
    ]

    return files_by_year, files


def load_chain_catalog() -> pd.DataFrame:
    """Load the chain catalogue created during preprocessing."""

    if not CHAIN_CATALOG_PATH.exists():
        raise FileNotFoundError(
            "Chain catalogue not found. Run preprocessing first: "
            f"{CHAIN_CATALOG_PATH}"
        )

    return pd.read_parquet(CHAIN_CATALOG_PATH)


def resolve_parquet_path(path_value: str | Path) -> Path:
    """Resolve paths stored in the catalogue."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_chain(info: pd.Series) -> pd.DataFrame:
    """Load one chain described by a catalogue row."""

    path = resolve_parquet_path(info["parquet_path"])
    monthly = pd.read_parquet(path)

    mask = (
        monthly["data_date"].eq(info["data_date"])
        & monthly["expiration_date"].eq(
            info["expiration_date"]
        )
        & monthly["target_dte_days"].eq(
            info["target_dte_days"]
        )
    )

    chain = monthly.loc[mask].copy()

    if chain.empty:
        raise KeyError("The requested chain was not found")

    return chain.sort_values(
        ["put_call", "strike_price"],
        ignore_index=True,
    )


def iter_chains(catalog: pd.DataFrame):
    """Iterate through chains while loading each month only once."""

    for stored_path, file_catalog in catalog.groupby(
        "parquet_path",
        sort=False,
    ):
        path = resolve_parquet_path(stored_path)
        monthly = pd.read_parquet(path)

        grouped = monthly.groupby(
            CHAIN_COLUMNS,
            sort=False,
        )

        for info in file_catalog.itertuples(index=False):
            key = (
                info.data_date,
                info.expiration_date,
                int(info.target_dte_days),
            )

            chain = (
                grouped.get_group(key)
                .sort_values(["put_call", "strike_price"])
                .reset_index(drop=True)
                .copy()
            )

            yield key, chain
