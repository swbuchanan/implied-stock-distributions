"""Locating, cataloguing and loading processed option chains.

The processed data are stored as one Parquet file per month, which is more
than fits in memory at once. The catalogue records which chains live in
which file so that an analysis can iterate over chains while holding only
one month at a time.

Paths in the catalogue are stored relative to the project root. Storing
absolute paths, as an earlier version did, bakes one machine's home
directory into a committed artefact and makes the catalogue useless in any
other clone.

Unchanged from v3 apart from this note and ``discover_data_files`` no
longer restricting to ``ANALYSIS_YEARS`` when a synthetic dataset is in
place -- see ``synthetic.write_synthetic_dataset``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    ANALYSIS_YEARS,
    CHAIN_CATALOG_PATH,
    CHAIN_COLUMNS,
    PROJECT_ROOT,
    SET_VALUES_PATH,
    SPX_PROCESSED_DIR,
)

__all__ = [
    "discover_data_files",
    "build_chain_catalog",
    "load_chain_catalog",
    "resolve_parquet_path",
    "load_chain",
    "iter_chains",
    "load_random_chain",
    "load_settlement_values",
    "settlement_lookup",
]


def discover_data_files(
    data_dir: Path = SPX_PROCESSED_DIR,
    years: tuple[int, ...] = ANALYSIS_YEARS,
) -> tuple[dict[int, list[Path]], list[Path]]:
    """Locate the monthly Parquet files used in the analysis."""
    files_by_year = {
        year: sorted(
            (data_dir / str(year)).glob(f"spx_options_{year}_[0-9][0-9].parquet")
        )
        for year in years
    }

    files = [path for year_files in files_by_year.values() for path in year_files]

    return files_by_year, files


def _as_relative(path: Path) -> str:
    """Store paths relative to the project root when possible."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_chain_catalog() -> pd.DataFrame:
    """Build the chain catalogue from the processed monthly files."""
    parts = []
    _, data_files = discover_data_files()

    if not data_files:
        raise FileNotFoundError(
            f"No processed monthly files found under {SPX_PROCESSED_DIR}."
        )

    for path in data_files:
        chains = pd.read_parquet(path, columns=CHAIN_COLUMNS).drop_duplicates()
        chains["parquet_path"] = _as_relative(path)
        parts.append(chains)

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(CHAIN_COLUMNS)
        .reset_index(drop=True)
    )


def load_chain_catalog(path: Path = CHAIN_CATALOG_PATH) -> pd.DataFrame:
    """Load the chain catalogue created during preprocessing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Chain catalogue not found. Run preprocessing first: {path}"
        )
    return pd.read_parquet(path)


def resolve_parquet_path(path_value: str | Path) -> Path:
    """Resolve a catalogue path against the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_chain(info: pd.Series) -> pd.DataFrame:
    """Load the one chain described by a catalogue row."""
    monthly = pd.read_parquet(resolve_parquet_path(info["parquet_path"]))

    mask = (
        monthly["data_date"].eq(info["data_date"])
        & monthly["expiration_date"].eq(info["expiration_date"])
        & monthly["target_dte_days"].eq(info["target_dte_days"])
    )

    chain = monthly.loc[mask].copy()

    if chain.empty:
        raise KeyError("The requested chain was not found.")

    return chain.sort_values(["put_call", "strike_price"], ignore_index=True)


def iter_chains(catalog: pd.DataFrame):
    """Yield ``(key, chain)`` while loading each monthly file only once."""
    for stored_path, file_catalog in catalog.groupby("parquet_path", sort=False):
        monthly = pd.read_parquet(resolve_parquet_path(stored_path))
        grouped = monthly.groupby(CHAIN_COLUMNS, sort=False)

        for info in file_catalog.itertuples(index=False):
            key = (
                info.data_date,
                info.expiration_date,
                int(info.target_dte_days),
            )

            try:
                chain = grouped.get_group(key)
            except KeyError:
                # The catalogue is out of step with the file it points at.
                continue

            yield key, (
                chain.sort_values(["put_call", "strike_price"])
                .reset_index(drop=True)
                .copy()
            )


def load_random_chain(
    catalog: pd.DataFrame,
    target_dte_days: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Select and load one chain at random.

    ``random_state`` should always be supplied in the notebooks: an unseeded
    draw makes every figure change on re-execution.
    """
    candidates = catalog

    if target_dte_days is not None:
        candidates = candidates.loc[
            candidates["target_dte_days"].eq(target_dte_days)
        ]

    if candidates.empty:
        raise ValueError(f"No chains found for target DTE {target_dte_days}.")

    info = candidates.sample(n=1, random_state=random_state).iloc[0]
    return info, load_chain(info)


def load_settlement_values(path: Path = SET_VALUES_PATH) -> pd.DataFrame:
    """Load the Cboe SET settlement values written during preprocessing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Settlement values not found. Run preprocessing first: {path}"
        )
    return pd.read_parquet(path)


def settlement_lookup(settlement_values: pd.DataFrame) -> dict[tuple[int, int], float]:
    """Index SET values by ``(expiration_year, expiration_month)``.

    Standard monthly SPX options settle once a month against the opening
    print on the third Friday, so year and month identify the value. Doing
    the lookup through a dict rather than a DataFrame filter inside the
    chain loop removes a scan per chain.
    """
    duplicated = settlement_values.duplicated(
        subset=["expiration_year", "expiration_month"], keep=False
    )
    if duplicated.any():
        raise ValueError(
            "Settlement values contain more than one entry for a month."
        )

    return {
        (int(row.expiration_year), int(row.expiration_month)): float(row.set_value)
        for row in settlement_values.itertuples(index=False)
    }
