"""Clean daily SPX option CSVs and write monthly Parquet files.

Output layout:

    processed/SPX/2021/spx_options_2021_01.parquet
    processed/SPX/2021/spx_options_2021_01_invalid.parquet

Only the option chains nearest to 30 and 90 calendar days are retained.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("raw/SPX")
PROCESSED_DIR = Path("processed/SPX")
OVERWRITE = True

VALID_SYMBOLS = {"SPX"}

# Exact 30- and 90-day expirations rarely exist. For each observation date,
# retain the nearest distinct expiration to each target, provided that it is
# no more than this many calendar days away from the target.
TARGET_DTES = (30, 90)
TARGET_DTE_TOLERANCE = 20
MINIMUM_DTE = 7
STANDARD_MONTHLY_ONLY = True

REQUIRED_COLUMNS = {
    "OptionKey",
    "Symbol",
    "ExpirationDate",
    "AskPrice",
    "AskSize",
    "BidPrice",
    "BidSize",
    "LastPrice",
    "PutCall",
    "StrikePrice",
    "Volume",
    "OpenInterest",
    "UnderlyingPrice",
    "DataDate",
}

COLUMN_NAMES = {
    "OptionKey": "option_key",
    "Symbol": "symbol",
    "ExpirationDate": "expiration_date",
    "AskPrice": "ask_price",
    "AskSize": "ask_size",
    "BidPrice": "bid_price",
    "BidSize": "bid_size",
    "LastPrice": "last_price",
    "PutCall": "put_call",
    "StrikePrice": "strike_price",
    "Volume": "volume",
    "OpenInterest": "open_interest",
    "UnderlyingPrice": "underlying_price",
    "DataDate": "data_date",
}

READ_DTYPES = {
    "OptionKey": "string",
    "Symbol": "string",
    "AskPrice": "float64",
    "AskSize": "Int64",
    "BidPrice": "float64",
    "BidSize": "Int64",
    "LastPrice": "float64",
    "PutCall": "string",
    "StrikePrice": "float64",
    "Volume": "Int64",
    "OpenInterest": "Int64",
    "UnderlyingPrice": "float64",
}

# Columns created while preparing a daily file.
DAILY_COLUMNS = [
    "option_key",
    "symbol",
    "data_date",
    "expiration_date",
    "put_call",
    "strike_price",
    "underlying_price",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
    "last_price",
    "volume",
    "open_interest",
    "dte_calendar_days",
    "moneyness_k_over_s",
    "log_moneyness_k_over_s",
    "quote_is_valid",
    "quote_is_two_sided",
    "source_file",
]

# target_dte_days records whether a chain was selected for the 30- or 90-day
# horizon. dte_calendar_days records its actual time to expiration.
OUTPUT_COLUMNS = [
    *DAILY_COLUMNS[:15],
    "target_dte_days",
    *DAILY_COLUMNS[15:],
]


def parse_date_column(series: pd.Series) -> pd.Series:
    """Parse both date formats found in the source files."""
    parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    unresolved = parsed.isna()
    parsed.loc[unresolved] = pd.to_datetime(
        series.loc[unresolved],
        format="%m/%d/%Y",
        errors="coerce",
    )
    return parsed


def inspect_file_data_date(path: Path) -> pd.Timestamp:
    """Read only DataDate and return the most common valid date in a CSV."""
    try:
        date_text = pd.read_csv(
            path,
            usecols=["DataDate"],
            dtype={"DataDate": "string"},
        )["DataDate"]
    except ValueError as exc:
        raise ValueError(f"{path.name}: missing DataDate column") from exc

    dates = parse_date_column(date_text).dropna().dt.normalize()
    if dates.empty:
        raise ValueError(f"{path.name}: no valid DataDate values")

    counts = dates.value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        raise ValueError(f"{path.name}: cannot determine a single data date")
    return pd.Timestamp(counts.index[0])


def discover_files_by_month(
    raw_dir: Path,
) -> dict[tuple[int, int], list[tuple[Path, pd.Timestamp]]]:
    """Group daily CSV paths by the year and month of their DataDate."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files_by_month = defaultdict(list)
    for year_dir in sorted(raw_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            folder_year = int(year_dir.name)
        except ValueError:
            print(f"Warning: skipping non-year directory {year_dir}")
            continue

        for path in sorted(year_dir.glob("*.csv")):
            data_date = inspect_file_data_date(path)
            if data_date.year != folder_year:
                print(
                    f"Warning: {path.name}: DataDate year {data_date.year} "
                    f"does not match directory {folder_year}"
                )
            key = (int(data_date.year), int(data_date.month))
            files_by_month[key].append((path, data_date))

    return dict(files_by_month)


def read_and_prepare_daily_file(
    path: Path,
    expected_data_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read, validate, standardize, and enrich one daily CSV file."""
    raw_df = pd.read_csv(path, dtype=READ_DTYPES)

    # Check the schema before accessing individual columns.
    missing = REQUIRED_COLUMNS.difference(raw_df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    # Normalize before filtering, so values such as " spx" are retained.
    raw_df["Symbol"] = raw_df["Symbol"].str.strip().str.upper()
    valid_symbol = raw_df["Symbol"].isin(VALID_SYMBOLS)
    if not valid_symbol.all():
        invalid_symbols = sorted(
            raw_df.loc[~valid_symbol, "Symbol"].dropna().unique().tolist()
        )
        print(
            f"Warning: {path.name}: discarding "
            f"{int((~valid_symbol).sum())} rows with invalid symbols "
            f"{invalid_symbols}"
        )
        raw_df = raw_df.loc[valid_symbol].copy()

    if raw_df.empty:
        raise ValueError(f"{path.name}: no SPX rows remain")

    df = raw_df.loc[:, list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES).copy()
    df["data_date"] = parse_date_column(df["data_date"])
    df["expiration_date"] = parse_date_column(df["expiration_date"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["put_call"] = df["put_call"].str.strip().str.lower()

    bad_date_rows = df[["data_date", "expiration_date"]].isna().any(axis=1)
    if bad_date_rows.any():
        print(
            f"Warning: {path.name}: discarding "
            f"{int(bad_date_rows.sum())} rows with invalid dates"
        )
        df = df.loc[~bad_date_rows].copy()

    if df.empty:
        raise ValueError(f"{path.name}: all rows have invalid dates")

    # Keep the modal date identified during discovery and discard anomalous
    # dates. The previous duplicated() approach removed the normal rows.
    expected_data_date = pd.Timestamp(expected_data_date).normalize()
    wrong_data_date = df["data_date"].dt.normalize() != expected_data_date
    if wrong_data_date.any():
        print(
            f"Warning: {path.name}: discarding "
            f"{int(wrong_data_date.sum())} rows whose DataDate is not "
            f"{expected_data_date.date()}"
        )
        df = df.loc[~wrong_data_date].copy()

    valid_put_call = df["put_call"].isin(["call", "put"])
    if not valid_put_call.all():
        print(
            f"Warning: {path.name}: discarding "
            f"{int((~valid_put_call).sum())} rows with invalid PutCall"
        )
        df = df.loc[valid_put_call].copy()

    positive_prices = (df["strike_price"] > 0) & (df["underlying_price"] > 0)
    if not positive_prices.all():
        print(
            f"Warning: {path.name}: discarding "
            f"{int((~positive_prices).sum())} rows with non-positive "
            "strike or underlying prices"
        )
        df = df.loc[positive_prices].copy()

    if df.empty:
        raise ValueError(f"{path.name}: all rows failed validation")

    df["quote_is_valid"] = (
        (df["ask_price"] > 0)
        & (df["bid_price"] >= 0)
        & (df["ask_price"] >= df["bid_price"])
    ).astype("boolean")
    df["quote_is_two_sided"] = (
        df["quote_is_valid"] & (df["bid_price"] > 0)
    ).astype("boolean")

    df["dte_calendar_days"] = (
        df["expiration_date"] - df["data_date"]
    ).dt.days.astype("Int16")
    df["moneyness_k_over_s"] = df["strike_price"] / df["underlying_price"]
    df["log_moneyness_k_over_s"] = np.log(df["moneyness_k_over_s"])
    df["source_file"] = path.name

    return df.loc[:, DAILY_COLUMNS]


def is_standard_monthly_expiration(series: pd.Series) -> pd.Series:
    """Identify standard monthly dates under old and new date conventions.

    Older files record the Saturday following the third Friday. Newer files
    generally record the third Friday itself. A Thursday is allowed for the
    occasional holiday-adjusted third-Friday expiration. The original option
    root remains a better settlement identifier when it is available.
    """
    dates = pd.to_datetime(series)
    weekday = dates.dt.dayofweek
    day = dates.dt.day

    third_friday = (weekday == 4) & day.between(15, 21)
    saturday_after_third_friday = (weekday == 5) & day.between(16, 22)
    holiday_thursday = (weekday == 3) & day.between(14, 20)

    return third_friday | saturday_after_third_friday | holiday_thursday


def select_target_expirations(
    df: pd.DataFrame,
    targets: tuple[int, ...] = TARGET_DTES,
    tolerance: int = TARGET_DTE_TOLERANCE,
) -> pd.DataFrame:
    """Retain the nearest distinct expiration to each target DTE per date."""
    eligible = df["dte_calendar_days"] >= MINIMUM_DTE
    if STANDARD_MONTHLY_ONLY:
        eligible &= is_standard_monthly_expiration(df["expiration_date"])

    expiry_table = df.loc[
        eligible,
        ["data_date", "expiration_date", "dte_calendar_days"],
    ].drop_duplicates()

    selections = []
    for data_date, available in expiry_table.groupby("data_date", sort=False):
        available = available.copy()
        selected_expirations = set()

        for target in targets:
            candidates = available.loc[
                ~available["expiration_date"].isin(selected_expirations)
            ].copy()
            if candidates.empty:
                continue

            candidates["distance"] = (
                candidates["dte_calendar_days"] - target
            ).abs()
            # If two expirations are equally close, prefer the longer one.
            candidates["is_shorter_than_target"] = (
                candidates["dte_calendar_days"] < target
            )
            choice = candidates.sort_values(
                ["distance", "is_shorter_than_target", "expiration_date"],
                kind="stable",
            ).iloc[0]

            if int(choice["distance"]) > tolerance:
                continue

            selected_expirations.add(choice["expiration_date"])
            selections.append(
                {
                    "data_date": data_date,
                    "expiration_date": choice["expiration_date"],
                    "target_dte_days": target,
                }
            )

    selection_table = pd.DataFrame(
        selections,
        columns=["data_date", "expiration_date", "target_dte_days"],
    )
    if selection_table.empty:
        print(
            "Warning: no eligible standard-monthly chains were found "
            f"within {tolerance} days of targets {targets}"
        )
        empty = df.iloc[0:0].copy()
        empty["target_dte_days"] = pd.Series(dtype="Int16")
        return empty.loc[:, OUTPUT_COLUMNS]

    selected = df.merge(
        selection_table,
        on=["data_date", "expiration_date"],
        how="inner",
        validate="many_to_one",
    )
    selected["target_dte_days"] = selected["target_dte_days"].astype("Int16")

    expected_chains = df["data_date"].nunique() * len(targets)
    if len(selection_table) < expected_chains:
        print(
            "Warning: selected "
            f"{len(selection_table):,} of {expected_chains:,} requested "
            "date/horizon chains"
        )

    return selected.loc[:, OUTPUT_COLUMNS]


def prepare_month(
    files: list[tuple[Path, pd.Timestamp]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine and validate one month, then retain 30- and 90-day chains."""
    frames = [
        read_and_prepare_daily_file(path, data_date)
        for path, data_date in sorted(files, key=lambda item: (item[1], item[0].name))
    ]
    monthly = pd.concat(frames, ignore_index=True, copy=False)

    invalid_parts = []

    # OptionKey contains DataDate, so duplicate groups cannot cross months.
    duplicate_keys = monthly["option_key"].duplicated(keep=False)
    if duplicate_keys.any():
        duplicate_rows = monthly.loc[duplicate_keys].copy()
        duplicate_rows["invalid_reason"] = "duplicate_option_key"
        invalid_parts.append(duplicate_rows)
        examples = duplicate_rows["option_key"].drop_duplicates().head(5).tolist()
        print(
            "Warning: duplicate option_key values found; examples: "
            f"{examples}. Discarding {int(duplicate_keys.sum()):,} rows "
            f"({duplicate_keys.mean() * 100:.2f}% of the month)."
        )
        monthly = monthly.loc[~duplicate_keys].copy()

    expired = monthly["expiration_date"] < monthly["data_date"]
    if expired.any():
        expired_rows = monthly.loc[expired].copy()
        expired_rows["invalid_reason"] = "expiration_before_data_date"
        invalid_parts.append(expired_rows)
        print(
            f"Warning: discarding {int(expired.sum()):,} rows with "
            "expiration before DataDate"
        )
        monthly = monthly.loc[~expired].copy()

    if invalid_parts:
        invalid_data = pd.concat(invalid_parts, ignore_index=True, copy=False)
    else:
        invalid_data = monthly.iloc[0:0].copy()
        invalid_data["invalid_reason"] = pd.Series(dtype="string")

    monthly = select_target_expirations(monthly)
    monthly = monthly.sort_values(
        [
            "data_date",
            "target_dte_days",
            "expiration_date",
            "put_call",
            "strike_price",
        ],
        kind="stable",
        ignore_index=True,
    )
    return monthly, invalid_data


def write_parquet(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool = OVERWRITE,
) -> None:
    """Write one compressed monthly Parquet file."""
    if path.exists() and not overwrite:
        print(f"Warning: {path} already exists; skipping write")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def process_all_months(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    overwrite: bool = OVERWRITE,
) -> None:
    """Process every discovered month without constructing annual DataFrames."""
    files_by_month = discover_files_by_month(raw_dir)

    for (year, month), files in sorted(files_by_month.items()):
        monthly, invalid_data = prepare_month(files)

        year_dir = processed_dir / str(year)
        stem = f"spx_options_{year}_{month:02d}"
        output_path = year_dir / f"{stem}.parquet"
        invalid_path = year_dir / f"{stem}_invalid.parquet"

        write_parquet(monthly, output_path, overwrite=overwrite)
        write_parquet(invalid_data, invalid_path, overwrite=overwrite)

        valid_quotes = int(monthly["quote_is_valid"].sum())
        two_sided_quotes = int(monthly["quote_is_two_sided"].sum())
        raw_chain_counts = (
            monthly[["data_date", "expiration_date", "target_dte_days"]]
            .drop_duplicates()["target_dte_days"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        chain_counts = {
            int(target): int(count) for target, count in raw_chain_counts.items()
        }
        print(
            f"{year}-{month:02d}: {len(files)} files, {len(monthly):,} rows, "
            f"{valid_quotes:,} valid quotes, "
            f"{two_sided_quotes:,} two-sided quotes, "
            f"chains by target DTE {chain_counts} -> {output_path}"
        )
        print(f"{len(invalid_data):,} invalid rows -> {invalid_path}")

        # Make it explicit that the completed month can be released before the
        # next one is constructed.
        del monthly, invalid_data


if __name__ == "__main__":
    process_all_months()
