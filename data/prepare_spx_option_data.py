"""Prepare daily SPX option-chain CSV files as one Parquet file per year.

Example
-------
python prepare_spx_option_data.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --start-year 2005 \
    --end-year 2024

The script keeps the raw quote fields, standardises names and types, and adds
useful derived columns.  It validates rather than silently dropping suspect
observations.  Parquet output requires pandas and pyarrow.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


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

OUTPUT_COLUMNS = [
    "option_key",
    "symbol",
    "data_date",
    "expiration_date",
    "put_call",
    "strike_price",
    "underlying_price",
    "bid_price",
    "ask_price",
    "mid_price",
    "bid_ask_spread",
    "relative_bid_ask_spread",
    "bid_size",
    "ask_size",
    "last_price",
    "volume",
    "open_interest",
    "dte_calendar_days",
    "time_to_expiry_years_act36525",
    "moneyness_k_over_s",
    "log_moneyness_k_over_s",
    "quote_is_valid",
    "quote_is_two_sided",
    "is_otm",
    "source_file",
]

FILE_DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


def file_date(path: Path) -> pd.Timestamp:
    """Read YYYYMMDD from a filename such as 20050103_OData_SPX.csv."""
    match = FILE_DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"No YYYYMMDD date found in filename: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="raise")


def discover_files(raw_dir: Path, pattern: str) -> dict[int, list[Path]]:
    """Find input files recursively and group them by filename year."""
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(raw_dir.rglob(pattern)):
        if path.is_file():
            grouped[file_date(path).year].append(path)
    return dict(grouped)


def read_and_prepare_daily_file(path: Path) -> pd.DataFrame:
    """Read, validate, and enrich one daily CSV file."""
    raw = pd.read_csv(path, dtype=READ_DTYPES)

    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    extra = set(raw.columns).difference(REQUIRED_COLUMNS)
    if extra:
        print(f"Warning: {path.name}: ignoring extra columns {sorted(extra)}")

    df = raw.loc[:, list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES).copy()
    df["data_date"] = pd.to_datetime(
        df["data_date"], format="%Y-%m-%d", errors="coerce"
    )
    df["expiration_date"] = pd.to_datetime(
        df["expiration_date"], format="%Y-%m-%d", errors="coerce"
    )
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["put_call"] = df["put_call"].str.strip().str.lower()

    if df[["data_date", "expiration_date"]].isna().any().any():
        raise ValueError(f"{path.name}: invalid or missing date")
    if df["data_date"].nunique() != 1:
        raise ValueError(f"{path.name}: expected exactly one data date")
    if df["data_date"].iloc[0].normalize() != file_date(path):
        raise ValueError(f"{path.name}: filename date and DataDate do not match")
    if not df["put_call"].isin(["call", "put"]).all():
        raise ValueError(f"{path.name}: PutCall must contain only call or put")
    if (df["strike_price"] <= 0).any() or (df["underlying_price"] <= 0).any():
        raise ValueError(f"{path.name}: strike and underlying prices must be positive")
    if (df["expiration_date"] < df["data_date"]).any():
        raise ValueError(f"{path.name}: expiration before data date")

    # A zero bid can occur for illiquid options.  Keep it, but flag the quote as
    # one-sided.  A non-positive ask or a crossed/negative quote is invalid.
    df["quote_is_valid"] = (
        (df["ask_price"] > 0)
        & (df["bid_price"] >= 0)
        & (df["ask_price"] >= df["bid_price"])
    ).astype("boolean")
    df["quote_is_two_sided"] = (
        df["quote_is_valid"] & (df["bid_price"] > 0)
    ).astype("boolean")

    midpoint = (df["bid_price"] + df["ask_price"]) / 2.0
    spread = df["ask_price"] - df["bid_price"]
    df["mid_price"] = midpoint.where(df["quote_is_valid"])
    df["bid_ask_spread"] = spread.where(df["quote_is_valid"])
    df["relative_bid_ask_spread"] = (
        df["bid_ask_spread"] / df["mid_price"]
    ).where(df["mid_price"] > 0)

    df["dte_calendar_days"] = (
        df["expiration_date"] - df["data_date"]
    ).dt.days.astype("Int32")
    df["time_to_expiry_years_act36525"] = df["dte_calendar_days"] / 365.25

    # Moneyness conventions differ, so the column name states the definition.
    df["moneyness_k_over_s"] = df["strike_price"] / df["underlying_price"]
    df["log_moneyness_k_over_s"] = np.log(df["moneyness_k_over_s"])
    df["is_otm"] = (
        ((df["put_call"] == "call") & (df["strike_price"] > df["underlying_price"]))
        | ((df["put_call"] == "put") & (df["strike_price"] < df["underlying_price"]))
    ).astype("boolean")

    df["source_file"] = path.name
    return df.loc[:, OUTPUT_COLUMNS]


def prepare_year(files: list[Path]) -> pd.DataFrame:
    """Combine all daily files for one year and perform annual checks."""
    annual = pd.concat(
        [read_and_prepare_daily_file(path) for path in files],
        ignore_index=True,
    )

    duplicate_keys = annual["option_key"].duplicated(keep=False)
    if duplicate_keys.any():
        examples = annual.loc[duplicate_keys, "option_key"].head(5).tolist()
        raise ValueError(f"Duplicate option_key values found; examples: {examples}")

    return annual.sort_values(
        ["data_date", "expiration_date", "put_call", "strike_price"],
        kind="stable",
        ignore_index=True,
    )


def write_parquet(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    """Write one compressed annual Parquet file."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output needs pyarrow. Install it with: python -m pip install pyarrow"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*_OData_SPX.csv")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and summarise without writing Parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = discover_files(args.raw_dir, args.pattern)
    years = [
        year
        for year in sorted(grouped)
        if (args.start_year is None or year >= args.start_year)
        and (args.end_year is None or year <= args.end_year)
    ]
    if not years:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} found for the requested years"
        )

    for year in years:
        annual = prepare_year(grouped[year])
        valid = int(annual["quote_is_valid"].sum())
        two_sided = int(annual["quote_is_two_sided"].sum())
        message = (
            f"{year}: {len(grouped[year])} files, {len(annual):,} rows, "
            f"{valid:,} valid quotes, {two_sided:,} two-sided quotes"
        )
        if args.dry_run:
            print(message + " [dry run]")
            continue

        output_path = args.output_dir / f"spx_options_{year}.parquet"
        write_parquet(annual, output_path, overwrite=args.overwrite)
        print(message + f" -> {output_path}")


if __name__ == "__main__":
    main()
