"""Build a small synthetic processed dataset so the notebooks can be smoke-run.

This writes monthly Parquet files, a chain catalogue and a settlement-value
table in the same layout the real preprocessing produces, using
Black-Scholes chains rather than market data. It exists so the analysis
notebook can be executed end to end without the vendor files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from implied_stock_distributions.config import (
    AM_SETTLEMENT_DAY_ADJUSTMENT,
    DAY_COUNT_DAYS_PER_YEAR,
    SET_VALUES_PATH,
    SPX_PROCESSED_DIR,
)
from implied_stock_distributions.pricing import black_call_price, black_put_price


def third_friday(year: int, month: int) -> pd.Timestamp:
    start = pd.Timestamp(year=year, month=month, day=1)
    return start + pd.Timedelta(days=(4 - start.dayofweek) % 7 + 14)


def build_fixture(years=(2021, 2022), months=(3, 6, 9, 12), observations=4, seed=0):
    generator = np.random.default_rng(seed)
    settlement_rows = []
    level = 3800.0

    for year in years:
        for month in months:
            expiries = {
                horizon: third_friday(
                    year + (month + offset - 1) // 12,
                    (month + offset - 1) % 12 + 1,
                )
                for horizon, offset in ((30, 1), (60, 2), (90, 3))
            }

            rows = []
            for day in range(observations):
                data_date = third_friday(year, month) + pd.Timedelta(days=1 + 3 * day)
                level *= float(np.exp(generator.normal(0.004, 0.03)))
                volatility = float(np.clip(generator.normal(0.18, 0.04), 0.08, 0.6))
                rate = 0.02 + 0.01 * generator.standard_normal()

                for horizon, expiry in expiries.items():
                    days = (expiry - data_date).days
                    if days < 7:
                        continue
                    time = (days - AM_SETTLEMENT_DAY_ADJUSTMENT) / DAY_COUNT_DAYS_PER_YEAR
                    forward = level * float(np.exp((rate - 0.017) * time))

                    strikes = np.arange(
                        np.floor(0.55 * forward / 25) * 25,
                        1.35 * forward, 25.0,
                    )
                    # A downward-sloping smile in log-moneyness, floored so
                    # the far call wing stays positive.
                    smile = np.clip(
                        volatility - 0.45 * np.log(strikes / forward), 0.05, 1.5
                    )

                    for strike, local_vol in zip(strikes, smile):
                        for kind, price in (
                            ("call", black_call_price),
                            ("put", black_put_price),
                        ):
                            mid = float(price(forward, strike, local_vol, time, rate))
                            mid = max(mid + float(generator.normal(0, 0.05)), 0.05)
                            spread = max(0.05, 0.02 * mid)

                            rows.append({
                                "option_key": f"{data_date:%Y%m%d}{strike:.0f}{kind[0]}{horizon}",
                                "symbol": "SPX",
                                "data_date": data_date,
                                "expiration_date": expiry,
                                "settlement_date": expiry,
                                "time_to_expiry_years": time,
                                "put_call": kind,
                                "strike_price": float(strike),
                                "underlying_price": level,
                                "bid_price": max(mid - spread / 2, 0.05),
                                "ask_price": mid + spread / 2,
                                "mid_price": mid,
                                "bid_size": 10, "ask_size": 10,
                                "last_price": mid, "volume": 100,
                                "open_interest": 500,
                                "dte_calendar_days": days,
                                "is_standard_monthly": True,
                                "quote_is_valid": True,
                                "quote_is_two_sided": True,
                                "source_file": "synthetic.csv",
                                "target_dte_days": np.int16(horizon),
                            })

            frame = pd.DataFrame(rows)
            directory = SPX_PROCESSED_DIR / str(year)
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(
                directory / f"spx_options_{year}_{month:02d}.parquet", index=False
            )

    # Settlement values for every expiration the chains reference, plus a
    # margin either side.
    for year in range(min(years), max(years) + 2):
        for month in range(1, 13):
            settlement_rows.append({
                "expiration_year": year,
                "expiration_month": month,
                "settlement_date": third_friday(year, month),
                "set_value": float(level * np.exp(generator.normal(0, 0.05))),
            })

    SET_VALUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(settlement_rows).to_parquet(SET_VALUES_PATH, index=False)


if __name__ == "__main__":
    from implied_stock_distributions.config import CHAIN_CATALOG_PATH
    from implied_stock_distributions.data_access import build_chain_catalog

    build_fixture()
    catalog = build_chain_catalog()
    catalog.to_parquet(CHAIN_CATALOG_PATH, index=False)
    print(f"fixture: {len(catalog)} chains, catalogue at {CHAIN_CATALOG_PATH}")
