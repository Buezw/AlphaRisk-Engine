"""
One-off/occasional fetch of long-horizon price history for strategy backtesting.

Unlike main.py (which keeps a rolling ~1-year window in `daily_prices` so risk/VaR calculations
never blend a stale volatility regime — see db_client.py's schema comments), this script fetches
several years of history and writes it into the separate, never-pruned `backtest_prices` table.
Run this before backtest_momentum.py, and re-run occasionally (e.g. every few months) to extend
the window as time passes — it does not need to run daily like main.py.

Usage:
    python fetch_backtest_history.py --years 5
"""
import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from src.data_fetcher import MarketDataFetcher
from src.cleaner import FinancialDataCleaner
from src.db_client import RiskDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BacktestHistoryFetch")


def main():
    parser = argparse.ArgumentParser(description="Fetch long-horizon price history for backtesting.")
    parser.add_argument("--years", type=int, default=5, help="How many years of history to fetch.")
    args = parser.parse_args()

    if args.years < 1:
        logger.error("--years must be >= 1")
        sys.exit(1)

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365 * args.years)).isoformat()
    logger.info(f"Fetching {args.years}-year backtest history: {start_date} to {end_date}")

    try:
        with RiskDatabase("factor_risk.db") as db:
            sec_df = pd.read_sql_query("SELECT symbol FROM securities", db.conn)
            if sec_df.empty:
                logger.error("No securities found in factor_risk.db — run `python main.py` first to populate the universe.")
                sys.exit(1)
            tickers = sec_df['symbol'].tolist()

            logger.info(f"Fetching {len(tickers)} tickers — this can take a while for a multi-year window.")
            raw_df = MarketDataFetcher.fetch_daily_prices_batch(tickers, start_date, end_date)
            if raw_df.empty:
                logger.error("No price data was fetched for any ticker — aborting.")
                sys.exit(1)

            clean_df = FinancialDataCleaner.clean_raw_quotes(raw_df)
            db.upsert_backtest_prices(clean_df)
            logger.info(f"Done. {clean_df['symbol'].nunique()} symbols, "
                        f"{clean_df['trade_date'].min()} to {clean_df['trade_date'].max()} now in backtest_prices.")
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
