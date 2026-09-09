import logging
import pandas as pd
import numpy as np
from src.data_fetcher import MarketDataFetcher
from src.cleaner import FinancialDataCleaner
from src.db_client import RiskDatabase
from src.factor_model import FactorExposureEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskPipeline")

def run_pipeline():
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    # 1. Dynamically load the S&P 500 constituent list
    sp500_df = MarketDataFetcher.fetch_sp500_constituents()
    logger.info(f"Loaded {len(sp500_df)} constituents from S&P 500.")

    # The Wikipedia constituent list doesn't include the SPY ETF itself, but the app requires SPY
    # in the database as its benchmark anchor for active-risk calculations — add it explicitly so
    # a fresh run of this pipeline always produces a working benchmark row.
    if 'SPY' not in sp500_df['symbol'].values:
        spy_row = pd.DataFrame([{'symbol': 'SPY', 'sec_name': 'SPDR S&P 500 ETF Trust', 'sector': 'Benchmark'}])
        sp500_df = pd.concat([sp500_df, spy_row], ignore_index=True)

    # 2. Write asset metadata to the database
    db = RiskDatabase("factor_risk.db")
    cursor = db.conn.cursor()
    sec_data = [
        (r['symbol'], r['sec_name'], r['sector'], 1, "2020-01-01", None)
        for _, r in sp500_df.iterrows()
    ]
    cursor.executemany(
        """INSERT OR REPLACE INTO securities 
           (symbol, sec_name, sector, is_active, ipo_date, delist_date) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        sec_data
    )
    db.conn.commit()

    # 3. Batch-fetch daily quotes, clean them, and persist
    # Tip: for a quick smoke test, slice this down to e.g. sp500_df['symbol'].tolist()[:80] —
    # fetching the full universe (500+ tickers) takes a while and is more exposed to yfinance
    # rate limits, since fetch_daily_prices_batch has to chunk it into many separate requests.
    target_tickers = sp500_df['symbol'].tolist()
    logger.info(f"Fetching the full universe: {len(target_tickers)} tickers.")
    raw_df = MarketDataFetcher.fetch_daily_prices_batch(target_tickers, start_date, end_date)
    clean_df = FinancialDataCleaner.clean_raw_quotes(raw_df)
    db.upsert_prices(clean_df)

    # 4. Fetch factor data and persist
    factors_wide_df = MarketDataFetcher.fetch_multi_factor_series(start_date, end_date)
    db.upsert_factors_narrow(factors_wide_df)

    # 5. Batch-solve the 6-factor OLS regression via linear algebra
    logger.info("Computing multi-factor regressions across active universe...")
    merged = pd.merge(clean_df, factors_wide_df, on='trade_date')
    factor_cols = [c for c in factors_wide_df.columns if c not in ['trade_date', 'RF']]
    
    exposure_records = []
    stats_records = []
    
    for ticker in target_tickers:
        sub = merged[merged['symbol'] == ticker].sort_values('trade_date')
        if len(sub) < 50:
            continue
            
        y = sub['daily_return'].values
        y_excess = y - sub['RF'].values
        X = np.column_stack([np.ones(len(sub))] + [sub[col].values for col in factor_cols])
        
        try:
            betas, spec_var, r2 = FactorExposureEngine.solve_ols(X, y_excess)
            as_of_date = sub['trade_date'].max()
            
            for idx, factor_name in enumerate(factor_cols):
                exposure_records.append({
                    'symbol': ticker,
                    'as_of_date': as_of_date,
                    'factor_name': factor_name,
                    'beta': betas[idx + 1]
                })
            stats_records.append({
                'symbol': ticker,
                'as_of_date': as_of_date,
                'idiosyncratic_var': spec_var,
                'r_squared': r2
            })
        except Exception as e:
            logger.warning(f"Regression failed for {ticker}: {e}")

    db.upsert_exposures_and_stats(exposure_records, stats_records)
    logger.info(f"Multi-factor exposures calculated and persisted for {len(stats_records)} assets.")

if __name__ == "__main__":
    run_pipeline()