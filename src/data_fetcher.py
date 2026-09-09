import io
import requests
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetches market quotes, constituent universes, and official multi-factor series."""

    @staticmethod
    def fetch_sp500_constituents() -> pd.DataFrame:
        """Dynamically scrapes the latest S&P 500 constituent metadata from Wikipedia (with a spoofed request header and a fallback on failure)."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        try:
            logger.info("Fetching S&P 500 constituent list from Wikipedia...")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Wrap the HTML text in StringIO to avoid the 403 / deprecation warning that comes
            # from passing a raw string or URL directly
            tables = pd.read_html(io.StringIO(response.text))
            df = tables[0][['Symbol', 'Security', 'GICS Sector']].copy()
            df.columns = ['symbol', 'sec_name', 'sector']
            # Handle special tickers (e.g. BRK.B -> BRK-B)
            df['symbol'] = df['symbol'].str.replace('.', '-', regex=False)
            return df
        except Exception as e:
            logger.warning(f"Wikipedia request failed ({e}). Falling back to default top blue chips.")
            # Fallback asset pool
            fallback_data = [
                {"symbol": "AAPL", "sec_name": "Apple Inc.", "sector": "Information Technology"},
                {"symbol": "MSFT", "sec_name": "Microsoft Corp.", "sector": "Information Technology"},
                {"symbol": "NVDA", "sec_name": "NVIDIA Corp.", "sector": "Information Technology"},
                {"symbol": "AMZN", "sec_name": "Amazon.com Inc.", "sector": "Consumer Discretionary"},
                {"symbol": "GOOGL", "sec_name": "Alphabet Inc.", "sector": "Communication Services"},
                {"symbol": "META", "sec_name": "Meta Platforms", "sector": "Communication Services"},
                {"symbol": "JPM",  "sec_name": "JPMorgan Chase", "sector": "Financials"},
                {"symbol": "BAC",  "sec_name": "Bank of America", "sector": "Financials"},
                {"symbol": "XOM",  "sec_name": "Exxon Mobil", "sector": "Energy"},
                {"symbol": "CVX",  "sec_name": "Chevron Corp.", "sector": "Energy"},
                {"symbol": "JNJ",  "sec_name": "Johnson & Johnson", "sector": "Health Care"},
                {"symbol": "PG",   "sec_name": "Procter & Gamble", "sector": "Consumer Staples"}
            ]
            return pd.DataFrame(fallback_data)

    @staticmethod
    def fetch_daily_prices_batch(tickers: list[str], start_date: str, end_date: str, chunk_size: int = 50) -> pd.DataFrame:
        """Downloads quotes in chunks to avoid overloading or timing out a single request."""
        logger.info(f"Batch fetching daily prices for {len(tickers)} assets...")
        dfs = []
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i:i + chunk_size]
            try:
                data = yf.download(batch, start=start_date, end=end_date, progress=False, group_by='ticker')
                if data.empty:
                    continue
                for ticker in batch:
                    try:
                        ticker_data = data[ticker].dropna(how='all').reset_index()
                        if ticker_data.empty:
                            continue
                        close_col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
                        temp_df = ticker_data[['Date', close_col]].copy()
                        temp_df.columns = ['trade_date', 'adj_close']
                        temp_df['symbol'] = ticker
                        dfs.append(temp_df)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Batch {i // chunk_size} failed: {e}")
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def fetch_multi_factor_series(start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches the Fama-French 5 factors and the momentum factor, and merges them."""
        try:
            logger.info(f"Fetching 5-Factor & Momentum series from {start_date} to {end_date}")
            ff5_dict = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start=start_date, end=end_date)
            df_ff5 = ff5_dict[0] / 100.0
            df_ff5 = df_ff5.reset_index()
            df_ff5.columns = ['trade_date', 'MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            df_ff5['trade_date'] = df_ff5['trade_date'].dt.strftime('%Y-%m-%d')

            mom_dict = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start=start_date, end=end_date)
            df_mom = mom_dict[0] / 100.0
            df_mom = df_mom.reset_index()
            df_mom.columns = ['trade_date', 'MOM']
            df_mom['MOM'] = pd.to_numeric(df_mom['MOM'], errors='coerce')
            df_mom['trade_date'] = df_mom['trade_date'].dt.strftime('%Y-%m-%d')

            return pd.merge(df_ff5, df_mom, on='trade_date', how='inner').dropna()
        except Exception as e:
            logger.error(f"Failed to fetch factor series: {e}")
            raise