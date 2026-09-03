import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Fetches market data from Yahoo Finance."""
    
    @staticmethod
    def fetch_daily_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Downloads historical data and formats it for the pipeline.
        """
        logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")
        df_list = []
        
        for ticker in tickers:
            try:
                # yfinance download
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data.empty:
                    continue
                    
                data = data.reset_index()
                data['symbol'] = ticker
                
                # Handle yfinance column name variations
                close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                
                temp_df = data[['Date', close_col, 'symbol']].copy()
                temp_df.columns = ['date', 'adj_close', 'symbol'] # Rename to match our DB/Processor
                df_list.append(temp_df)
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                
        if not df_list:
            return pd.DataFrame()
            
        return pd.concat(df_list, ignore_index=True)