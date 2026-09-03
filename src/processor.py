import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data cleaning, normalization, and outlier removal."""
    
    @staticmethod
    def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans raw price data.
        
        Args:
            df (pd.DataFrame): Raw price dataframe with columns ['symbol', 'date', 'adj_close'].
            
        Returns:
            pd.DataFrame: Cleaned data with calculated daily returns.
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to cleaner.")
            return df

        df = df.sort_values(by=['symbol', 'date'])
        
        # Forward fill missing prices (max 3 days to avoid stale data)
        df['adj_close'] = df.groupby('symbol')['adj_close'].ffill(limit=3)
        df = df.dropna(subset=['adj_close'])
        
        # Calculate daily returns
        df['daily_return'] = df.groupby('symbol')['adj_close'].pct_change()
        
        # Winsorize outliers at 1st and 99th percentiles (Common quantitative practice)
        lower_bound = df['daily_return'].quantile(0.01)
        upper_bound = df['daily_return'].quantile(0.99)
        df['daily_return'] = np.clip(df['daily_return'], lower_bound, upper_bound)
        
        return df.dropna()