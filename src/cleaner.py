import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FinancialDataCleaner:
    """Production-grade data cleaning for equities and factors."""
    
    @staticmethod
    def clean_raw_quotes(df: pd.DataFrame, max_stale_days: int = 3, outlier_threshold: float = 0.5) -> pd.DataFrame:
        """
        Cleans raw financial quotes with strict constraints.
        
        Args:
            df: Raw DataFrame containing ['symbol', 'trade_date', 'adj_close']
            max_stale_days: Max consecutive days to forward fill missing quotes.
            outlier_threshold: Daily return threshold (e.g., 50%) to flag potential corrupt spikes.
        """
        if df.empty:
            return df

        required_cols = {'symbol', 'trade_date', 'adj_close'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"clean_raw_quotes: input is missing required column(s): {sorted(missing_cols)}")

        try:
            data = df.copy()
            data['trade_date'] = pd.to_datetime(data['trade_date']).dt.strftime('%Y-%m-%d')
            data = data.sort_values(by=['symbol', 'trade_date'])

            # 1. Filter out non-positive prices (defense against financial data glitches)
            invalid_prices = data['adj_close'] <= 0
            if invalid_prices.any():
                logger.warning(f"Dropping {invalid_prices.sum()} non-positive prices.")
                data = data[~invalid_prices]

            # 2. Cap the forward-fill for trading-halt gaps (so a long-suspended stock doesn't keep inheriting a stale price forever)
            data['adj_close'] = data.groupby('symbol')['adj_close'].ffill(limit=max_stale_days)
            data = data.dropna(subset=['adj_close'])

            # 3. Compute returns
            data['daily_return'] = data.groupby('symbol')['adj_close'].pct_change()

            # 4. Detect abnormal jumps (e.g. from an unadjusted stock split)
            # Clip per-symbol, not on a pooled quantile across the whole batch: a cross-sectional
            # quantile mixes a sleepy utility's returns with a volatile momentum name, so a single
            # global threshold either clips a volatile stock's genuine moves or fails to catch a
            # stable stock's real outlier, depending on which symbols happen to be in the batch.
            unusual_spikes = data['daily_return'].abs() > outlier_threshold
            if unusual_spikes.any():
                logger.warning(f"Detected {unusual_spikes.sum()} return outliers exceeding {outlier_threshold:.0%}.")
                grouped = data.groupby('symbol')['daily_return']
                q_low = grouped.transform(lambda s: s.quantile(0.01))
                q_high = grouped.transform(lambda s: s.quantile(0.99))
                data['daily_return'] = data['daily_return'].clip(lower=q_low, upper=q_high)

            return data.dropna(subset=['daily_return'])
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to clean raw quotes ({len(df)} rows): {e}")
            raise