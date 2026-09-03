import sqlite3
import logging
from typing import List, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDBClient:
    """Handles database connections and SQL operations for market data."""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create normalized tables for prices and factors."""
        query = """
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT,
            trade_date DATE,
            adj_close REAL,
            daily_return REAL,
            PRIMARY KEY (symbol, trade_date)
        );
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            self.conn.commit()
            logger.info("Database tables initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")

    def insert_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Bulk insert pandas DataFrame to SQL using best practices."""
        try:
            df.to_sql(table_name, self.conn, if_exists='append', index=False)
            logger.info(f"Inserted {len(df)} rows into {table_name}.")
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")