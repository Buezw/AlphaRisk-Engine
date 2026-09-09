import sqlite3
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RiskDatabase:
    """Manages SQLite schema, constraints, and dynamic narrow-table persistence."""
    
    def __init__(self, db_path: str = "factor_risk.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        schema = """
        -- 1. Asset metadata table
        CREATE TABLE IF NOT EXISTS securities (
            symbol TEXT PRIMARY KEY,
            sec_name TEXT,
            sector TEXT,
            is_active INTEGER DEFAULT 1,
            ipo_date TEXT,
            delist_date TEXT
        );

        -- 2. Normalized daily quotes
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT,
            trade_date TEXT,
            adj_close REAL,
            daily_return REAL,
            PRIMARY KEY (symbol, trade_date),
            FOREIGN KEY (symbol) REFERENCES securities(symbol)
        );

        -- 3. Dynamic daily factor-return table (narrow table: factors can be extended horizontally without limit)
        CREATE TABLE IF NOT EXISTS factors_daily (
            trade_date TEXT,
            factor_name TEXT,
            factor_return REAL,
            PRIMARY KEY (trade_date, factor_name)
        );

        -- 4. Dynamic factor-exposure table (narrow table: any factor's beta stored vertically)
        CREATE TABLE IF NOT EXISTS factor_exposures (
            symbol TEXT,
            as_of_date TEXT,
            factor_name TEXT,
            beta REAL,
            PRIMARY KEY (symbol, as_of_date, factor_name),
            FOREIGN KEY (symbol) REFERENCES securities(symbol)
        );

        -- 5. Asset regression stats & idiosyncratic-risk table
        CREATE TABLE IF NOT EXISTS asset_regression_stats (
            symbol TEXT,
            as_of_date TEXT,
            idiosyncratic_var REAL,
            r_squared REAL,
            PRIMARY KEY (symbol, as_of_date),
            FOREIGN KEY (symbol) REFERENCES securities(symbol)
        );
        """
        cursor = self.conn.cursor()
        cursor.executescript(schema)
        self.conn.commit()
        logger.info("Dynamic narrow-table database schema initialized.")

    def upsert_securities(self, symbols: list[str]) -> None:
        data = [(s, s, "Equity", 1, "2020-01-01", None) for s in symbols]
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR IGNORE INTO securities (symbol, sec_name, sector, is_active, ipo_date, delist_date) VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        self.conn.commit()

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df_to_db = df[['symbol', 'trade_date', 'adj_close', 'daily_return']].copy()
        data_tuples = list(df_to_db.itertuples(index=False, name=None))
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO daily_prices (symbol, trade_date, adj_close, daily_return) VALUES (?, ?, ?, ?)",
            data_tuples
        )
        self.conn.commit()

    def upsert_factors_narrow(self, df_wide: pd.DataFrame) -> None:
        """Takes wide-format factor data, melts it flat automatically, and batch-writes it to the narrow table."""
        if df_wide.empty:
            return
        # Wide-to-narrow: trade_date, factor_name, factor_return
        df_narrow = df_wide.melt(id_vars=['trade_date'], var_name='factor_name', value_name='factor_return')
        data_tuples = list(df_narrow.itertuples(index=False, name=None))
        
        cursor = self.conn.cursor()
        cursor.executemany(
            "INSERT OR REPLACE INTO factors_daily (trade_date, factor_name, factor_return) VALUES (?, ?, ?)",
            data_tuples
        )
        self.conn.commit()
        logger.info(f"Upserted {len(data_tuples)} narrow records into factors_daily.")

    def upsert_exposures_and_stats(self, exposure_rows: list[dict], stats_rows: list[dict]) -> None:
        """Batch-writes to the narrow factor-exposure table and the idiosyncratic-variance stats table."""
        cursor = self.conn.cursor()
        if exposure_rows:
            exp_data = [(r['symbol'], r['as_of_date'], r['factor_name'], r['beta']) for r in exposure_rows]
            cursor.executemany(
                "INSERT OR REPLACE INTO factor_exposures (symbol, as_of_date, factor_name, beta) VALUES (?, ?, ?, ?)",
                exp_data
            )
        if stats_rows:
            stats_data = [(r['symbol'], r['as_of_date'], r['idiosyncratic_var'], r['r_squared']) for r in stats_rows]
            cursor.executemany(
                "INSERT OR REPLACE INTO asset_regression_stats (symbol, as_of_date, idiosyncratic_var, r_squared) VALUES (?, ?, ?, ?)",
                stats_data
            )
        self.conn.commit()
        logger.info(f"Persisted {len(exposure_rows)} betas and {len(stats_rows)} stats records.")

    def load_factors_wide(self) -> pd.DataFrame:
        """Reads from the vertical narrow table and pivots it back into a wide-format matrix."""
        df_narrow = pd.read_sql_query("SELECT trade_date, factor_name, factor_return FROM factors_daily", self.conn)
        if df_narrow.empty:
            return pd.DataFrame()
        return df_narrow.pivot(index='trade_date', columns='factor_name', values='factor_return').reset_index()

    def load_exposures_wide(self) -> pd.DataFrame:
        """Reads from the vertical narrow table, pivots it back into a wide exposure matrix, and inner-joins the regression stats."""
        df_betas = pd.read_sql_query("SELECT symbol, as_of_date, factor_name, beta FROM factor_exposures", self.conn)
        df_stats = pd.read_sql_query("SELECT symbol, as_of_date, idiosyncratic_var, r_squared FROM asset_regression_stats", self.conn)
        if df_betas.empty:
            return pd.DataFrame()
        pivoted = df_betas.pivot(index=['symbol', 'as_of_date'], columns='factor_name', values='beta').reset_index()
        return pd.merge(pivoted, df_stats, on=['symbol', 'as_of_date'], how='inner')