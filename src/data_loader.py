"""Data loading utilities for OHLC database."""
import sqlite3
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DataConfig


class OHLCDataLoader:
    """Load and manage OHLC data from SQLite database."""

    def __init__(self, db_path: Path, config: DataConfig):
        """Initialize data loader.

        Args:
            db_path: Path to SQLite database
            config: Data configuration
        """
        self.db_path = Path(db_path)
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        """Establish database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)

    def _disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_ohlcv_data(self, symbol: str, start_timestamp: Optional[int] = None,
                       end_timestamp: Optional[int] = None) -> pd.DataFrame:
        """Load OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            start_timestamp: Optional start time (Unix timestamp)
            end_timestamp: Optional end time (Unix timestamp)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self._connect()

        query = """
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM ohlc_data
            WHERE symbol = ? AND timeframe = ?
        """
        params = [symbol, self.config.timeframe]

        if start_timestamp:
            query += " AND timestamp >= ?"
            params.append(start_timestamp)
        if end_timestamp:
            query += " AND timestamp <= ?"
            params.append(end_timestamp)

        query += " ORDER BY timestamp ASC"

        df = pd.read_sql_query(query, self.conn, params=params)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        return df

    def get_all_symbols_data(self) -> dict[str, pd.DataFrame]:
        """Load data for all configured symbols.

        Returns:
            Dictionary mapping symbol to OHLCV DataFrame
        """
        data = {}
        for symbol in tqdm(self.config.symbols, desc="Loading symbol data"):
            try:
                data[symbol] = self.get_ohlcv_data(symbol)
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        return data

    def normalize_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Normalize OHLCV data using min-max scaling.

        Args:
            df: OHLCV DataFrame

        Returns:
            Tuple of (normalized_df, scaling_params)
        """
        df_norm = df.copy()
        scaling_params = {}

        for col in ['open', 'high', 'low', 'close']:
            min_val = df[col].min()
            max_val = df[col].max()
            scaling_params[col] = {'min': min_val, 'max': max_val}

            if max_val > min_val:
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.0

        # Normalize volume using log scale
        volume_log = np.log1p(df['volume'])
        min_vol = volume_log.min()
        max_vol = volume_log.max()
        scaling_params['volume'] = {'min': min_vol, 'max': max_vol}

        if max_vol > min_vol:
            df_norm['volume'] = (volume_log - min_vol) / (max_vol - min_vol)
        else:
            df_norm['volume'] = 0.0

        return df_norm, scaling_params

    def get_data_summary(self) -> dict:
        """Get summary statistics about available data.

        Returns:
            Dictionary with data coverage info
        """
        self._connect()

        query = """
            SELECT symbol, timeframe, COUNT(*) as count,
                   MIN(timestamp) as earliest, MAX(timestamp) as latest
            FROM ohlc_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """

        df = pd.read_sql_query(query, self.conn)
        return df.to_dict('records')

    def __enter__(self):
        """Context manager entry."""
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._disconnect()
