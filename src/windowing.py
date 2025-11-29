"""Windowing and labeling utilities for sequence data."""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import LabelingConfig


class SequenceWindower:
    """Create fixed-length windows from time series OHLCV data."""

    def __init__(self, sequence_length: int, future_bars: int = 5):
        """Initialize windower.

        Args:
            sequence_length: Number of candles per window
            future_bars: Lookahead window for labels
        """
        self.sequence_length = sequence_length
        self.future_bars = future_bars

    def create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows from OHLCV data.

        Args:
            df: DataFrame with normalized OHLC(V) data

        Returns:
            Tuple of (windows, indices) where:
            - windows: shape (num_windows, sequence_length, 5) for OHLCV
            - indices: original row indices for each window
        """
        data = df[['open', 'high', 'low', 'close', 'volume']].values
        num_samples = len(data) - self.sequence_length - self.future_bars + 1

        if num_samples <= 0:
            raise ValueError(
                f"Not enough data. Need {self.sequence_length + self.future_bars} "
                f"bars, got {len(data)}"
            )

        windows = np.zeros((num_samples, self.sequence_length, 5))
        indices = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            windows[i] = data[i:i + self.sequence_length]
            indices[i] = i  # Index of first candle in window

        return windows, indices


class LabelGenerator:
    """Generate classification labels based on future price movement."""

    def __init__(self, config: LabelingConfig, sequence_length: int = 128):
        """Initialize label generator.

        Args:
            config: Labeling configuration
            sequence_length: Number of candles per window
        """
        self.config = config
        self.sequence_length = sequence_length

    def generate_labels(self, df: pd.DataFrame, windows: np.ndarray,
                       indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate up/down labels based on future close price.

        Args:
            df: Original DataFrame with price data
            windows: Window array from SequenceWindower
            indices: Window indices from SequenceWindower

        Returns:
            Tuple of (labels, valid_mask) where:
            - labels: binary labels (0=down, 1=up)
            - valid_mask: boolean mask indicating non-neutral samples
        """
        num_windows = len(windows)
        labels = np.zeros(num_windows, dtype=int)
        valid_mask = np.ones(num_windows, dtype=bool)

        close_prices = df['close'].values

        for i, idx in enumerate(indices):
            # Reference: close price of last candle in window
            current_close = close_prices[idx + self.sequence_length - 1]

            # Target: close price after future_bars
            future_idx = idx + self.sequence_length + self.config.future_bars - 1

            if future_idx >= len(close_prices):
                valid_mask[i] = False
                continue

            future_close = close_prices[future_idx]

            # Calculate percentage change
            pct_change = ((future_close - current_close) / current_close) * 100

            # Label based on threshold
            if abs(pct_change) < self.config.threshold_pct:
                if self.config.remove_neutral:
                    valid_mask[i] = False
            else:
                labels[i] = 1 if pct_change > 0 else 0

        return labels, valid_mask


def split_dataset(
    windows: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    test_split: float = 0.2,
    val_split: float = 0.1,
    random_seed: int = 42
) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split data into train/val/test sets.

    Args:
        windows: Input windows
        labels: Output labels
        valid_mask: Mask of valid samples
        test_split: Test set fraction
        val_split: Validation set fraction (of remaining after test)
        random_seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing (windows, labels) tuples
    """
    # Filter valid samples
    valid_windows = windows[valid_mask]
    valid_labels = labels[valid_mask]

    # Simple random split (faster than stratified for large datasets)
    np.random.seed(random_seed)
    n = len(valid_windows)
    indices = np.random.permutation(n)

    test_size = int(n * test_split)
    val_size = int(n * (1 - test_split) * val_split)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    return {
        'train': (valid_windows[train_idx], valid_labels[train_idx]),
        'val': (valid_windows[val_idx], valid_labels[val_idx]),
        'test': (valid_windows[test_idx], valid_labels[test_idx])
    }
