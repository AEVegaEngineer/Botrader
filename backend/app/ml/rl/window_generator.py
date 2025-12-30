import numpy as np
import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class WindowGenerator:
    """
    Generates overlapping trading windows from historical candlestick data.
    Each window represents a complete trading episode.
    """
    
    def __init__(self, window_size: int = 240, step_size: int = 60):
        """
        Args:
            window_size: Number of candles per window (default: 240 = 4 hours at 1-minute intervals)
            step_size: Number of candles to step between windows (default: 60 = 25% overlap)
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def generate_windows(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate list of (start_idx, end_idx) tuples for overlapping windows.
        
        Args:
            df: DataFrame with candlestick data
            
        Returns:
            List of (start_idx, end_idx) tuples representing window boundaries
        """
        if len(df) < self.window_size:
            logger.warning(f"Data length ({len(df)}) is less than window size ({self.window_size})")
            return []
        
        windows = []
        start_idx = 0
        
        while start_idx + self.window_size <= len(df):
            end_idx = start_idx + self.window_size
            windows.append((start_idx, end_idx))
            start_idx += self.step_size
        
        logger.info(f"Generated {len(windows)} overlapping windows from {len(df)} candles")
        return windows
    
    def get_window_data(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Extract a window of data from the DataFrame.
        
        Args:
            df: Full DataFrame
            start_idx: Start index (inclusive)
            end_idx: End index (exclusive)
            
        Returns:
            DataFrame slice for the specified window
        """
        return df.iloc[start_idx:end_idx].copy()
    
    def split_windows(self, windows: List[Tuple[int, int]], train_pct: float = 0.7, 
                     val_pct: float = 0.15) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Split windows into train/validation/test sets chronologically.
        
        Args:
            windows: List of (start_idx, end_idx) tuples
            train_pct: Percentage of windows for training
            val_pct: Percentage of windows for validation
            
        Returns:
            Tuple of (train_windows, val_windows, test_windows)
        """
        total = len(windows)
        train_end = int(total * train_pct)
        val_end = train_end + int(total * val_pct)
        
        train_windows = windows[:train_end]
        val_windows = windows[train_end:val_end]
        test_windows = windows[val_end:]
        
        logger.info(f"Split windows: Train={len(train_windows)}, Val={len(val_windows)}, Test={len(test_windows)}")
        return train_windows, val_windows, test_windows

