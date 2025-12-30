import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
DATASET_PATH = "dataset.parquet"

def load_data():
    """
    Load preprocessed dataset with features and indicators.
    
    Returns:
        DataFrame with OHLCV data and computed features, or None if file not found
    """
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)
