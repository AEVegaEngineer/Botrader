import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
DATASET_PATH = "dataset.parquet"

def load_data():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str, seq_len: int = 64, target_dtype=torch.long):
        self.seq_len = seq_len
        
        # Convert to float32 for PyTorch
        self.features = df[feature_cols].values.astype(np.float32)
        
        if target_dtype == torch.long:
            self.targets = df[target_col].values.astype(np.int64)
        else:
            self.targets = df[target_col].values.astype(np.float32)
        
    def __len__(self):
        # We need a sequence of length seq_len.
        # If N=100, seq_len=10.
        # Last index is 99.
        # We need window ending at 99: [90, ..., 99].
        # Start index is 90.
        # So idx goes from 0 to 90.
        # Count is 91.
        # N - seq_len + 1.
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        # Sequence: [idx, idx + seq_len] (exclusive end) -> indices [idx, ..., idx + seq_len - 1]
        # Target: [idx + seq_len - 1] (the label corresponding to the last step of the sequence)
        
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        
        return torch.tensor(x), torch.tensor(y)
