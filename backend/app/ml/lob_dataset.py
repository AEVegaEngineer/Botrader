import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class LOBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 100):
        self.seq_len = seq_len
        
        # Features: bid_p_1..5, bid_v_1..5, ask_p_1..5, ask_v_1..5 (20 features)
        feature_cols = [c for c in df.columns if 'bid_' in c or 'ask_' in c]
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df['label'].values.astype(np.int64)
        
        # Normalize features (Z-score per column)
        # In production, use saved scaler statistics!
        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0) + 1e-8
        self.features = (self.features - mean) / std
        
    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Sequence: [idx, idx + seq_len]
        # Target: [idx + seq_len]
        
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        
        # DeepLOB expects input shape: (1, seq_len, num_features) or similar
        # But usually CNNs over LOB treat 'levels' as spatial dimension.
        # For simplicity here, we treat it as (seq_len, features) and let the model handle it.
        
        return torch.tensor(x), torch.tensor(y)
