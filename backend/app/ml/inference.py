import torch
import pandas as pd
import numpy as np
import json
import os
import logging
from app.ml.models.action_transformer import ActionTransformer
from app.features.registry import FeatureRegistry

logger = logging.getLogger(__name__)

class ActionPredictor:
    def __init__(self, model_path='app/ml/models/action_transformer.pth', stats_path='dataset_stats.json', device=None):
        self.model_path = model_path
        self.stats_path = stats_path
        self.device = device if device else torch.device("cpu")
        self.model = None
        self.stats = None
        self.feature_cols = None
        self.seq_len = 64 # Should match training
        self.d_model = 64
        self.nhead = 4
        self.num_layers = 2
        
        self.load()

    def load(self):
        if not os.path.exists(self.stats_path):
            logger.error(f"Stats not found at {self.stats_path}")
            return
            
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)
            
        self.feature_cols = list(self.stats.keys())
        input_dim = len(self.feature_cols)
        
        self.model = ActionTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=3
        )
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.model = None
        else:
            logger.error(f"Model not found at {self.model_path}")

    def predict(self, df: pd.DataFrame) -> int:
        """
        Predict action from a DataFrame of history.
        df should have OHLCV columns.
        Returns: 0 (HOLD), 1 (SELL), 2 (BUY)
        """
        if self.model is None:
            return 0
            
        # Compute indicators
        # We assume df has enough history
        df = FeatureRegistry.compute_all(df)
        
        # Log returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['log_vol'] = np.log(df['volume'] / df['volume'].shift(1)).replace([np.inf, -np.inf], 0)
        df = df.fillna(0)
        
        # Normalize
        for col in self.feature_cols:
            if col in df.columns:
                mu = self.stats[col]['mean']
                sigma = self.stats[col]['std']
                if sigma != 0:
                    df[col] = (df[col] - mu) / sigma
            else:
                df[col] = 0.0
        
        # Extract sequence
        if len(df) < self.seq_len:
            logger.warning("Not enough history for prediction")
            return 0
            
        seq_data = df[self.feature_cols].iloc[-self.seq_len:].values
        x = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            action = torch.argmax(logits, dim=1).item()
            
        return action
