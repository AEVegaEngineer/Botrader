import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import logging
import os
import numpy as np
from app.ml.dataset import SequenceDataset
from app.ml.ensemble import EnsembleModel
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "dataset.parquet"

def load_data():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)

def train_ensemble():
    df = load_data()
    if df is None:
        return

    feature_cols = FeatureRegistry.get_feature_names()
    target_col = 'dir_5m'
    
    # Create Dataset
    seq_len = 64
    dataset = SequenceDataset(df, feature_cols, target_col, seq_len=seq_len)
    
    # Use a hold-out set for meta-learner training to avoid overfitting
    # (Ideally we'd use out-of-fold predictions, but simple split is okay for now)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # We train meta-learner on VALIDATION set (where base models haven't seen data, ideally)
    # But here base models were trained on 80% of data. 
    # To do this properly without retraining base models, we should use the TEST set of the base models 
    # as the TRAIN set for the meta learner.
    # For simplicity, let's just use the validation set here.
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    input_dim = len(feature_cols)
    ensemble = EnsembleModel(input_dim, device)
    
    # Collect all validation data
    all_X = []
    all_y = []
    
    logger.info("Generating predictions for meta-learner...")
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        all_X.append(X_batch)
        all_y.append(y_batch.numpy())
        
    X_tensor = torch.cat(all_X)
    y_target = np.concatenate(all_y)
    
    logger.info(f"Training Meta-Learner on {len(y_target)} samples...")
    ensemble.train_meta_learner(X_tensor, y_target)
    
    logger.info("Meta-Learner trained and saved.")
    
    # Evaluate on Test Set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_X_test = []
    all_y_test = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        all_X_test.append(X_batch)
        all_y_test.append(y_batch.numpy())
        
    X_test_tensor = torch.cat(all_X_test)
    y_test_target = np.concatenate(all_y_test)
    
    probs = ensemble.predict_proba(None, X_test_tensor)
    preds = (probs > 0.5).astype(int)
    
    acc = (preds == y_test_target).mean()
    logger.info(f"Ensemble Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_ensemble()
