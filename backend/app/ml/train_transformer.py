import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import logging
import os
from app.ml.dataset import SequenceDataset
from app.ml.models.transformer import TimeSeriesTransformer
from app.ml.loss import QuantileLoss
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"
DATASET_PATH = "dataset.parquet"

def load_data():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)

def train_transformer():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data()
    if df is None:
        return

    feature_cols = FeatureRegistry.get_feature_names()
    # For regression, we predict the return itself, e.g., 'ret_5m'
    # Ensure this column exists in dataset. If not, we might need to rely on 'close' price derivation or ensure build_dataset adds it.
    # Checking build_dataset.py: it adds 'ret_{h}m'.
    target_col = 'ret_5m' 
    
    # Check columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return
    if target_col not in df.columns:
        logger.error(f"Missing target column: {target_col}")
        return

    # Create Dataset
    seq_len = 64
    # Use float targets for regression
    dataset = SequenceDataset(df, feature_cols, target_col, seq_len=seq_len, target_dtype=torch.float)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model Hyperparams
    input_dim = len(feature_cols)
    d_model = 64
    nhead = 4
    num_layers = 2
    quantiles = [0.1, 0.5, 0.9]
    num_quantiles = len(quantiles)
    
    model = TimeSeriesTransformer(input_dim, d_model, nhead, num_layers, num_quantiles).to(device)
    criterion = QuantileLoss(quantiles)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    num_epochs = 10
    best_val_loss = float('inf')
    save_name = "transformer_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, save_name))
            
    logger.info(f"Best Val Loss: {best_val_loss:.6f}. Model saved to {save_name}")

if __name__ == "__main__":
    train_transformer()
