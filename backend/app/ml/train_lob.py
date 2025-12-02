import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import logging
import os
from app.ml.lob_dataset import LOBDataset
from app.ml.models.deeplob import DeepLOB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"
DATASET_PATH = "lob_dataset.parquet"

def load_data():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_lob_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)

def train_deeplob():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data()
    if df is None:
        return

    # Create Dataset
    seq_len = 100
    dataset = LOBDataset(df, seq_len=seq_len)
    
    if len(dataset) < 100:
        logger.warning("Dataset too small for training.")
        return
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model
    num_features = 20
    num_classes = 3
    model = DeepLOB(seq_len, num_features, num_classes).to(device)
    
    # Weighted Loss (Simple heuristic for now, assuming class 1 (Flat) is dominant)
    # Ideally calculate from data
    weights = torch.tensor([1.0, 0.5, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    num_epochs = 10
    best_val_loss = float('inf')
    save_name = "deeplob_model.pth"
    
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100 * correct / (total + 1e-8)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, save_name))
            
    logger.info(f"Best Val Loss: {best_val_loss:.4f}. Model saved to {save_name}")

if __name__ == "__main__":
    train_deeplob()
