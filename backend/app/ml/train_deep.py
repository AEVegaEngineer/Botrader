import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import logging
import os
from app.ml.dataset import SequenceDataset, load_data
from app.ml.models.deep import LSTMClassifier, CNNClassifier
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "app/ml/models"
DATASET_PATH = "dataset.parquet"

import mlflow
import mlflow.pytorch

# ... imports ...

def train_deep_models():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Deep_Models")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data()
    if df is None:
        return

    feature_cols = FeatureRegistry.get_feature_names()
    target_col = 'dir_5m' # 1 for up, 0 for down
    
    # Check columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return

    # Create Dataset
    seq_len = 64
    dataset = SequenceDataset(df, feature_cols, target_col, seq_len=seq_len)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Train LSTM
    with mlflow.start_run(run_name="LSTM"):
        logger.info("Training LSTM...")
        input_dim = len(feature_cols)
        hidden_dim = 64
        num_layers = 2
        output_dim = 2 # Binary classification (0, 1)
        
        model = LSTMClassifier(input_dim, hidden_dim, num_layers, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("hidden_dim", hidden_dim)
        mlflow.log_param("num_layers", num_layers)
        
        train_loop(model, train_loader, val_loader, criterion, optimizer, device, "lstm_model.pth")
        
        mlflow.pytorch.log_model(model, "model")
    
    # 2. Train CNN
    with mlflow.start_run(run_name="CNN"):
        logger.info("Training CNN...")
        num_filters = 64
        
        model = CNNClassifier(input_dim, num_filters, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        mlflow.log_param("model_type", "CNN")
        mlflow.log_param("num_filters", num_filters)
        
        train_loop(model, train_loader, val_loader, criterion, optimizer, device, "cnn_model.pth")
        
        mlflow.pytorch.log_model(model, "model")

def train_loop(model, train_loader, val_loader, criterion, optimizer, device, save_name):
    num_epochs = 10
    best_val_loss = float('inf')
    
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
        val_acc = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, save_name))
            
    logger.info(f"Best Val Loss: {best_val_loss:.4f}. Model saved to {save_name}")

if __name__ == "__main__":
    train_deep_models()
