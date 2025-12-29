import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import logging
import os
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix
from app.ml.dataset import SequenceDataset
from app.ml.models.action_transformer import ActionTransformer
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
    
    # MLflow Setup
    # Use mlflow service name for Docker compatibility
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Botrader_Action_Transformer")
    
    df = load_data()
    if df is None:
        return

    # Features: Indicators + Log Returns
    feature_cols = FeatureRegistry.get_feature_names() + ['log_ret', 'log_vol']
    target_col = 'label'
    
    # Check columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns: {missing_cols}")
        return
    if target_col not in df.columns:
        logger.error(f"Missing target column: {target_col}")
        return

    # Hyperparams
    SEQ_LEN = 64
    BATCH_SIZE = 64
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1
    LR = 0.001
    NUM_EPOCHS = 10
    
    # Create Dataset
    dataset = SequenceDataset(df, feature_cols, target_col, seq_len=SEQ_LEN, target_dtype=torch.long)
    
    # Chronological Split
    total_len = len(dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size
    
    # Indices are 0 to total_len-1
    # Train: 0 to train_size
    # Val: train_size to train_size + val_size
    # Test: rest
    
    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, total_len))
    
    # Shuffle train is okay for windowed dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Compute Class Weights
    # We need to look at the targets in the training set
    # Since Subset doesn't expose targets directly easily without iterating, let's use the df
    # We approximate by taking the corresponding slice of the dataframe
    # Note: dataset indices map to df indices roughly as idx -> idx (if we ignore the start offset logic for a moment)
    # SequenceDataset: idx 0 corresponds to window [0..seq_len]. Target is at seq_len-1.
    # So targets are from seq_len-1 to ...
    
    train_targets = df[target_col].iloc[SEQ_LEN-1 : train_size + SEQ_LEN - 1]
    class_counts = train_targets.value_counts().sort_index()
    total_samples = len(train_targets)
    
    class_weights = []
    # Classes: 0 (HOLD), 1 (SELL), 2 (BUY)
    for i in range(3):
        count = class_counts.get(i, 0)
        if count > 0:
            weight = total_samples / (3 * count)
        else:
            weight = 1.0
        class_weights.append(weight)
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights: {class_weights}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = ActionTransformer(
        input_dim=len(feature_cols),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_classes=3,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    best_val_loss = float('inf')
    save_name = "action_transformer.pth"
    
    with mlflow.start_run():
        mlflow.log_params({
            "seq_len": SEQ_LEN,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "lr": LR,
            "batch_size": BATCH_SIZE
        })
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
            train_loss /= len(train_loader)
            train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
                    
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, save_name))
                mlflow.pytorch.log_model(model, "model")
                logger.info("Saved best model")
        
        # Test Evaluation
        logger.info("Evaluating on Test Set...")
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, save_name)))
        model.eval()
        
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(y_batch.numpy())
                
        report = classification_report(test_targets, test_preds, target_names=['HOLD', 'SELL', 'BUY'])
        logger.info(f"Test Report:\n{report}")
        
        # Log artifacts
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

if __name__ == "__main__":
    train_transformer()
