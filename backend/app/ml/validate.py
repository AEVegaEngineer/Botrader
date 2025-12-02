import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import backtrader as bt
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
from app.core.config import settings
from app.ml.dataset import load_data, SequenceDataset
from app.features.registry import FeatureRegistry
from app.backtest.strategies.ml_strategy import MLBasedStrategy
from app.backtest.engine import BacktestEngine
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom PandasData to include 'signal'
class PandasDataWithSignal(bt.feeds.PandasData):
    lines = ('signal',)
    params = (
        ('signal', -1), # -1 means autodetect based on column name 'signal'
    )

def validate_model(run_id=None, model_uri=None, threshold=0.6):
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # 1. Load Data
    df = load_data()
    if df is None:
        logger.error("No data found.")
        return

    feature_cols = FeatureRegistry.get_feature_names()
    target_col = 'dir_5m'
    
    # Ensure features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return

    # Split (same as training)
    # We want to validate on the TEST set (last 20%)
    train_size = int(0.8 * len(df))
    test_df = df.iloc[train_size:].copy()
    
    logger.info(f"Validating on {len(test_df)} samples.")

    # 2. Load Model & Generate Predictions
    predictions = None
    
    if run_id:
        model_uri = f"runs:/{run_id}/model"
    
    logger.info(f"Loading model from {model_uri}...")
    
    try:
        # Try loading as sklearn
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded sklearn model.")
        
        X_test = test_df[feature_cols]
        # predict_proba returns [prob_0, prob_1]
        predictions = model.predict_proba(X_test)[:, 1]
        
    except Exception as e_sklearn:
        logger.warning(f"Failed to load as sklearn model: {e_sklearn}")
        try:
            # Try loading as PyTorch
            model = mlflow.pytorch.load_model(model_uri)
            logger.info("Loaded PyTorch model.")
            
            # Prepare data for PyTorch
            seq_len = 64 # Must match training
            # Note: SequenceDataset expects the full DF to create sequences
            # We need to be careful to align indices. 
            # For simplicity here, we'll assume the model is not sequence-based OR 
            # we handle sequence generation. 
            # Actually, if it's a sequence model, we need sequences.
            # Let's check if it's a deep model based on some heuristic or metadata?
            # For now, let's assume if sklearn failed, it might be deep.
            
            # TODO: Handle sequence generation for deep models correctly in validation
            # This is complex because we need to slide windows over test_df
            # For now, let's fail gracefully if it's a deep model and requires sequences
            # Or implement a simple loop.
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            # Simple inference loop (assuming non-sequence for now or adapting)
            # If it's a sequence model, we need to reconstruct sequences.
            # Let's try to infer from input shape?
            # For this MVP, let's stick to sklearn models or simple feed-forward.
            # If sequence model, we'd need to use SequenceDataset on test_df
            
            # Let's try to use SequenceDataset
            ds = SequenceDataset(test_df, feature_cols, target_col, seq_len=64)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            
            all_preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_preds.extend(probs.cpu().numpy())
            
            # Align predictions with dataframe
            # SequenceDataset truncates the first seq_len-1 samples
            predictions = np.full(len(test_df), 0.5) # Default neutral
            predictions[64-1:] = all_preds
            
        except Exception as e_torch:
            logger.error(f"Failed to load model as PyTorch: {e_torch}")
            return

    if predictions is None:
        logger.error("Failed to generate predictions.")
        return

    # 3. Add signal to DataFrame
    test_df['signal'] = predictions
    # Ensure index is datetime
    if not isinstance(test_df.index, pd.DatetimeIndex):
        test_df.index = pd.to_datetime(test_df.index)

    # 4. Run Backtest
    cerebro = bt.Cerebro()
    
    # Add Strategy
    cerebro.addstrategy(MLBasedStrategy, threshold=threshold)
    
    # Add Data
    data = PandasDataWithSignal(dataname=test_df)
    cerebro.adddata(data)
    
    # Add Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Set Cash & Comm
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001) # 0.1%
    
    logger.info("Starting Backtest...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    
    strat = results[0]
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    
    logger.info("---------------------------------")
    logger.info(f"Validation Results for {model_uri}")
    logger.info("---------------------------------")
    logger.info(f"Initial Value: {initial_value:.2f}")
    logger.info(f"Final Value:   {final_value:.2f}")
    logger.info(f"Return:        {returns.get('rtot', 0.0)*100:.2f}%")
    logger.info(f"Sharpe Ratio:  {sharpe.get('sharperatio', 0.0)}")
    logger.info(f"Max Drawdown:  {drawdown.get('max', {}).get('drawdown', 0.0):.2f}%")
    logger.info("---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, help="MLflow Run ID")
    parser.add_argument("--model-uri", type=str, help="Model URI (e.g., runs:/.../model)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Trading threshold")
    
    args = parser.parse_args()
    
    if not args.run_id and not args.model_uri:
        print("Please provide --run-id or --model-uri")
        sys.exit(1)
        
    validate_model(run_id=args.run_id, model_uri=args.model_uri, threshold=args.threshold)
