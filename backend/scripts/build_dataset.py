import asyncio
import pandas as pd
import logging
import os
import json
import numpy as np
from sqlalchemy import text
from app.core.database import engine
from app.ml.labeling import compute_labels, get_label_distribution
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_data(symbol: str, start_date: str = "2024-01-01"):
    from datetime import datetime, timezone
    
    # Parse start_date to datetime object
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    query = text(f"""
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol AND time >= :start_date
        ORDER BY time ASC
    """)
    
    async with engine.connect() as conn:
        result = await conn.execute(query, {"symbol": symbol, "start_date": start_dt})
        rows = result.fetchall()
        
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df

async def main():
    symbol = "BTCUSDT"
    logger.info(f"Fetching data for {symbol}...")
    
    # Fetch more data to ensure we have enough for training
    df = await fetch_data(symbol, start_date="2023-01-01")
    
    if df.empty:
        logger.warning("No data found in database.")
        return

    logger.info(f"Fetched {len(df)} rows. Computing indicators...")
    
    # 1. Compute Indicators
    df = FeatureRegistry.compute_all(df)
    
    # 2. Feature Engineering for Stationarity
    # Convert raw prices to log returns for better ML performance
    # Use numpy directly for log to avoid pandas issues
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['log_vol'] = np.log(df['volume'] / df['volume'].shift(1)).replace([np.inf, -np.inf], 0)
    
    # Add normalized OHLCV features
    # Normalize prices relative to close price to capture intraday patterns
    # This makes features more stationary while preserving range information
    df['norm_open'] = (df['open'] - df['close']) / df['close']
    df['norm_high'] = (df['high'] - df['close']) / df['close']
    df['norm_low'] = (df['low'] - df['close']) / df['close']
    # Note: norm_close would be same as log_ret, so we skip it
    df['norm_volume'] = np.log(df['volume'] + 1)  # Log transform for volume (add 1 to avoid log(0))
    
    # Fill NaNs from shifting
    df = df.fillna(0) 
    
    # Define feature columns to use (Indicators + transformed OHLCV + normalized OHLCV)
    feature_cols = FeatureRegistry.get_feature_names() + ['log_ret', 'log_vol'] + ['norm_open', 'norm_high', 'norm_low', 'norm_volume']
    
    # 3. Labeling
    logger.info("Computing labels...")
    # Configurable parameters
    HORIZON = 15
    TH_UP = 0.002  # 0.2%
    TH_DOWN = -0.002 # -0.2%
    COST = 0.0004 # 0.04% taker fee approx
    
    df_labeled = compute_labels(df, horizon_candles=HORIZON, th_up=TH_UP, th_down=TH_DOWN, transaction_cost=COST)
    
    logger.info(f"Label distribution: {get_label_distribution(df_labeled)}")
    
    # 4. Normalization Stats
    logger.info("Computing normalization stats...")
    stats = {}
    for col in feature_cols:
        if col in df_labeled.columns:
            stats[col] = {
                'mean': float(df_labeled[col].mean()),
                'std': float(df_labeled[col].std())
            }
            # Apply normalization
            if stats[col]['std'] != 0:
                df_labeled[col] = (df_labeled[col] - stats[col]['mean']) / stats[col]['std']
        else:
            logger.warning(f"Feature {col} not found in dataframe.")
            
    # Drop rows with NaNs (again, just in case)
    df_final = df_labeled.dropna()
    
    # 5. Save
    output_file = "dataset.parquet"
    stats_file = "dataset_stats.json"
    
    df_final.to_parquet(output_file)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Dataset saved to {output_file}. Shape: {df_final.shape}")
    logger.info(f"Stats saved to {stats_file}")

if __name__ == "__main__":
    asyncio.run(main())
