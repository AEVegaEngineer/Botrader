import pandas as pd
import numpy as np
import logging
import os
from sqlalchemy import text
from app.core.database import engine
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "lob_dataset.parquet"

async def fetch_lob_data(symbol="BTCUSDT", limit=100000):
    query = text("""
        SELECT time, bids, asks
        FROM lob_snapshots
        WHERE symbol = :symbol
        ORDER BY time ASC
        LIMIT :limit
    """)
    
    async with engine.connect() as conn:
        result = await conn.execute(query, {"symbol": symbol, "limit": limit})
        rows = result.fetchall()
        
    return rows

def process_lob_data(rows):
    data = []
    for row in rows:
        time = row.time
        bids = row.bids # List of [price, qty]
        asks = row.asks # List of [price, qty]
        
        # Take top 5 levels
        # Bids are sorted desc, Asks are sorted asc usually. 
        # Binance API returns them sorted.
        
        # Flatten features: bid_p_1, bid_v_1, ..., ask_p_1, ask_v_1, ...
        features = {'time': time}
        
        for i in range(5):
            # Bids
            if i < len(bids):
                features[f'bid_p_{i+1}'] = float(bids[i][0])
                features[f'bid_v_{i+1}'] = float(bids[i][1])
            else:
                features[f'bid_p_{i+1}'] = 0.0
                features[f'bid_v_{i+1}'] = 0.0
                
            # Asks
            if i < len(asks):
                features[f'ask_p_{i+1}'] = float(asks[i][0])
                features[f'ask_v_{i+1}'] = float(asks[i][1])
            else:
                features[f'ask_p_{i+1}'] = 0.0
                features[f'ask_v_{i+1}'] = 0.0
                
        data.append(features)
        
    df = pd.DataFrame(data)
    if df.empty:
        return df
        
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    
    # Compute Mid Price
    df['mid_price'] = (df['bid_p_1'] + df['ask_p_1']) / 2
    
    # Labeling: Next 10 ticks direction
    # 0: Down, 1: Flat, 2: Up
    k = 10
    threshold = 0.00001 # Small threshold for "Flat"
    
    future_mid = df['mid_price'].shift(-k)
    returns = (future_mid - df['mid_price']) / df['mid_price']
    
    conditions = [
        (returns < -threshold),
        (returns > threshold)
    ]
    choices = [0, 2] # 0: Down, 2: Up
    df['label'] = np.select(conditions, choices, default=1) # 1: Flat
    
    # Drop NaNs
    df.dropna(inplace=True)
    
    return df

async def main():
    logger.info("Fetching LOB data...")
    rows = await fetch_lob_data()
    logger.info(f"Fetched {len(rows)} snapshots.")
    
    if not rows:
        logger.warning("No LOB data found. Skipping dataset build.")
        return

    logger.info("Processing LOB data...")
    df = process_lob_data(rows)
    
    if df.empty:
        logger.warning("Processed dataframe is empty.")
        return
        
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    
    df.to_parquet(DATASET_PATH)
    logger.info(f"Saved to {DATASET_PATH}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
