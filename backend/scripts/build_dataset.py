import asyncio
import pandas as pd
import logging
import os
from sqlalchemy import text
from app.core.database import engine
from app.lib.labeling import add_labels
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_data(symbol: str, limit: int = 10000):
    query = text(f"""
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol
        ORDER BY time DESC
        LIMIT :limit
    """)
    
    async with engine.connect() as conn:
        result = await conn.execute(query, {"symbol": symbol, "limit": limit})
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
    
    df = await fetch_data(symbol)
    
    if df.empty:
        logger.warning("No data found in database.")
        return

    logger.info(f"Fetched {len(df)} rows. Computing indicators...")
    
    df_indicators = FeatureRegistry.compute_all(df)
    df_labeled = add_labels(df_indicators, horizons=[5])
    
    # Drop NaNs created by indicators/lagging
    df_final = df_labeled.dropna()
    
    output_file = "dataset.parquet"
    df_final.to_parquet(output_file)
    logger.info(f"Dataset saved to {output_file}. Shape: {df_final.shape}")

if __name__ == "__main__":
    asyncio.run(main())
