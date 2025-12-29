import asyncio
import pandas as pd
import logging
import ta
from sqlalchemy import text
from app.core.database import engine, AsyncSessionLocal
from app.models.market_data import CandleIndicators
from app.lib.indicators import compute_indicators
from app.core.db_utils import upsert_object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_candles(symbol: str):
    query = text(f"""
        SELECT time, open, high, low, close, volume
        FROM candles
        WHERE symbol = :symbol
        ORDER BY time ASC
    """)
    
    async with engine.connect() as conn:
        result = await conn.execute(query, {"symbol": symbol})
        rows = result.fetchall()
        
    if not rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

async def save_indicators(symbol: str, df: pd.DataFrame):
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for time, row in df.iterrows():
                # Skip rows with NaNs (warmup period)
                if row.isna().any():
                    continue
                    
                # Convert to dict
                values = {
                    'time': time,
                    'symbol': symbol,
                    'sma_20': row.get('SMA_20'),
                    'ema_20': row.get('EMA_20'),
                    'sma_50': row.get('SMA_50'),
                    'ema_50': row.get('EMA_50'),
                    'rsi_14': row.get('RSI_14'),
                    'macd': row.get('MACD'),
                    'macd_signal': row.get('MACD_SIGNAL'),
                    'macd_diff': row.get('MACD_DIFF'),
                    'bb_lower': row.get('BBL_20_2.0'),
                    'bb_middle': row.get('BBM_20_2.0'),
                    'bb_upper': row.get('BBU_20_2.0'),
                    'atr_14': row.get('ATR_14')
                }
                
                await upsert_object(session, CandleIndicators, values, ['time', 'symbol'])
                
    logger.info(f"Saved indicators for {symbol}")

async def main():
    symbol = "BTCUSDT"
    logger.info(f"Fetching candles for {symbol}...")
    
    df = await fetch_candles(symbol)
    if df.empty:
        logger.warning("No candles found.")
        return
        
    logger.info(f"Fetched {len(df)} candles. Computing indicators...")
    df_indicators = compute_indicators(df)
    
    logger.info("Saving indicators to DB...")
    await save_indicators(symbol, df_indicators)
    logger.info("Backfill complete.")

if __name__ == "__main__":
    asyncio.run(main())
