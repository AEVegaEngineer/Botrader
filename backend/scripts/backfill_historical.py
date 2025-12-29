"""
Fetch historical klines data from Binance REST API and save to TimescaleDB.

This script fetches historical candlestick data from Binance starting from a specified date
and populates the TimescaleDB with the data for backtesting and analysis.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict
import aiohttp
from sqlalchemy import text
from app.core.database import AsyncSessionLocal, engine
from app.models.market_data import Candle
from app.core.db_utils import upsert_object

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BINANCE_REST_API = "https://api.binance.com/api/v3/klines"


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int = None,
    limit: int = 1000
) -> List[List]:
    """
    Fetch historical klines from Binance REST API.
    
    Args:
        session: aiohttp client session
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
        start_time: Start timestamp in milliseconds
        end_time: Optional end timestamp in milliseconds
        limit: Number of candles to fetch (max 1000)
    
    Returns:
        List of kline data
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit
    }
    
    if end_time:
        params["endTime"] = end_time
    
    async with session.get(BINANCE_REST_API, params=params) as response:
        if response.status != 200:
            logger.error(f"Error fetching klines: {response.status}")
            return []
        
        data = await response.json()
        return data


async def save_candles_batch(candles: List[Candle]):
    """Save a batch of candles to the database."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for candle in candles:
                values = {
                    'time': candle.time,
                    'symbol': candle.symbol,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume
                }
                await upsert_object(session, Candle, values, ['time', 'symbol'])


async def backfill_historical_data(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    start_date: str = "2025-10-01",
    end_date: str = None
):
    """
    Backfill historical kline data from Binance.
    
    Args:
        symbol: Trading pair
        interval: Kline interval
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: Optional end date in 'YYYY-MM-DD' format (defaults to now)
    """
    # Convert dates to timestamps
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_timestamp = int(start_dt.timestamp() * 1000)
    
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_timestamp = int(end_dt.timestamp() * 1000)
    else:
        end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date or 'now'}")
    logger.info(f"Interval: {interval}")
    
    total_candles = 0
    current_start = start_timestamp
    
    async with aiohttp.ClientSession() as session:
        while current_start < end_timestamp:
            # Fetch batch of candles
            klines = await fetch_klines(
                session,
                symbol,
                interval,
                current_start,
                end_timestamp,
                limit=1000
            )
            
            if not klines:
                logger.info("No more data to fetch")
                break
            
            # Convert to Candle objects
            candles = []
            for kline in klines:
                # Kline format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
                candle = Candle(
                    time=datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),
                    symbol=symbol,
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                )
                candles.append(candle)
            
            # Save batch to database
            await save_candles_batch(candles)
            total_candles += len(candles)
            
            logger.info(f"Saved {len(candles)} candles. Total: {total_candles}")
            
            # Update start time for next batch
            # Use the close time of the last candle + 1ms
            current_start = klines[-1][6] + 1
            
            # Rate limiting - Binance allows 1200 requests per minute
            await asyncio.sleep(0.1)
    
    logger.info(f"Backfill complete! Total candles saved: {total_candles}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill historical data from Binance")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument("--start-date", default="2025-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD), defaults to now")
    
    args = parser.parse_args()
    
    await backfill_historical_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == "__main__":
    asyncio.run(main())
