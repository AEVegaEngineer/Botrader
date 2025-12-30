"""
API endpoints for technical indicators and candle data.
Supports chart visualization with SMA, EMA, RSI, MACD, etc.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/indicators", tags=["indicators"])

# ============================================================================
# Candle Data Endpoint
# ============================================================================

@router.get("/candles")
async def get_candles(
    symbol: str = Query("BTCUSDT", description="Trading symbol"),
    interval: str = Query("1m", description="Candle interval"),
    limit: int = Query(500, description="Number of candles to return"),
    hours_back: Optional[int] = Query(None, description="Hours back from now to fetch (auto-calculated if not provided)")
):
    """
    Get historical candle data with volume.
    
    Returns OHLCV data for charting.
    Aggregates 1-minute candles into the requested interval.
    Uses time-based window to avoid gaps in data.
    """
    try:
        from app.core.database import engine
        from sqlalchemy import text
        
        # Map interval strings to PostgreSQL interval format
        interval_map = {
            '1m': '1 minute',
            '3m': '3 minutes',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '6h': '6 hours',
            '8h': '8 hours',
            '12h': '12 hours',
            '1d': '1 day',
            '3d': '3 days',
            '1w': '1 week',
            '1M': '1 month'
        }
        
        pg_interval = interval_map.get(interval, '1 minute')
        
        # Calculate time window based on interval if not provided
        if hours_back is None:
            # Default hours back based on interval to show reasonable amount of data
            hours_back_map = {
                '1m': 2,      # 2 hours for 1m = ~120 candles
                '3m': 6,      # 6 hours for 3m = ~120 candles
                '5m': 10,     # 10 hours for 5m = ~120 candles
                '15m': 24,    # 24 hours for 15m = ~96 candles
                '30m': 48,    # 48 hours for 30m = ~96 candles
                '1h': 72,     # 3 days for 1h = ~72 candles
                '2h': 168,    # 7 days for 2h = ~84 candles
                '4h': 336,    # 14 days for 4h = ~84 candles
                '6h': 504,    # 21 days for 6h = ~84 candles
                '8h': 672,    # 28 days for 8h = ~84 candles
                '12h': 720,   # 30 days for 12h = ~60 candles
                '1d': 720,    # 30 days for 1d = ~30 candles
                '3d': 2160,   # 90 days for 3d = ~30 candles
                '1w': 5040,   # 210 days for 1w = ~30 candles
                '1M': 21600   # 900 days for 1M = ~30 candles
            }
            hours_back = hours_back_map.get(interval, 24)
        
        # Calculate start time
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours_back)
        
        # Use time_bucket to aggregate candles
        # Filter by time window to avoid gaps
        query = text(f"""
            WITH recent_candles AS (
                SELECT time, open, high, low, close, volume
                FROM candles
                WHERE symbol = :symbol
                  AND time >= :start_time
                ORDER BY time DESC
            )
            SELECT 
                time_bucket('{pg_interval}'::interval, time) AS time,
                FIRST(open, time) AS open,
                MAX(high) AS high,
                MIN(low) AS low,
                LAST(close, time) AS close,
                SUM(volume) AS volume
            FROM recent_candles
            GROUP BY time_bucket('{pg_interval}'::interval, time)
            ORDER BY time DESC
            LIMIT :limit
        """)
        
        async with engine.connect() as conn:
            result = await conn.execute(query, {
                "symbol": symbol,
                "start_time": start_time,
                "limit": limit
            })
            rows = result.fetchall()
            
        candles = [
            {
                "time": row.time.isoformat(),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume
            }
            for row in rows
        ]
        
        # Return in chronological order
        candles.reverse()
        
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": candles
        }
    except Exception as e:
        logger.error(f"Error fetching candles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Moving Averages
# ============================================================================

@router.get("/sma")
async def get_sma(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(20, description="SMA period"),
    interval: str = Query("1m", description="Candle interval"),
    limit: int = Query(500)
):
    """Get Simple Moving Average from candle_indicators table"""
    try:
        from app.core.database import engine
        from sqlalchemy import text
        
        # Map interval strings to PostgreSQL interval format (reuse from candles endpoint)
        interval_map = {
            '1m': '1 minute',
            '3m': '3 minutes',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '6h': '6 hours',
            '8h': '8 hours',
            '12h': '12 hours',
            '1d': '1 day',
            '3d': '3 days',
            '1w': '1 week',
            '1M': '1 month'
        }
        
        pg_interval = interval_map.get(interval, '1 minute')
        
        # Calculate time window (reuse logic from candles endpoint)
        hours_back_map = {
            '1m': 2,      # 2 hours for 1m = ~120 candles
            '3m': 6,      # 6 hours for 3m = ~120 candles
            '5m': 10,     # 10 hours for 5m = ~120 candles
            '15m': 24,    # 24 hours for 15m = ~96 candles
            '30m': 48,    # 48 hours for 30m = ~96 candles
            '1h': 72,     # 3 days for 1h = ~72 candles
            '2h': 168,    # 7 days for 2h = ~84 candles
            '4h': 336,    # 14 days for 4h = ~84 candles
            '6h': 504,    # 21 days for 6h = ~84 candles
            '8h': 672,    # 28 days for 8h = ~84 candles
            '12h': 720,   # 30 days for 12h = ~60 candles
            '1d': 720,    # 30 days for 1d = ~30 candles
            '3d': 2160,   # 90 days for 3d = ~30 candles
            '1w': 5040,   # 210 days for 1w = ~30 candles
            '1M': 21600   # 900 days for 1M = ~30 candles
        }
        hours_back = hours_back_map.get(interval, 24)
        
        # Calculate start time
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours_back)
        
        # Select the appropriate SMA column based on period
        sma_column = f"sma_{period}" if period in [20, 50] else "sma_20"
        
        # Aggregate indicators using time_bucket to match candle intervals
        query = text(f"""
            WITH recent_indicators AS (
                SELECT time, {sma_column} as sma_value
                FROM candle_indicators
                WHERE symbol = :symbol
                  AND time >= :start_time
                  AND {sma_column} IS NOT NULL
                ORDER BY time DESC
            )
            SELECT 
                time_bucket('{pg_interval}'::interval, time) AS time,
                LAST(sma_value, time) AS sma_value
            FROM recent_indicators
            GROUP BY time_bucket('{pg_interval}'::interval, time)
            ORDER BY time DESC
            LIMIT :limit
        """)
        
        async with engine.connect() as conn:
            result = await conn.execute(query, {
                "symbol": symbol,
                "start_time": start_time,
                "limit": limit
            })
            rows = result.fetchall()
        
        data = [
            {
                "timestamp": row.time.isoformat(),
                "value": float(row.sma_value)
            }
            for row in rows if row.sma_value is not None
        ]
        
        # Return in chronological order
        data.reverse()
        
        return {
            "symbol": symbol,
            "indicator": "SMA",
            "period": period,
            "interval": interval,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ema")
async def get_ema(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(50, description="EMA period"),
    interval: str = Query("1m", description="Candle interval"),
    limit: int = Query(500)
):
    """Get Exponential Moving Average from candle_indicators table"""
    try:
        from app.core.database import engine
        from sqlalchemy import text
        
        # Map interval strings to PostgreSQL interval format (reuse from candles endpoint)
        interval_map = {
            '1m': '1 minute',
            '3m': '3 minutes',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '2h': '2 hours',
            '4h': '4 hours',
            '6h': '6 hours',
            '8h': '8 hours',
            '12h': '12 hours',
            '1d': '1 day',
            '3d': '3 days',
            '1w': '1 week',
            '1M': '1 month'
        }
        
        pg_interval = interval_map.get(interval, '1 minute')
        
        # Calculate time window (reuse logic from candles endpoint)
        hours_back_map = {
            '1m': 2,      # 2 hours for 1m = ~120 candles
            '3m': 6,      # 6 hours for 3m = ~120 candles
            '5m': 10,     # 10 hours for 5m = ~120 candles
            '15m': 24,    # 24 hours for 15m = ~96 candles
            '30m': 48,    # 48 hours for 30m = ~96 candles
            '1h': 72,     # 3 days for 1h = ~72 candles
            '2h': 168,    # 7 days for 2h = ~84 candles
            '4h': 336,    # 14 days for 4h = ~84 candles
            '6h': 504,    # 21 days for 6h = ~84 candles
            '8h': 672,    # 28 days for 8h = ~84 candles
            '12h': 720,   # 30 days for 12h = ~60 candles
            '1d': 720,    # 30 days for 1d = ~30 candles
            '3d': 2160,   # 90 days for 3d = ~30 candles
            '1w': 5040,   # 210 days for 1w = ~30 candles
            '1M': 21600   # 900 days for 1M = ~30 candles
        }
        hours_back = hours_back_map.get(interval, 24)
        
        # Calculate start time
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(hours=hours_back)
        
        # Select the appropriate EMA column based on period
        ema_column = f"ema_{period}" if period in [20, 50] else "ema_50"
        
        # Aggregate indicators using time_bucket to match candle intervals
        query = text(f"""
            WITH recent_indicators AS (
                SELECT time, {ema_column} as ema_value
                FROM candle_indicators
                WHERE symbol = :symbol
                  AND time >= :start_time
                  AND {ema_column} IS NOT NULL
                ORDER BY time DESC
            )
            SELECT 
                time_bucket('{pg_interval}'::interval, time) AS time,
                LAST(ema_value, time) AS ema_value
            FROM recent_indicators
            GROUP BY time_bucket('{pg_interval}'::interval, time)
            ORDER BY time DESC
            LIMIT :limit
        """)
        
        async with engine.connect() as conn:
            result = await conn.execute(query, {
                "symbol": symbol,
                "start_time": start_time,
                "limit": limit
            })
            rows = result.fetchall()
        
        data = [
            {
                "timestamp": row.time.isoformat(),
                "value": float(row.ema_value)
            }
            for row in rows if row.ema_value is not None
        ]
        
        # Return in chronological order
        data.reverse()
        
        return {
            "symbol": symbol,
            "indicator": "EMA",
            "period": period,
            "interval": interval,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Oscillators
# ============================================================================

@router.get("/rsi")
async def get_rsi(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(14, description="RSI period"),
    limit: int = Query(500)
):
    """Get Relative Strength Index"""
    try:
        return {
            "symbol": symbol,
            "indicator": "RSI",
            "period": period,
            "data": [],
            "overbought": 70,
            "oversold": 30
        }
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macd")
async def get_macd(
    symbol: str = Query("BTCUSDT"),
    fast_period: int = Query(12),
    slow_period: int = Query(26),
    signal_period: int = Query(9),
    limit: int = Query(500)
):
    """Get MACD (Moving Average Convergence Divergence)"""
    try:
        return {
            "symbol": symbol,
            "indicator": "MACD",
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
            "data": [
                # {
                #     "timestamp": "2024-01-01T00:00:00Z",
                #     "macd": 150.0,
                #     "signal": 140.0,
                #     "histogram": 10.0
                # }
            ]
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Volatility Indicators
# ============================================================================

@router.get("/bollinger-bands")
async def get_bollinger_bands(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(20),
    std_dev: float = Query(2.0, description="Standard deviations"),
    limit: int = Query(500)
):
    """Get Bollinger Bands"""
    try:
        return {
            "symbol": symbol,
            "indicator": "Bollinger Bands",
            "period": period,
            "std_dev": std_dev,
            "data": [
                # {
                #     "timestamp": "2024-01-01T00:00:00Z",
                #     "upper": 51000.0,
                #     "middle": 50000.0,
                #     "lower": 49000.0
                # }
            ]
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/atr")
async def get_atr(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(14, description="ATR period"),
    limit: int = Query(500)
):
    """Get Average True Range"""
    try:
        return {
            "symbol": symbol,
            "indicator": "ATR",
            "period": period,
            "data": []
        }
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Volume Indicators
# ============================================================================

@router.get("/volume-sma")
async def get_volume_sma(
    symbol: str = Query("BTCUSDT"),
    period: int = Query(20),
    limit: int = Query(500)
):
    """Get Volume Moving Average"""
    try:
        return {
            "symbol": symbol,
            "indicator": "Volume SMA",
            "period": period,
            "data": []
        }
    except Exception as e:
        logger.error(f"Error calculating Volume SMA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Multi-Indicator Batch Endpoint
# ============================================================================

@router.get("/batch")
async def get_indicators_batch(
    symbol: str = Query("BTCUSDT"),
    indicators: str = Query("sma_20,ema_50,rsi_14,macd", description="Comma-separated list"),
    limit: int = Query(500)
):
    """
    Get multiple indicators in a single request for efficiency.
    
    Example: indicators=sma_20,ema_50,rsi_14,macd
    """
    try:
        indicator_list = [i.strip() for i in indicators.split(',')]
        
        result = {
            "symbol": symbol,
            "indicators": {}
        }
        
        # Parse and fetch each indicator
        # This would call the actual computation functions
        for indicator in indicator_list:
            # Parse indicator_period format
            parts = indicator.split('_')
            indicator_type = parts[0]
            
            # Mock response
            result["indicators"][indicator] = {
                "type": indicator_type,
                "data": []
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching batch indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))
