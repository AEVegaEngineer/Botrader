import backtrader as bt
import pandas as pd
from sqlalchemy import text
from app.core.database import engine

class TimescaleDBData(bt.feeds.PandasData):
    """
    Backtrader feed that loads data from TimescaleDB via Pandas.
    """
    params = (
        ('datetime', None), # Index
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

    @classmethod
    async def get_data(cls, symbol: str, timeframe: str = '1m', limit: int = 10000):
        """
        Fetch data from TimescaleDB and return a Pandas DataFrame formatted for Backtrader.
        """
        # TODO: Handle different timeframes by resampling if needed.
        # For now, we assume 1m candles are stored and we fetch them directly.
        
        query = text(f"""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol
            ORDER BY time ASC
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
        
        # Backtrader expects 'openinterest' column usually, or we map it to None
        df['openinterest'] = 0
        
        return df
