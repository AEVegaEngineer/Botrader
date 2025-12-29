import asyncio
import json
import websockets
import logging
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal, engine
from app.models.market_data import Candle, Trade, LOBSnapshot, CandleIndicators
from app.features.registry import FeatureRegistry

# Subscribe to multiple streams: kline_1m, aggTrade, depth5
BINANCE_WS_URL = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m/btcusdt@aggTrade/btcusdt@depth5"

logger = logging.getLogger(__name__)

class BinanceCollector:
    def __init__(self):
        self.running = False
        self.latest_price = 0.0

    def get_latest_price(self):
        return self.latest_price

    async def start(self):
        self.running = True
        while self.running:
            try:
                async with websockets.connect(BINANCE_WS_URL) as websocket:
                    logger.info(f"Connected to Binance WebSocket: {BINANCE_WS_URL}")
                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await self.process_message(data)
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                await asyncio.sleep(5)

    async def process_message(self, message):
        if 'stream' not in message or 'data' not in message:
            return
            
        stream = message['stream']
        data = message['data']
        
        try:
            if 'kline_1m' in stream:
                await self.process_kline(data)
            elif 'aggTrade' in stream:
                await self.process_trade(data)
            elif 'depth5' in stream:
                await self.process_depth(data)
        except Exception as e:
            logger.error(f"Error processing message from {stream}: {e}")

    async def process_kline(self, data):
        kline = data['k']
        if kline['x']:  # Candle closed
            # Normalize timestamp to UTC
            ts = datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc)
            candle = Candle(
                time=ts,
                symbol="BTCUSDT",
                open=float(kline['o']),
                high=float(kline['h']),
                low=float(kline['l']),
                close=float(kline['c']),
                volume=float(kline['v'])
            )
            await self.save_to_db(candle)
            
            # Compute and save indicators
            await self.update_indicators(candle)

    async def update_indicators(self, new_candle: Candle):
        # Fetch last 100 candles for indicator computation
        query = text(f"""
            SELECT time, open, high, low, close, volume
            FROM candles
            WHERE symbol = :symbol AND time < :time
            ORDER BY time DESC
            LIMIT 100
        """)
        
        async with engine.connect() as conn:
            result = await conn.execute(query, {"symbol": new_candle.symbol, "time": new_candle.time})
            rows = result.fetchall()
            
        # Create DataFrame
        data = [
            {
                'time': row.time, 'open': row.open, 'high': row.high, 
                'low': row.low, 'close': row.close, 'volume': row.volume
            } for row in rows
        ]
        # Reverse to chronological order
        data.reverse()
        
        # Append new candle
        data.append({
            'time': new_candle.time,
            'open': new_candle.open,
            'high': new_candle.high,
            'low': new_candle.low,
            'close': new_candle.close,
            'volume': new_candle.volume
        })
        
        df = pd.DataFrame(data)
        if df.empty:
            return

        df.set_index('time', inplace=True)
        
        # Compute indicators
        df_indicators = FeatureRegistry.compute_all(df)
        
        # Get the last row (the new candle)
        last_row = df_indicators.iloc[-1]
        
        # Save to DB
        indicator = CandleIndicators(
            time=new_candle.time,
            symbol=new_candle.symbol,
            sma_20=last_row.get('SMA_20'),
            ema_20=last_row.get('EMA_20'),
            sma_50=last_row.get('SMA_50'),
            ema_50=last_row.get('EMA_50'),
            rsi_14=last_row.get('RSI_14'),
            macd=last_row.get('MACD'),
            macd_signal=last_row.get('MACD_SIGNAL'),
            macd_diff=last_row.get('MACD_DIFF'),
            bb_lower=last_row.get('BBL_20_2.0'),
            bb_middle=last_row.get('BBM_20_2.0'),
            bb_upper=last_row.get('BBU_20_2.0'),
            atr_14=last_row.get('ATR_14')
        )
        await self.save_to_db(indicator)

    async def process_trade(self, data):
        # aggTrade format:
        # "e": "aggTrade", "E": 123456789, "s": "BTCUSDT", "a": 12345, "p": "0.001", "q": "100", ...
        ts = datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc)
        trade = Trade(
            time=ts,
            symbol="BTCUSDT",
            trade_id=data['a'], # Aggregate trade ID
            price=float(data['p']),
            quantity=float(data['q']),
            side="SELL" if data['m'] else "BUY" # m=True means maker (sell), m=False means taker (buy)
        )
        self.latest_price = trade.price
        await self.save_to_db(trade)

    async def process_depth(self, data):
        # Partial Book Depth Streams:
        # "lastUpdateId": 160, "bids": [ [ "0.0024", "10" ] ], "asks": [ [ "0.0026", "100" ] ]
        # No explicit timestamp in depth payload usually, use local time or if available in wrapper
        # stream payload usually has no timestamp for depth updates unless it's diff depth.
        # But for @depth5 (partial book), it might not have 'E'.
        # Let's check if 'E' exists in data, otherwise use current time.
        # Actually standard stream response wrapper often has no timestamp for the message itself if it's raw stream?
        # Wait, combined stream format: {"stream": "...", "data": {...}}
        # Depth5 payload: { "lastUpdateId": ..., "bids": ..., "asks": ... } -> No timestamp.
        # We will use arrival time.
        
        ts = datetime.now(timezone.utc)
        
        snapshot = LOBSnapshot(
            time=ts,
            symbol="BTCUSDT",
            bids=data['bids'],
            asks=data['asks']
        )
        await self.save_to_db(snapshot)

    async def save_to_db(self, obj):
        from app.core.db_utils import upsert_object
        from app.models.market_data import Candle, Trade, LOBSnapshot, CandleIndicators
        
        async with AsyncSessionLocal() as session:
            async with session.begin():
                # Determine index elements based on object type
                if isinstance(obj, Candle):
                    index_elements = ['time', 'symbol']
                elif isinstance(obj, Trade):
                    index_elements = ['time', 'symbol', 'trade_id']
                elif isinstance(obj, LOBSnapshot):
                    index_elements = ['time', 'symbol']
                elif isinstance(obj, CandleIndicators):
                    index_elements = ['time', 'symbol']
                else:
                    # Fallback for unknown types (or just add if no PK conflict expected)
                    session.add(obj)
                    return

                # Convert object to dict (excluding internal SQLAlchemy state)
                values = {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
                
                await upsert_object(session, type(obj), values, index_elements)
    
    def stop(self):
        self.running = False
