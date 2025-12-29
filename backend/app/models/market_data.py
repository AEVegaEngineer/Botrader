from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, BigInteger
from app.core.database import Base

class Candle(Base):
    __tablename__ = "candles"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class Trade(Base):
    __tablename__ = "trades"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    trade_id = Column(BigInteger, primary_key=True) # Add trade_id to PK
    price = Column(Float)
    quantity = Column(Float)
    side = Column(String)  # 'buy' or 'sell'

class LOBSnapshot(Base):
    __tablename__ = "lob_snapshots"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    bids = Column(JSON)  # List of [price, quantity]
    asks = Column(JSON)  # List of [price, quantity]

class CandleIndicators(Base):
    __tablename__ = "candle_indicators"

    time = Column(DateTime(timezone=True), primary_key=True)
    symbol = Column(String, primary_key=True)
    
    # Trend
    sma_20 = Column(Float)
    ema_20 = Column(Float)
    sma_50 = Column(Float)
    ema_50 = Column(Float)
    
    # Momentum
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_diff = Column(Float)
    
    # Volatility
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    bb_upper = Column(Float)
    atr_14 = Column(Float)

class PaperTrade(Base):
    __tablename__ = "paper_trades"

    time = Column(DateTime(timezone=True), primary_key=True)  # Must be first for TimescaleDB hypertable
    symbol = Column(String, primary_key=True)  # Composite key with time
    side = Column(String, nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float)  # Profit/Loss for closed positions
    capital_after = Column(Float)  # Capital after trade
    strategy = Column(String)  # Strategy that generated the trade
