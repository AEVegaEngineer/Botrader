import asyncio
print("DEBUG: STARTING PAPER TRADER")
import logging
from app.services.collector import BinanceCollector
from app.backtest.strategies.simple_rsi import SimpleRSI
from app.risk.manager import RiskManager
from app.risk.limits import TradeRisk
from app.execution.engine import ExecutionEngine

# For paper trading, we need a loop that:
# 1. Connects to live data (Collector)
# 2. Maintains a virtual portfolio
# 3. Runs strategy logic on each new candle
# 4. Executes virtual orders

# Configure logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, initial_capital=10000.0):
        self.capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        
        # Risk & Execution
        self.risk_manager = RiskManager(initial_capital)
        self.execution = ExecutionEngine()
        
        # Strategy State (Simple RSI)
        self.rsi_period = 14
        self.closes = []

    async def on_candle(self, candle):
        price = candle.close
        self.closes.append(price)
        
        # Update Risk Manager with current valuation
        current_value = self.capital + (self.position * price)
        self.risk_manager.update_balance(current_value)
        
        if self.risk_manager.is_halted:
            logger.warning("Trading Halted by Risk Manager")
            # Logic to close all positions if needed
            if self.position > 0:
                await self.sell(price, "AGGRESSIVE")
            return

        if len(self.closes) > self.rsi_period:
            # Compute RSI (Simplified for paper trading demo)
            # In production, use the indicator library
            import pandas as pd
            import pandas_ta as ta
            
            s = pd.Series(self.closes)
            rsi = ta.rsi(s, length=self.rsi_period).iloc[-1]
            
            logger.info(f"Price: {price}, RSI: {rsi:.2f}")
            
            if self.position == 0:
                if rsi < 30:
                    await self.buy(price)
            elif self.position > 0:
                if rsi > 70:
                    await self.sell(price)

    async def buy(self, price):
        # 1. Get Sizing from Risk Manager
        target_size = self.risk_manager.get_target_position_size(price)
        quantity = min(target_size, (self.capital * 0.99) / price) # Cap at available capital
        
        # 2. Check Risk Limits
        trade_risk = TradeRisk(symbol="BTCUSDT", side="BUY", quantity=quantity, price=price)
        if not self.risk_manager.check_trade_risk(trade_risk):
            logger.warning("Buy rejected by Risk Manager")
            return

        # 3. Execute
        result = await self.execution.execute_order("BTCUSDT", "BUY", quantity, price, style="TWAP")
        
        cost = quantity * price
        self.capital -= cost
        self.position = quantity
        self.entry_price = price
        logger.info(f"PAPER BUY: {quantity:.4f} BTC @ {price:.2f}")

    async def sell(self, price, style="PASSIVE"):
        # Execute
        result = await self.execution.execute_order("BTCUSDT", "SELL", self.position, price, style=style)
        
        revenue = self.position * price
        pnl = revenue - (self.position * self.entry_price)
        self.capital += revenue
        self.position = 0
        self.trades.append({'pnl': pnl})
        logger.info(f"PAPER SELL: @ {price:.2f}, PnL: {pnl:.2f}, Capital: {self.capital:.2f}")

async def main():
    logger.info("Starting Paper Trader...")
    trader = PaperTrader()
    
    # We can hook into the collector or just listen to the stream directly here?
    # For simplicity, let's reuse the collector logic but just for the stream part
    # Or better, let's just run a loop that queries the DB for the latest candle every minute
    # to simulate "on_candle" event, since the collector is already running and populating DB.
    
    from app.core.database import engine
    from sqlalchemy import text
    
    last_time = None
    
    while True:
        try:
            # Poll for new candle
            query = text("SELECT * FROM candles ORDER BY time DESC LIMIT 1")
            async with engine.connect() as conn:
                result = await conn.execute(query)
                row = result.fetchone()
                
            if row:
                if last_time is None or row.time > last_time:
                    last_time = row.time
                    await trader.on_candle(row)
            
            await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in paper loop: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
