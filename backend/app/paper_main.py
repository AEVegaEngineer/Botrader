import asyncio
import logging
import os
import json
import pandas as pd
from datetime import datetime, timezone
from app.services.collector import BinanceCollector
from app.risk.manager import RiskManager
from app.risk.limits import TradeRisk
from app.execution.engine import ExecutionEngine
from app.ml.inference import ActionPredictor
from app.models.market_data import PaperTrade
from app.core.database import AsyncSessionLocal
from app.core.db_utils import upsert_object

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
        self.initial_capital = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        
        # Risk & Execution
        self.risk_manager = RiskManager(initial_capital)
        self.execution = ExecutionEngine()
        
        # Strategy Config
        self.strategy_name = os.getenv("ACTIVE_STRATEGY", "rsi_strategy")
        logger.info(f"Active Strategy: {self.strategy_name}")
        
        # Strategy State
        self.history = [] # For ML strategy
        self.rsi_period = 14 # For RSI strategy
        
        # Bot status file path
        self.bot_status_file = "bot_status.json"
        self.position_state_file = "position_state.json"
        
        if self.strategy_name == "ml_action_transformer":
            self.predictor = ActionPredictor()
            self.min_history = 100 + self.predictor.seq_len
        else:
            self.min_history = 50
        
        # Restore position state from file (async calculation from DB will happen in main loop)
        self._restore_position_state()
        self._position_restored = False  # Flag to track if we need to calculate from DB
    
    def _save_position_state(self):
        """Save current position state to file"""
        try:
            state = {
                "position": self.position,
                "entry_price": self.entry_price,
                "capital": self.capital,
                "initial_capital": self.initial_capital,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.position_state_file, 'w') as f:
                json.dump(state, f)
            logger.debug(f"Saved position state: position={self.position}, capital={self.capital}")
        except Exception as e:
            logger.error(f"Error saving position state: {e}")
    
    def _restore_position_state(self):
        """Restore position state from file"""
        # Try to restore from file
        if os.path.exists(self.position_state_file):
            try:
                with open(self.position_state_file, 'r') as f:
                    state = json.load(f)
                    self.position = float(state.get("position", 0.0))
                    self.entry_price = float(state.get("entry_price", 0.0))
                    self.capital = float(state.get("capital", self.initial_capital))
                    self.initial_capital = float(state.get("initial_capital", self.initial_capital))
                    logger.info(f"Restored position state from file: position={self.position}, entry_price={self.entry_price}, capital={self.capital}")
                    self._position_restored = True
                    return
            except Exception as e:
                logger.warning(f"Error reading position state file: {e}, will calculate from trade history")
        
        # Mark that we need to calculate from DB
        self._position_restored = False
    
    async def _calculate_position_from_history(self):
        """Calculate current position from trade history in database"""
        try:
            from app.core.database import engine
            from sqlalchemy import text
            
            query = text("""
                SELECT side, quantity, price, capital_after
                FROM paper_trades
                WHERE symbol = 'BTCUSDT'
                ORDER BY time ASC
            """)
            
            async with engine.connect() as conn:
                result = await conn.execute(query)
                rows = result.fetchall()
            
            if not rows:
                logger.info("No trade history found, using initial state")
                return
            
            # Calculate position by summing BUY and SELL trades
            position = 0.0
            total_cost = 0.0
            capital = self.initial_capital
            
            for row in rows:
                side = row.side
                quantity = float(row.quantity)
                price = float(row.price)
                capital_after = float(row.capital_after) if row.capital_after else None
                
                if side == "BUY":
                    position += quantity
                    total_cost += quantity * price
                    capital -= quantity * price
                elif side == "SELL":
                    position -= quantity
                    capital += quantity * price
                    # Reset cost basis when position closes
                    if position == 0:
                        total_cost = 0.0
                
                # Use capital_after from last trade if available
                if capital_after is not None:
                    capital = capital_after
            
            # Calculate average entry price
            entry_price = total_cost / position if position > 0 else 0.0
            
            self.position = position
            self.entry_price = entry_price
            self.capital = capital
            
            logger.info(f"Calculated position from history: position={position}, entry_price={entry_price}, capital={capital}")
            
            # Save the calculated state
            self._save_position_state()
            
        except Exception as e:
            logger.error(f"Error calculating position from history: {e}")
    
    def is_bot_running(self) -> bool:
        """Check if bot is running by reading status file"""
        if os.path.exists(self.bot_status_file):
            try:
                with open(self.bot_status_file, 'r') as f:
                    status = json.load(f)
                    return status.get("is_running", True)  # Default to True if key missing
            except Exception as e:
                logger.warning(f"Error reading bot status: {e}")
                return True  # Default to running if file exists but can't be read
        # If file doesn't exist, default to running for backward compatibility
        # API will create file and control state going forward
        return True
    
    async def save_trade_to_db(self, side: str, quantity: float, price: float, pnl: float = None):
        """Save trade to database"""
        try:
            trade_time = datetime.now(timezone.utc)
            symbol = "BTCUSDT"
            
            trade = PaperTrade(
                time=trade_time,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                pnl=pnl,
                capital_after=self.capital,
                strategy=self.strategy_name
            )
            
            async with AsyncSessionLocal() as session:
                async with session.begin():
                    session.add(trade)
                    await session.commit()
            logger.debug(f"Saved {side} trade to database")
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")

    async def on_candle(self, candle):
        # Check if bot is running
        if not self.is_bot_running():
            return
        
        price = candle.close
        
        # Update history
        row = {
            'time': candle.time,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        }
        self.history.append(row)
        if len(self.history) > self.min_history + 100:
            self.history = self.history[-self.min_history:]
            
        # Update Risk Manager
        current_value = self.capital + (self.position * price)
        self.risk_manager.update_balance(current_value)
        
        if self.risk_manager.is_halted:
            logger.warning("Trading Halted by Risk Manager")
            if self.position > 0:
                await self.sell(price, "AGGRESSIVE")
            return

        # Strategy Logic
        if self.strategy_name == "ml_action_transformer":
            await self.run_ml_strategy(price)
        else:
            await self.run_rsi_strategy(price)

    async def run_rsi_strategy(self, price):
        if len(self.history) <= self.rsi_period:
            return
            
        closes = [x['close'] for x in self.history]
        
        import ta
        s = pd.Series(closes)
        rsi_indicator = ta.momentum.RSIIndicator(s, window=self.rsi_period)
        rsi = rsi_indicator.rsi().iloc[-1]
        
        logger.info(f"Price: {price}, RSI: {rsi:.2f}")
        
        if self.position == 0:
            if rsi < 30:
                await self.buy(price)
        elif self.position > 0:
            if rsi > 70:
                await self.sell(price)

    async def run_ml_strategy(self, price):
        if len(self.history) < self.min_history:
            return
            
        df = pd.DataFrame(self.history)
        action = self.predictor.predict(df)
        
        logger.info(f"Price: {price}, ML Action: {action}")
        
        # 0: HOLD, 1: SELL, 2: BUY
        if action == 2: # BUY
            if self.position == 0:
                await self.buy(price)
            elif self.position < 0: # Short
                await self.close_position(price)
                await self.buy(price)
                
        elif action == 1: # SELL
            if self.position > 0:
                await self.sell(price)
            # If we want to support shorting:
            # elif self.position == 0:
            #     await self.short(price)

    async def buy(self, price):
        target_size = self.risk_manager.get_target_position_size(price)
        quantity = min(target_size, (self.capital * 0.99) / price)
        
        trade_risk = TradeRisk(symbol="BTCUSDT", side="BUY", quantity=quantity, price=price)
        if not self.risk_manager.check_trade_risk(trade_risk):
            logger.warning("Buy rejected by Risk Manager")
            return

        await self.execution.execute_order("BTCUSDT", "BUY", quantity, price, style="TWAP")
        
        cost = quantity * price
        self.capital -= cost
        self.position = quantity
        self.entry_price = price
        
        # Save to database
        await self.save_trade_to_db("BUY", quantity, price)
        
        # Save position state
        self._save_position_state()
        
        logger.info(f"PAPER BUY: {quantity:.4f} BTC @ {price:.2f}")

    async def sell(self, price, style="PASSIVE"):
        quantity = self.position
        await self.execution.execute_order("BTCUSDT", "SELL", quantity, price, style=style)
        
        revenue = quantity * price
        pnl = revenue - (quantity * self.entry_price)
        self.capital += revenue
        self.position = 0
        self.entry_price = 0.0
        
        # Save to database
        await self.save_trade_to_db("SELL", quantity, price, pnl=pnl)
        
        # Save position state
        self._save_position_state()
        
        self.trades.append({'pnl': pnl})
        logger.info(f"PAPER SELL: @ {price:.2f}, PnL: {pnl:.2f}, Capital: {self.capital:.2f}")

    async def close_position(self, price):
        if self.position > 0:
            await self.sell(price)
        elif self.position < 0:
            # Implement buy to cover
            pass

async def main():
    logger.info("Starting Paper Trader...")
    trader = PaperTrader()
    
    from app.core.database import engine
    from sqlalchemy import text
    
    # Calculate position from trade history if not restored from file
    if not trader._position_restored:
        await trader._calculate_position_from_history()
    
    last_time = None
    
    while True:
        try:
            # Check for strategy update
            if os.path.exists("active_strategy.txt"):
                with open("active_strategy.txt", "r") as f:
                    new_strategy = f.read().strip()
                    if new_strategy != trader.strategy_name:
                        logger.info(f"Switching strategy to {new_strategy}")
                        trader.strategy_name = new_strategy
                        # Re-init if needed
                        if new_strategy == "ml_action_transformer":
                            if not hasattr(trader, 'predictor'):
                                trader.predictor = ActionPredictor()
                            trader.min_history = 100 + trader.predictor.seq_len
                        else:
                            trader.min_history = 50

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
