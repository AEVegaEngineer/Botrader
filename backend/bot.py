import asyncio
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from datetime import datetime
from config import API_KEY, API_SECRET, TESTNET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, symbol="BTCUSDT", interval="1m", limit=100):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.is_running = False
        self.client = None
        self.trades = []
        self.status = "Stopped"
        
        if API_KEY and API_SECRET:
            try:
                self.client = Client(API_KEY, API_SECRET, testnet=TESTNET)
                logger.info(f"Connected to Binance (Testnet: {TESTNET})")
            except Exception as e:
                logger.error(f"Failed to connect to Binance: {e}")
        else:
            logger.warning("API keys not found. Bot will run in simulation mode (no real connection).")

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.status = "Running"
        logger.info("Bot started")
        
        # Initialize thresholds based on current price
        current_price = self.get_current_price()
        if current_price > 0:
            self.buy_threshold = current_price - 100
            self.sell_threshold = current_price + 100
            logger.info(f"Strategy initialized: Buy < {self.buy_threshold}, Sell > {self.sell_threshold}")
        
        asyncio.create_task(self._run_loop())

    def stop(self):
        self.is_running = False
        self.status = "Stopped"
        logger.info("Bot stopped")

    async def _run_loop(self):
        open_trades = 0
        max_open_trades = 10
        trade_quantity = 0.001

        while self.is_running:
            try:
                if self.client:
                    price = self.get_current_price()
                    logger.info(f"Current price: {price}")
                    
                    if hasattr(self, 'buy_threshold'):
                        if open_trades < max_open_trades and price < self.buy_threshold:
                            logger.info(f"Price {price} < {self.buy_threshold}. Buying.")
                            # In real mode: self.client.order_market_buy(...)
                            self.execute_trade("BUY", trade_quantity, price)
                            open_trades += 1
                        
                        elif open_trades > 0 and price > self.sell_threshold:
                            logger.info(f"Price {price} > {self.sell_threshold}. Selling.")
                            # In real mode: self.client.order_market_sell(...)
                            self.execute_trade("SELL", trade_quantity, price)
                            open_trades -= 1
                
                await asyncio.sleep(5) # Wait 5 seconds
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                await asyncio.sleep(5)

    def get_current_price(self):
        if self.client:
            try:
                ticker = self.client.get_symbol_ticker(symbol=self.symbol)
                return float(ticker['price'])
            except Exception as e:
                logger.error(f"Error getting price: {e}")
                return 0.0
        return 0.0

    def get_history(self):
        # Return local trade history
        return self.trades

    def get_status(self):
        return {
            "status": self.status,
            "symbol": self.symbol,
            "is_running": self.is_running
        }

    def execute_trade(self, side, quantity, price):
        trade = {
            "id": len(self.trades) + 1,
            "time": datetime.now().isoformat(),
            "symbol": self.symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "FILLED"
        }
        self.trades.append(trade)
        logger.info(f"Trade executed: {trade}")
        return trade
