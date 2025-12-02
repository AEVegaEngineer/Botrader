import backtrader as bt
import logging
from .adapter import TimescaleDBData

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, symbol="BTCUSDT", initial_cash=10000.0, commission=0.001):
        self.cerebro = bt.Cerebro()
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.commission = commission
        
        # Set initial cash
        self.cerebro.broker.setcash(initial_cash)
        
        # Set commission (0.1% for Binance standard)
        self.cerebro.broker.setcommission(commission=commission)
        
        # Set Sizer (Use 10% of portfolio per trade to avoid margin issues)
        self.cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    async def load_data(self):
        logger.info(f"Loading data for {self.symbol}...")
        df = await TimescaleDBData.get_data(self.symbol)
        
        if df.empty:
            logger.warning("No data found for backtest.")
            return False
            
        data = TimescaleDBData(dataname=df)
        self.cerebro.adddata(data)
        return True

    def add_strategy(self, strategy_class, **kwargs):
        self.cerebro.addstrategy(strategy_class, **kwargs)

    def add_analyzers(self):
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    def run(self):
        logger.info("Starting Backtest...")
        logger.info(f"Starting Portfolio Value: {self.cerebro.broker.getvalue():.2f}")
        
        results = self.cerebro.run()
        strat = results[0]
        
        final_value = self.cerebro.broker.getvalue()
        logger.info(f"Final Portfolio Value: {final_value:.2f}")
        logger.info(f"PnL: {final_value - self.initial_cash:.2f}")
        
        # Print Analyzer Results
        self._print_analysis(strat)
        
        return results

    def _print_analysis(self, strat):
        # Sharpe
        sharpe = strat.analyzers.sharpe.get_analysis()
        logger.info(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
        
        # Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        max_dd = drawdown.get('max', {}).get('drawdown', 0)
        logger.info(f"Max Drawdown: {max_dd:.2f}%")
        
        # Trades
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get('total', {}).get('total', 0)
        win_rate = 0
        if total_trades > 0:
            won = trades.get('won', {}).get('total', 0)
            win_rate = (won / total_trades) * 100
            
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
