import backtrader as bt
from .base import BaseStrategy

class RsiSmaStrategy(BaseStrategy):
    """
    Simple strategy:
    - Buy if RSI < 30 AND Price > SMA_200 (Trend filter)
    - Sell if RSI > 70
    """
    params = (
        ('rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('sma_period', 200),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)

    def next(self):
        if not self.position:
            # Buy condition: Oversold in an uptrend
            if self.rsi < self.params.rsi_lower and self.data.close > self.sma:
                self.log(f'BUY SIGNAL: RSI {self.rsi[0]:.2f} < {self.params.rsi_lower} AND Price > SMA')
                self.buy()
        else:
            # Sell condition: Overbought
            if self.rsi > self.params.rsi_upper:
                self.log(f'SELL SIGNAL: RSI {self.rsi[0]:.2f} > {self.params.rsi_upper}')
                self.sell()
