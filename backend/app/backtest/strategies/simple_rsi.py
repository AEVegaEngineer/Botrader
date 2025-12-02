import backtrader as bt
from .base import BaseStrategy

class SimpleRSI(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        if not self.position:
            if self.rsi < self.params.rsi_lower:
                self.log(f'RSI {self.rsi[0]:.2f} < {self.params.rsi_lower} -> BUY CREATE')
                self.buy()
        else:
            if self.rsi > self.params.rsi_upper:
                self.log(f'RSI {self.rsi[0]:.2f} > {self.params.rsi_upper} -> SELL CREATE')
                self.sell()
