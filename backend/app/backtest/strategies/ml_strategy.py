import backtrader as bt
import logging

logger = logging.getLogger(__name__)

class MLBasedStrategy(bt.Strategy):
    """
    Strategy that trades based on a pre-computed signal.
    Expects the data feed to have a 'signal' line.
    
    Params:
        threshold (float): Probability threshold for buying.
                           Buy if signal > threshold.
                           Sell if signal < 1 - threshold.
    """
    params = (
        ('threshold', 0.6),
    )

    def __init__(self):
        self.signal = self.data.signal
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                logger.info(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.signal[0] > self.params.threshold:
                logger.info(f'BUY CREATE, Price: {self.data.close[0]:.2f}, Signal: {self.signal[0]:.4f}')
                self.order = self.buy()
        else:
            # Simple exit logic: Close if signal flips or drops below neutral
            # For now, let's just reverse if signal is strong sell
            if self.signal[0] < (1 - self.params.threshold):
                logger.info(f'SELL CREATE, Price: {self.data.close[0]:.2f}, Signal: {self.signal[0]:.4f}')
                self.order = self.close()
