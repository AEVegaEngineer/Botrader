import backtrader as bt
import pandas as pd
import logging
from app.backtest.strategies.base import BaseStrategy
from app.ml.inference import ActionPredictor

logger = logging.getLogger(__name__)

class MLActionTransformerStrategy(BaseStrategy):
    params = (
        ('model_path', 'app/ml/models/action_transformer.pth'),
        ('stats_path', 'dataset_stats.json'),
        ('risk_per_trade', 0.01),
    )

    def __init__(self):
        super().__init__()
        self.predictor = ActionPredictor(
            model_path=self.params.model_path,
            stats_path=self.params.stats_path
        )
        
        # Buffer to hold recent data
        self.min_history = 100 + self.predictor.seq_len
        self.history = []

    def next(self):
        # Append current data to history
        row = {
            'time': self.data.datetime.datetime(0),
            'open': self.data.open[0],
            'high': self.data.high[0],
            'low': self.data.low[0],
            'close': self.data.close[0],
            'volume': self.data.volume[0]
        }
        self.history.append(row)
        
        if len(self.history) < self.min_history:
            return
            
        if len(self.history) > self.min_history + 100:
            self.history = self.history[-self.min_history:]
            
        # Predict
        df = pd.DataFrame(self.history)
        action = self.predictor.predict(df)
            
        # Action Logic
        # 0: HOLD, 1: SELL, 2: BUY
        
        position = self.position.size
        
        if action == 2: # BUY
            if position <= 0:
                if position < 0:
                    self.close() # Close short
                
                # Calculate size
                cash = self.broker.get_cash()
                price = self.data.close[0]
                target_value = self.broker.get_value() * 0.95
                self.order_target_value(target=target_value)
                
        elif action == 1: # SELL
            if position >= 0:
                if position > 0:
                    self.close() # Close long
                
        elif action == 0: # HOLD
            pass
