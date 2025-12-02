import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, feature_cols: list, initial_balance=10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # State: [Balance, Position, AvgPrice, ...Features]
        # Features are from the dataframe row
        self.input_dim = 3 + len(feature_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.input_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0 # Amount of asset
        self.avg_price = 0.0
        
        # Pick random start
        if len(self.df) > 1000:
            self.current_step = np.random.randint(0, len(self.df) - 1000)
        else:
            self.current_step = 0
        self.max_steps = 500 # Episode length
        self.steps_taken = 0
        
        return self._get_observation(), {}
        
    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)
        
        obs = np.concatenate(([self.balance, self.position, self.avg_price], features))
        return obs
        
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        done = False
        
        # Execute Action
        if action == 1: # Buy
            if self.balance > 0:
                # Buy with all cash (simplified)
                amount = self.balance / current_price
                cost = amount * current_price
                fee = cost * 0.001 # 0.1% fee
                
                self.position += amount
                self.balance -= (cost + fee)
                self.avg_price = current_price # Simplified avg price update
                
        elif action == 2: # Sell
            if self.position > 0:
                # Sell all
                revenue = self.position * current_price
                fee = revenue * 0.001
                
                self.balance += (revenue - fee)
                self.position = 0.0
                self.avg_price = 0.0
                
        # Calculate Portfolio Value
        portfolio_value = self.balance + (self.position * current_price)
        
        # Reward: Change in Portfolio Value
        # (Ideally log return of portfolio value)
        # Here we use simple PnL change
        # To make it denser, we can use unrealized PnL change
        
        # Step forward
        self.current_step += 1
        self.steps_taken += 1
        
        next_price = self.df.iloc[self.current_step]['close']
        new_portfolio_value = self.balance + (self.position * next_price)
        
        reward = (new_portfolio_value - self.initial_balance) / self.initial_balance # Normalized Total Return so far? 
        # Better: Step reward = Change in value
        reward = (new_portfolio_value - portfolio_value) / portfolio_value
        
        if self.steps_taken >= self.max_steps or self.current_step >= len(self.df) - 1:
            done = True
            
        return self._get_observation(), reward, done, False, {}
