import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Trading environment for Deep RL with continuous action space.
    
    Action Space: Continuous [-1.0, 1.0]
    - -1.0: Sell entire position (100% of holdings)
    - 0.0: Hold (no action)
    - +1.0: Buy with all available cash (100% of balance)
    - Proportional sizing for values in between
    
    State Space: [normalized_balance, position_size, normalized_avg_entry_price, 
                  normalized_unrealized_pnl, ...feature_indicators...]
    """
    
    def __init__(self, df: pd.DataFrame, feature_cols: list, initial_balance: float = 10000.0, 
                 fee_rate: float = 0.001):
        """
        Args:
            df: DataFrame with OHLCV data and features (must be a window slice)
            feature_cols: List of feature column names
            initial_balance: Starting cash balance
            fee_rate: Broker fee rate (default 0.1% = 0.001)
        """
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        
        # Continuous action space: [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State space: [balance_norm, position_size, avg_price_norm, unrealized_pnl_norm, ...features]
        state_dim = 4 + len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position_size = 0.0  # Amount of asset (e.g., BTC)
        self.avg_entry_price = 0.0  # Average entry price for position
        self.current_step = 0
        self.prev_portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Construct observation vector."""
        if self.current_step >= len(self.df):
            # End of episode, return last valid observation
            self.current_step = len(self.df) - 1
        
        row = self.df.iloc[self.current_step]
        current_price = row['close']
        
        # Normalize balance
        balance_norm = self.balance / self.initial_balance
        
        # Position size (in asset units)
        position_size = self.position_size
        
        # Normalized average entry price
        if self.avg_entry_price > 0:
            avg_price_norm = self.avg_entry_price / current_price
        else:
            avg_price_norm = 1.0  # No position, neutral
        
        # Unrealized PnL (normalized)
        unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size if self.avg_entry_price > 0 else 0.0
        unrealized_pnl_norm = unrealized_pnl / self.initial_balance
        
        # Feature indicators (already normalized in dataset)
        features = row[self.feature_cols].values.astype(np.float32)
        
        # Combine into state vector
        state = np.concatenate([
            [balance_norm, position_size, avg_price_norm, unrealized_pnl_norm],
            features
        ]).astype(np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute action and step forward.
        
        Args:
            action: Continuous value in [-1, 1]
                - action > 0: Buy proportionally
                - action < 0: Sell proportionally
                - action == 0: Hold
        
        Returns:
            observation, reward, done, truncated, info
        """
        if self.current_step >= len(self.df) - 1:
            # End of episode
            return self._get_observation(), 0.0, True, False, {}
        
        current_price = self.df.iloc[self.current_step]['close']
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Clamp action to valid range
        action_value = np.clip(action_value, -1.0, 1.0)
        
        # Execute action
        if action_value > 0:  # Buy
            self._execute_buy(action_value, current_price)
        elif action_value < 0:  # Sell
            self._execute_sell(abs(action_value), current_price)
        # else: action == 0, hold (no action)
        
        # Step forward in time
        self.current_step += 1
        
        # Calculate reward (change in portfolio value)
        portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(portfolio_value)
        self.prev_portfolio_value = portfolio_value
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Info dict with trading statistics
        info = {
            'balance': self.balance,
            'position_size': self.position_size,
            'portfolio_value': portfolio_value,
            'unrealized_pnl': (current_price - self.avg_entry_price) * self.position_size if self.avg_entry_price > 0 else 0.0
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_buy(self, action_value: float, current_price: float):
        """Execute buy order proportionally to action value."""
        if self.balance <= 0:
            return
        
        # Calculate target position value (proportional to action)
        max_affordable_value = self.balance / (1 + self.fee_rate)  # Account for fee
        target_value = action_value * max_affordable_value
        
        # Calculate amount to buy
        cost = target_value
        fee = cost * self.fee_rate
        total_cost = cost + fee
        
        if self.balance >= total_cost:
            amount = cost / current_price
            
            # Update position
            if self.position_size > 0:
                # Average entry price calculation
                total_cost_basis = (self.avg_entry_price * self.position_size) + cost
                self.position_size += amount
                self.avg_entry_price = total_cost_basis / self.position_size
            else:
                self.position_size = amount
                self.avg_entry_price = current_price
            
            # Deduct from balance
            self.balance -= total_cost
    
    def _execute_sell(self, action_value: float, current_price: float):
        """Execute sell order proportionally to action value."""
        if self.position_size <= 0:
            return
        
        # Calculate amount to sell (proportional to action)
        sell_amount = action_value * self.position_size
        
        if sell_amount > 0:
            # Calculate revenue
            revenue = sell_amount * current_price
            fee = revenue * self.fee_rate
            net_revenue = revenue - fee
            
            # Update position
            self.position_size -= sell_amount
            
            # If fully sold, reset average entry price
            if self.position_size <= 1e-8:  # Small threshold for floating point
                self.position_size = 0.0
                self.avg_entry_price = 0.0
            
            # Add to balance
            self.balance += net_revenue
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        if self.current_step >= len(self.df):
            current_price = self.df.iloc[-1]['close']
        else:
            current_price = self.df.iloc[self.current_step]['close']
        
        return self.balance + (self.position_size * current_price)
    
    def _calculate_reward(self, current_portfolio_value: float) -> float:
        """
        Calculate reward as normalized change in portfolio value.
        Optionally penalize excessive trading.
        """
        if self.prev_portfolio_value <= 0:
            return 0.0
        
        # Reward is the percentage change in portfolio value
        reward = (current_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        
        # Optional: Small penalty for trading to encourage efficiency
        # This can be tuned or removed
        # if abs(action_value) > 0.01:  # Only penalize significant trades
        #     reward -= 0.0001  # Small penalty
        
        return reward
    
    def get_statistics(self) -> dict:
        """Get current trading statistics."""
        portfolio_value = self._get_portfolio_value()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'position_size': self.position_size,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'steps_taken': self.current_step
        }
