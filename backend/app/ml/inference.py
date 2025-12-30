import torch
import pandas as pd
import numpy as np
import json
import os
import logging
from app.ml.rl.ppo_agent import PPOAgent
from app.features.registry import FeatureRegistry

logger = logging.getLogger(__name__)

class ActionPredictor:
    """
    RL-based action predictor using PPO agent.
    Returns continuous actions in [-1, 1] range.
    """
    
    def __init__(self, model_path='app/ml/models/ppo_agent_budget_10000.pth', 
                 stats_path='dataset_stats.json', budget=10000.0, device=None):
        """
        Args:
            model_path: Path to saved PPO agent checkpoint
            stats_path: Path to dataset normalization stats
            budget: Initial balance (should match training budget)
            device: PyTorch device
        """
        self.model_path = model_path
        self.stats_path = stats_path
        self.budget = budget
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = None
        self.stats = None
        self.feature_cols = None
        
        self.load()
    
    def load(self):
        """Load PPO agent and normalization stats."""
        # Load normalization stats
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                self.stats = json.load(f)
            self.feature_cols = FeatureRegistry.get_feature_names()
        else:
            logger.warning(f"Stats not found at {self.stats_path}, will skip normalization")
            self.stats = None
            self.feature_cols = FeatureRegistry.get_feature_names()
        
        # Load PPO agent
        if not os.path.exists(self.model_path):
            logger.error(f"Model not found at {self.model_path}")
            logger.info("Available models:")
            model_dir = os.path.dirname(self.model_path)
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.startswith("ppo_agent") and f.endswith(".pth"):
                        logger.info(f"  - {os.path.join(model_dir, f)}")
            return
        
        try:
            # Determine state dimension from feature count
            state_dim = 4 + len(self.feature_cols)  # balance, position, avg_price, pnl + features
            
            # Create agent with same architecture as training
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=1,
                device=self.device
            )
            
            # Load checkpoint
            self.agent.load(self.model_path)
            self.agent.actor.eval()
            self.agent.critic.eval()
            
            logger.info(f"PPO agent loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load PPO agent: {e}")
            self.agent = None
    
    def _prepare_state(self, df: pd.DataFrame, balance: float = None, 
                      position_size: float = None, avg_entry_price: float = None) -> np.ndarray:
        """
        Prepare state vector from DataFrame and current trading state.
        
        Args:
            df: DataFrame with OHLCV and features
            balance: Current cash balance (defaults to budget)
            position_size: Current position size (defaults to 0)
            avg_entry_price: Average entry price (defaults to current price)
            
        Returns:
            State vector matching TradingEnv format
        """
        if len(df) == 0:
            logger.warning("Empty DataFrame provided")
            return None
        
        # Use latest row for current state
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Default values
        if balance is None:
            balance = self.budget
        if position_size is None:
            position_size = 0.0
        if avg_entry_price is None or avg_entry_price == 0:
            avg_entry_price = current_price
        
        # Normalize balance
        balance_norm = balance / self.budget
        
        # Position size (in asset units)
        pos_size = position_size
        
        # Normalized average entry price
        if avg_entry_price > 0:
            avg_price_norm = avg_entry_price / current_price
        else:
            avg_price_norm = 1.0
        
        # Unrealized PnL (normalized)
        unrealized_pnl = (current_price - avg_entry_price) * position_size if avg_entry_price > 0 else 0.0
        unrealized_pnl_norm = unrealized_pnl / self.budget
        
        # Get features (should already be computed)
        features = []
        for col in self.feature_cols:
            if col in latest.index:
                val = latest[col]
                # Normalize if stats available
                if self.stats and col in self.stats:
                    mu = self.stats[col]['mean']
                    sigma = self.stats[col]['std']
                    if sigma != 0:
                        val = (val - mu) / sigma
                features.append(float(val))
            else:
                features.append(0.0)
        
        # Combine into state vector
        state = np.concatenate([
            [balance_norm, pos_size, avg_price_norm, unrealized_pnl_norm],
            features
        ]).astype(np.float32)
        
        return state
    
    def predict(self, df: pd.DataFrame, balance: float = None, 
               position_size: float = None, avg_entry_price: float = None) -> float:
        """
        Predict continuous action from DataFrame of history.
        
        Args:
            df: DataFrame with OHLCV columns (should have enough history for indicators)
            balance: Current cash balance
            position_size: Current position size in asset units
            avg_entry_price: Average entry price for current position
            
        Returns:
            Continuous action in [-1, 1]:
            - -1.0: Sell entire position
            - 0.0: Hold
            - +1.0: Buy with all available cash
            - Values in between: Proportional actions
        """
        if self.agent is None:
            logger.warning("Agent not loaded, returning HOLD (0.0)")
            return 0.0
        
        # Compute indicators if not already present
        if not all(col in df.columns for col in self.feature_cols):
            df = FeatureRegistry.compute_all(df.copy())
        
        # Prepare state
        state = self._prepare_state(df, balance, position_size, avg_entry_price)
        if state is None:
            return 0.0
        
        # Get action from agent (deterministic for inference)
        action, _ = self.agent.act(state, deterministic=True)
        
        # Ensure action is scalar
        if isinstance(action, np.ndarray):
            action = float(action[0]) if len(action) > 0 else 0.0
        else:
            action = float(action)
        
        # Clamp to valid range
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def predict_discrete(self, df: pd.DataFrame, balance: float = None,
                       position_size: float = None, avg_entry_price: float = None) -> int:
        """
        Predict discrete action for backward compatibility.
        
        Returns:
            0 (HOLD), 1 (SELL), 2 (BUY)
        """
        continuous_action = self.predict(df, balance, position_size, avg_entry_price)
        
        # Convert continuous to discrete
        if continuous_action > 0.33:  # Strong buy signal
            return 2  # BUY
        elif continuous_action < -0.33:  # Strong sell signal
            return 1  # SELL
        else:
            return 0  # HOLD
