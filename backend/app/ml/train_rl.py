import logging
import os
import pandas as pd
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from typing import List, Tuple
from app.ml.rl.env import TradingEnv
from app.ml.rl.ppo_agent import PPOAgent
from app.ml.rl.window_generator import WindowGenerator
from app.features.registry import FeatureRegistry
from app.ml.dataset import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "dataset.parquet"
MODEL_DIR = "app/ml/models"

# Budget scenarios to train
BUDGET_SCENARIOS = [250.0, 1000.0]

def calculate_sharpe_ratio(returns: List[float]) -> float:
    """Calculate Sharpe ratio from list of returns."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized for 1-min candles

def calculate_sortino_ratio(returns: List[float]) -> float:
    """Calculate Sortino ratio (Sharpe but only penalizes downside volatility)."""
    if len(returns) == 0:
        return 0.0
    returns_array = np.array(returns)
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    downside_std = np.std(downside_returns)
    return np.mean(returns_array) / downside_std * np.sqrt(252 * 24 * 60)

def calculate_profit_factor(returns: List[float]) -> float:
    """Calculate profit factor: total wins / total losses (accounts for magnitude)."""
    if len(returns) == 0:
        return 0.0
    returns_array = np.array(returns)
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    total_wins = np.sum(wins) if len(wins) > 0 else 0.0
    total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    if total_losses == 0:
        return float('inf') if total_wins > 0 else 0.0
    return total_wins / total_losses

def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """Calculate Calmar ratio: annualized return / max drawdown."""
    if len(returns) == 0 or max_drawdown == 0:
        return 0.0
    annualized_return = np.mean(returns) * np.sqrt(252 * 24 * 60)
    return annualized_return / abs(max_drawdown)

def calculate_metrics(episode_returns: List[float], episode_lengths: List[int]) -> dict:
    """Calculate comprehensive performance metrics."""
    if len(episode_returns) == 0:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0
        }
    
    returns_array = np.array(episode_returns)
    
    # Cumulative returns for drawdown calculation
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Win rate
    win_rate = np.sum(returns_array > 0) / len(returns_array)
    
    # Profit factor (accounts for magnitude of wins vs losses)
    profit_factor = calculate_profit_factor(episode_returns)
    
    # Sortino ratio (only penalizes downside volatility)
    sortino_ratio = calculate_sortino_ratio(episode_returns)
    
    # Calmar ratio (return / max drawdown)
    calmar_ratio = calculate_calmar_ratio(episode_returns, max_drawdown)
    
    return {
        'mean_return': float(np.mean(returns_array)),
        'std_return': float(np.std(returns_array)),
        'sharpe_ratio': calculate_sharpe_ratio(episode_returns),
        'sortino_ratio': sortino_ratio,
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.0,
        'calmar_ratio': calmar_ratio
    }

def train_episode(env: TradingEnv, agent: PPOAgent, max_steps: int = 240) -> dict:
    """
    Run a single training episode.
    
    Returns:
        Dictionary with episode statistics
    """
    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    
    for step in range(max_steps):
        # Get action from agent
        action, log_prob = agent.act(state, deterministic=False)
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store transition
        agent.store_transition(state, action, reward, log_prob, done or truncated)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if done or truncated:
            break
    
    # Get final statistics
    stats = env.get_statistics()
    stats['episode_reward'] = episode_reward
    stats['episode_length'] = episode_length
    
    return stats

def train_rl(budget: float, df: pd.DataFrame, feature_cols: List[str], 
             train_windows: List[Tuple[int, int]], val_windows: List[Tuple[int, int]],
             num_episodes: int = 1000, update_frequency: int = 240):
    """
    Train PPO agent for a specific budget scenario.
    
    Args:
        budget: Initial balance for this training run
        df: Full DataFrame with features
        feature_cols: List of feature column names
        train_windows: List of (start_idx, end_idx) tuples for training
        val_windows: List of (start_idx, end_idx) tuples for validation
        num_episodes: Number of training episodes
        update_frequency: Update agent every N steps
    """
    logger.info(f"Training PPO agent with budget: ${budget:,.2f}")
    
    # Create agent
    state_dim = 4 + len(feature_cols)  # balance, position, avg_price, pnl + features
    
    # Device selection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) for acceleration on Apple Silicon")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=1,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device=device
    )
    
    # Training loop
    train_returns = []
    train_lengths = []
    best_val_sharpe = -np.inf
    best_model_path = None
    
    for episode in range(num_episodes):
        # Sample random training window
        window_idx = np.random.randint(0, len(train_windows))
        start_idx, end_idx = train_windows[window_idx]
        window_df = df.iloc[start_idx:end_idx].copy()
        
        # Create environment for this window
        env = TradingEnv(window_df, feature_cols, initial_balance=budget)
        
        # Run episode
        stats = train_episode(env, agent, max_steps=240)
        
        train_returns.append(stats['total_return'])
        train_lengths.append(stats['episode_length'])
        
        # Update agent periodically
        if len(agent.states) >= update_frequency:
            update_stats = agent.update(epochs=10, batch_size=32)
            logger.debug(f"Episode {episode+1}: Updated agent. Loss: {update_stats}")
        
        # Validation and logging
        if (episode + 1) % 100 == 0:
            # Validation
            val_returns = []
            for val_start, val_end in val_windows[:10]:  # Sample 10 validation windows
                val_window_df = df.iloc[val_start:val_end].copy()
                val_env = TradingEnv(val_window_df, feature_cols, initial_balance=budget)
                
                # Run validation episode (deterministic)
                val_state, _ = val_env.reset()
                for _ in range(240):
                    val_action, _ = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, val_truncated, _ = val_env.step(val_action)
                    if val_done or val_truncated:
                        break
                
                val_stats = val_env.get_statistics()
                val_returns.append(val_stats['total_return'])
            
            val_metrics = calculate_metrics(val_returns, [240] * len(val_returns))
            train_metrics = calculate_metrics(train_returns[-100:], train_lengths[-100:])
            
            # Calculate overfitting metric (train_return - val_return)
            overfitting_gap = train_metrics['mean_return'] - val_metrics['mean_return']
            
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Train Return: {train_metrics['mean_return']:.4f} | "
                f"Val Return: {val_metrics['mean_return']:.4f} | "
                f"Val Sharpe: {val_metrics['sharpe_ratio']:.4f} | "
                f"Overfitting Gap: {overfitting_gap:.4f}"
            )
            
            # Log comprehensive metrics to MLflow
            mlflow.log_metrics({
                # Training metrics
                'train_mean_return': train_metrics['mean_return'],
                'train_sharpe': train_metrics['sharpe_ratio'],
                'train_sortino': train_metrics['sortino_ratio'],
                'train_profit_factor': train_metrics['profit_factor'],
                
                # Validation metrics (more reliable)
                'val_mean_return': val_metrics['mean_return'],
                'val_sharpe': val_metrics['sharpe_ratio'],
                'val_sortino': val_metrics['sortino_ratio'],
                'val_max_drawdown': val_metrics['max_drawdown'],
                'val_win_rate': val_metrics['win_rate'],
                'val_profit_factor': val_metrics['profit_factor'],
                'val_calmar_ratio': val_metrics['calmar_ratio'],
                
                # Overfitting detection
                'overfitting_gap': overfitting_gap,
                'train_val_ratio': train_metrics['mean_return'] / val_metrics['mean_return'] if val_metrics['mean_return'] != 0 else 0.0
            }, step=episode)
            
            # Save best model
            if val_metrics['sharpe_ratio'] > best_val_sharpe:
                best_val_sharpe = val_metrics['sharpe_ratio']
                best_model_path = os.path.join(MODEL_DIR, f"ppo_agent_budget_{int(budget)}.pth")
                agent.save(best_model_path)
                logger.info(f"Saved best model with Sharpe: {best_val_sharpe:.4f}")
    
    # Final update
    if len(agent.states) > 0:
        agent.update(epochs=10, batch_size=32)
    
    logger.info(f"Training complete. Best validation Sharpe: {best_val_sharpe:.4f}")
    return best_model_path

def main():
    """Main training function."""
    # MLflow setup
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Deep_RL_Trading")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    logger.info("Loading dataset...")
    df = load_data()
    if df is None:
        logger.error("Failed to load dataset. Run build_dataset.py first.")
        return
    
    # Ensure we have features
    feature_cols = FeatureRegistry.get_feature_names()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return
    
    # Generate windows
    logger.info("Generating trading windows...")
    window_gen = WindowGenerator(window_size=240, step_size=60)
    windows = window_gen.generate_windows(df)
    
    if len(windows) == 0:
        logger.error("No windows generated. Need more data.")
        return
    
    # Split windows
    train_windows, val_windows, test_windows = window_gen.split_windows(windows, train_pct=0.7, val_pct=0.15)
    
    logger.info(f"Training windows: {len(train_windows)}, Validation: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Train for each budget scenario
    for budget in BUDGET_SCENARIOS:
        with mlflow.start_run(run_name=f"PPO_Budget_{int(budget)}"):
            mlflow.log_param("budget", budget)
            mlflow.log_param("window_size", 240)
            mlflow.log_param("step_size", 60)
            mlflow.log_param("num_episodes", 1000)
            
            model_path = train_rl(
                budget=budget,
                df=df,
                feature_cols=feature_cols,
                train_windows=train_windows,
                val_windows=val_windows,
                num_episodes=1000,
                update_frequency=240
            )
            
            if model_path:
                mlflow.log_artifact(model_path)
                logger.info(f"Model logged to MLflow: {model_path}")

if __name__ == "__main__":
    main()
