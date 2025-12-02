import logging
import os
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from app.ml.rl.env import TradingEnv
from app.ml.rl.agent import DQNAgent
from app.features.registry import FeatureRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "dataset.parquet"
MODEL_DIR = "app/ml/models"

def load_data():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return None
    return pd.read_parquet(DATASET_PATH)

def run_simulation(env, agent=None, model=None, strategy="buy_hold"):
    """
    Run a simulation on the environment.
    strategy: 'rl', 'supervised', 'buy_hold'
    """
    # Force environment to start at the beginning of the provided DF
    obs, _ = env.reset()
    env.current_step = 0 
    env.balance = env.initial_balance
    env.position = 0.0
    env.avg_price = 0.0
    env.steps_taken = 0
    
    done = False
    portfolio_values = []
    
    while not done:
        current_price = env.df.iloc[env.current_step]['close']
        portfolio_value = env.balance + (env.position * current_price)
        portfolio_values.append(portfolio_value)
        
        action = 0 # Hold
        
        if strategy == "rl":
            if agent:
                # Disable exploration
                epsilon_backup = agent.epsilon
                agent.epsilon = 0.0
                action = agent.act(obs)
                agent.epsilon = epsilon_backup
                
        elif strategy == "supervised":
            if model:
                # Extract features from observation (skip balance, pos, avg_price)
                # Obs: [balance, position, avg_price, ...features]
                features = obs[3:].reshape(1, -1)
                
                # LightGBM predict
                prob = model.predict_proba(features)[0][1] # Prob of Up
                
                # Threshold logic
                if prob > 0.6:
                    action = 1 # Buy
                elif prob < 0.4:
                    action = 2 # Sell
                else:
                    action = 0 # Hold
                    
        elif strategy == "buy_hold":
            # Buy at start, hold forever
            if env.current_step == 0:
                action = 1
            else:
                action = 0
        
        obs, reward, done, _, _ = env.step(action)
        
    return portfolio_values

def benchmark():
    df = load_data()
    if df is None:
        return

    # Use the last 20% as test set
    test_size = int(0.2 * len(df))
    test_df = df.iloc[-test_size:].reset_index(drop=True)
    
    feature_cols = FeatureRegistry.get_feature_names()
    env = TradingEnv(test_df, feature_cols)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load RL Agent
    logger.info("Loading RL Agent...")
    agent = DQNAgent(env.input_dim, env.action_space.n, device)
    rl_path = os.path.join(MODEL_DIR, "dqn_agent.pth")
    if os.path.exists(rl_path):
        agent.policy_net.load_state_dict(torch.load(rl_path, map_location=device))
        agent.policy_net.eval()
    else:
        logger.warning("RL Agent not found. Skipping RL benchmark.")
        agent = None

    # 2. Load Supervised Model
    logger.info("Loading Supervised Model...")
    lgbm_path = os.path.join(MODEL_DIR, "lgbm_baseline.joblib")
    if os.path.exists(lgbm_path):
        lgbm = joblib.load(lgbm_path)
    else:
        logger.warning("LightGBM model not found. Skipping Supervised benchmark.")
        lgbm = None
        
    # Run Simulations
    logger.info("Running Buy & Hold...")
    bh_values = run_simulation(env, strategy="buy_hold")
    
    rl_values = []
    if agent:
        logger.info("Running RL Agent...")
        rl_values = run_simulation(env, agent=agent, strategy="rl")
        
    sup_values = []
    if lgbm:
        logger.info("Running Supervised Model...")
        sup_values = run_simulation(env, model=lgbm, strategy="supervised")
        
    # Results
    initial = 10000.0
    
    bh_final = bh_values[-1] if bh_values else initial
    rl_final = rl_values[-1] if rl_values else initial
    sup_final = sup_values[-1] if sup_values else initial
    
    logger.info("-" * 30)
    logger.info("BENCHMARK RESULTS (Test Set)")
    logger.info("-" * 30)
    logger.info(f"Initial Balance: {initial:.2f}")
    logger.info(f"Buy & Hold:      {bh_final:.2f} ({(bh_final-initial)/initial*100:.2f}%)")
    if agent:
        logger.info(f"RL Agent:        {rl_final:.2f} ({(rl_final-initial)/initial*100:.2f}%)")
    if lgbm:
        logger.info(f"Supervised:      {sup_final:.2f} ({(sup_final-initial)/initial*100:.2f}%)")
    logger.info("-" * 30)

if __name__ == "__main__":
    benchmark()
