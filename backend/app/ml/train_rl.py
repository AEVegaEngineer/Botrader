import logging
import os
import pandas as pd
import torch
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

def train_rl():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    feature_cols = FeatureRegistry.get_feature_names()
    
    # Ensure features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        return
        
    env = TradingEnv(df, feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    agent = DQNAgent(env.input_dim, env.action_space.n, device)
    
    num_episodes = 50
    
    for e in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            
        agent.update_target_network()
        logger.info(f"Episode {e+1}/{num_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.2f}")
        
    # Save Agent
    torch.save(agent.policy_net.state_dict(), os.path.join(MODEL_DIR, "dqn_agent.pth"))
    logger.info("DQN Agent saved.")

if __name__ == "__main__":
    train_rl()
