import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class Actor(nn.Module):
    """Actor network for PPO - outputs mean and std for action distribution."""
    
    def __init__(self, state_dim: int, action_dim: int = 1, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.fc_std.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std of action distribution."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.fc_mean(x))  # Bound to [-1, 1]
        std = F.softplus(self.fc_std(x)) + 1e-5  # Ensure positive std
        
        return mean, std
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy distribution."""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Clamp action to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate action under current policy."""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class Critic(nn.Module):
    """Critic network for PPO - estimates state value."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc_value.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning state value."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_value(x)
        return value


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for continuous action spaces.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = None
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (default 1 for continuous)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: PyTorch device
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        # Training buffers
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Select action given state.
        
        Args:
            state: Current state
            deterministic: If True, return mean action; else sample from distribution
            
        Returns:
            action, log_prob
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = mean
                # For deterministic, we still need a log_prob for compatibility
                _, log_prob = self.actor.get_action(state_tensor)
            else:
                action, log_prob = self.actor.get_action(state_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, 
                        log_prob: float, done: bool):
        """Store transition in buffer."""
        state_value = self.critic(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(state_value.item())
        self.dones.append(done)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Returns:
            advantages, returns
        """
        advantages = []
        returns = []
        
        gae = 0.0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values[step + 1] - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return advantages, returns
    
    def update(self, epochs: int = 10, batch_size: int = 32):
        """
        Update policy using PPO algorithm.
        
        Args:
            epochs: Number of update epochs
            batch_size: Mini-batch size
        """
        if len(self.states) < batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).unsqueeze(1).to(self.device)
        
        # Compute advantages and returns
        next_value = self.critic(torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)).item()
        advantages, returns = self.compute_gae(self.rewards, self.values, self.dones, next_value)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                values = self.critic(batch_states)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Reset buffer after update
        self.reset_buffer()
        
        return {
            'policy_loss': total_policy_loss / (epochs * (len(states) // batch_size + 1)),
            'value_loss': total_value_loss / (epochs * (len(states) // batch_size + 1)),
            'entropy': total_entropy / (epochs * (len(states) // batch_size + 1))
        }
    
    def save(self, filepath: str):
        """Save agent state."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Agent loaded from {filepath}")

