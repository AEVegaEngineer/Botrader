import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, input_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        
        self.policy_net = DQN(input_dim, action_dim).to(device)
        self.target_net = DQN(input_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # V(s') = max Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
