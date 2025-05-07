import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


# PPO network structure
class PPONet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # shared feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        # Actor branch
        self.actor = nn.Sequential(
            nn.Linear(64*12*12, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        # Critic branch
        self.critic = nn.Sequential(
            nn.Linear(64*12*12, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.actor(x), self.critic(x)


# experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, copy=False)
        next_state = np.array(next_state, copy=False)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

# Dueling DQN
class StabilizedDuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)  # add BN layer
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully connected layer
        self.feature = nn.Linear(64*8*8, 256)
        self.bn_fc = nn.BatchNorm1d(256)  # FC layer BN
        
        # Dueling branch
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),        # use LeakyReLU
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 80)
        )
        
        # add a numerical stability layer before the final output
        self.q_normalizer = nn.LayerNorm(80)  # action dimension 80
        
        # initialize the weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0.1)  # avoid dead neurons

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = x.view(x.size(0), -1)
        # x = self.feature(x)
        # x = F.leaky_relu(self.bn_fc(x), 0.01)
        x = F.leaky_relu(self.feature(x), 0.01)
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return self.q_normalizer(Q) * 10 

