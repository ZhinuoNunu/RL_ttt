import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque

# 经验回放
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

# Dueling DQN网络定义
class StabilizedDuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)  # 添加BN层
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 全连接层
        self.feature = nn.Linear(64*8*8, 512)
        self.bn_fc = nn.BatchNorm1d(512)  # FC层BN
        
        # Dueling分支
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),        # 改用LeakyReLU
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 80)
        )
        
        # 在最后输出前添加数值稳定层
        self.q_normalizer = nn.LayerNorm(80)  # 动作维度80
        
        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0.1)  # 避免死神经元

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


# # DQN网络定义
# class DQN(nn.Module):
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.fc1 = nn.Linear(64*8*8, 512)
#         self.fc2 = nn.Linear(512, 80)  # 80 possible actions

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)