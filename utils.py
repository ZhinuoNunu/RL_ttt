import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from model import StabilizedDuelingDQN

# 加载训练好的模型
def load_model(model_path):
    model = StabilizedDuelingDQN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 贪心策略（规则基础AI）
def greedy_policy(env):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None
    
    # 简单规则：优先占据中心区域
    center_actions = [a for a in valid_actions if 5 <= a[0] <= 6 and 5 <= a[1] <= 6]
    if center_actions:
        return random.choice(center_actions)
    return random.choice(valid_actions)

# 随机策略
def random_policy(env):
    valid_actions = env.get_valid_actions()
    return random.choice(valid_actions) if valid_actions else None


# 坐标与动作编号转换
def coord_to_action(r, c):
    if 0 <= r < 4 and 4 <= c < 8:   # Top
        return (r) * 4 + (c-4)
    elif 8 <= r < 12 and 4 <= c < 8: # Bottom
        return 16 + (r-8)*4 + (c-4)
    elif 4 <= r < 8 and 0 <= c < 4:  # Left
        return 32 + (r-4)*4 + c
    elif 4 <= r < 8 and 8 <= c < 12: # Right
        return 48 + (r-4)*4 + (c-8)
    elif 4 <= r < 8 and 4 <= c < 8:  # Center
        return 64 + (r-4)*4 + (c-4)
    else:
        return None