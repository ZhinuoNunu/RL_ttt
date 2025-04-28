from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

# 环境定义
class CrossTicTacToe:
    def __init__(self):
        self.board = np.zeros((12, 12), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.total_regions = range(12)
        self.valid_regions_slice = [
            (slice(0, 4), slice(4, 8)),   # Top
            (slice(8, 12), slice(4, 8)),  # Bottom
            (slice(4, 8), slice(0, 4)),   # Left
            (slice(4, 8), slice(8, 12)),  # Right
            (slice(4, 8), slice(4, 8)),   # Center
        ]
        self.valid_regions = self.get_valid_regions(self.valid_regions_slice)
        self._init_board()

    def get_valid_regions(self, valid_regions_slice):
        valid_regions = []
        for r_slice, c_slice in valid_regions_slice:
            for i in self.total_regions[r_slice]:
                for j in self.total_regions[c_slice]:
                    valid_regions.append([i, j])
        return valid_regions
    
    def _init_board(self):
        self.board.fill(-1)
        for region in self.valid_regions:
            rows, cols = region
            self.board[rows, cols] = 0

    def is_valid_position(self, pos):
        row, col = pos
        if 0 <= row < 12 and 0 <= col < 12:
            tmp = [row, col]
            if tmp in self.valid_regions:
                return True
        return False

    def get_valid_actions(self):
        return [(r, c) for r in range(12) for c in range(12) 
                if self.is_valid_position((r, c)) and self.board[r, c] == 0]

    def check_line(self, player, start_r, start_c, dr, dc, length):
        for i in range(length):
            r = start_r + i*dr
            c = start_c + i*dc
            if not self.is_valid_position((r, c)) or self.board[r, c] != player:
                return False
        return True

    def check_win(self, player):
        cum_reward = 0
        # Check rows and columns for 4 in a row
        for r in range(12):
            for c in range(9):
                if self.check_line(player, r, c, 0, 1, 4):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 0, 1, 3):
                        cum_reward += 0.1
                    
                
        for c in range(12):
            for r in range(9):
                if self.check_line(player, r, c, 1, 0, 4):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, 0, 3):
                        cum_reward += 0.1

        # Check diagonals for 5 in a line
        for r in range(8):
            for c in range(8):
                if self.check_line(player, r, c, 1, 1, 5):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, 1, 4):
                        cum_reward += 0.1
                        
        for r in range(8):
            for c in range(4, 12):
                if self.check_line(player, r, c, 1, -1, 5):
                    return 2, True
                else:
                    if self.check_line(player, r, c, 1, -1, 4):
                        cum_reward += 0.1
                        
        return cum_reward, False

    def step(self, action):
        if self.done:
            return self.board.copy(), 0, True, {}

        # Action processing
        r, c = action
        if not self.is_valid_position(action) or self.board[r, c] != 0:
            self.done = True
            return self.board.copy(), -1, True, {}

        # Placement attempt
        # if np.random.rand() < 0.5:
        if np.random.rand() < 2:
            final_pos = action
        else:
            directions = [(-1,-1), (-1,0), (-1,1),
                        (0,-1), (0,1),
                        (1,-1), (1,0), (1,1)]
            candidates = [(r+dr, c+dc) for dr, dc in directions]
            rand_idx = np.random.randint(0, 16)
            final_pos = candidates[rand_idx%8] if rand_idx < 8 else None

        reward = 0
        if final_pos and self.is_valid_position(final_pos) and self.board[final_pos] == 0:
            self.board[final_pos] = self.current_player
            tmp_reward, tmp_done = self.check_win(self.current_player)
            reward = tmp_reward if self.current_player == 1 else -1*tmp_reward
            if tmp_done:
                self.done = True
                self.winner = self.current_player
            else:
                self.current_player = 3 - self.current_player
        else:
            self.current_player = 3 - self.current_player

        return self.board.copy(), reward, self.done, {}

    def reset(self):
        self._init_board()
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

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

# 测试AI对特定策略的胜率
def test_ai_performance(ai_model, opponent_policy, num_games=100):
    env = CrossTicTacToe()
    results = defaultdict(int)  # 'win', 'lose', 'draw'
    game_records = []
    
    for game_idx in range(num_games):
        state = env.reset()
        record = {"moves": [], "winner": None}
        current_player = 1  # AI始终是player 1
        
        while True:
            if current_player == 1:  # AI的回合
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                    q_values = ai_model(state_t)
                valid_actions = env.get_valid_actions()
                valid_indices = [coord_to_action(*a) for a in valid_actions]
                action_idx = torch.argmax(q_values[0, valid_indices]).item()
                action = valid_actions[action_idx]
            else:  # 对手的回合
                action = opponent_policy(env)
                if action is None:  # 无合法动作
                    break
            
            # 执行动作并记录
            next_state, reward, done, _ = env.step(action)
            record["moves"].append({
                "player": current_player,
                "action": action,
                "board": state.tolist()  # 保存当前棋盘状态
            })
            
            if done:
                record["winner"] = env.winner
                if env.winner == 1:
                    results["win"] += 1
                elif env.winner == 2:
                    results["lose"] += 1
                else:
                    results["draw"] += 1
                break
                
            state = next_state
            current_player = 3 - current_player  # 切换玩家
        
        # 只保存第一局的完整记录
        if game_idx == 0:
            game_records.append(record)
    
    return results, game_records

# 主测试函数
def run_tests(model_path="cross_tictactoe_dqn_reward_old.pth"):
    # 加载模型
    ai_model = load_model(model_path)
    
    print("\n=== Testing AI vs Random Policy ===")
    random_results, random_game = test_ai_performance(ai_model, random_policy)
    print(f"Win: {random_results['win']}%, Lose: {random_results['lose']}%, Draw: {random_results['draw']}%")
    
    print("\n=== Testing AI vs Greedy Policy ===")
    greedy_results, greedy_game = test_ai_performance(ai_model, greedy_policy)
    print(f"Win: {greedy_results['win']}%, Lose: {greedy_results['lose']}%, Draw: {greedy_results['draw']}%")
    
    # 保存对局记录
    import json
    with open("ai_vs_random_game.json", "w") as f:
        json.dump(random_game[0], f, indent=2)
    with open("ai_vs_greedy_game.json", "w") as f:
        json.dump(greedy_game[0], f, indent=2)
    
    print("\nSaved game records to JSON files")

if __name__ == "__main__":
    run_tests()