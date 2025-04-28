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


# Set random seeds for reproducibility
SEED = 42

# Python random module
random.seed(SEED)

# Numpy
np.random.seed(SEED)

# PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 训练参数
BATCH_SIZE = 256
GAMMA = 0.99  # 0.9 - 0.999 for gamma, larger for more stable 0.995
EPS_START = 0.95  # random action ratio  0.9
EPS_END = 0.1
EPS_DECAY = 400  # 200 - 1000 for exp decay  200
TARGET_UPDATE = 5  # 10
MEMORY_CAPACITY = 50000  # 100000
LR = 0.1  # 0.03
NUM_EPOCHS = 3000  # 4000

# 初始化
env = CrossTicTacToe()
# policy_net = DQN()
# target_net = DQN()
policy_net = StabilizedDuelingDQN()
target_net = StabilizedDuelingDQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR, eps=1e-5)  # 更稳定的eps
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20)
memory = ReplayBuffer(MEMORY_CAPACITY)
epsilon = EPS_START

# 训练循环
rewards_history = []
loss_history = []

with open('dqn_reward_training_log.txt', 'w') as f:
    f.write(f"epoch, reward, loss\n")

writer = SummaryWriter()

for epoch in range(NUM_EPOCHS):
    epoch_rewards = []
    epoch_losses = []
    
    for _ in range(10):  # 10 episodes per epoch
        state = env.reset()
        total_reward = 0
        while True:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    q_values = policy_net(state_t)
                valid_indices = [coord_to_action(*a) for a in valid_actions]
                action_idx = torch.argmax(q_values[0, valid_indices]).item()
                action = valid_actions[action_idx]

            # Execute action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition
            action_idx = coord_to_action(*action)
            memory.push(state, action_idx, reward, next_state, done)
            
            # Train
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                states = torch.FloatTensor(states).unsqueeze(1)
                next_states = torch.FloatTensor(next_states).unsqueeze(1)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)
                
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                
                next_q = target_net(next_states).max(1)[0].detach()
                target_q = rewards + (1 - dones) * GAMMA * next_q
                
                target_q = torch.clamp(target_q, min=-1e3, max=1e3)  # 根据环境奖励范围调整

                
                # loss = F.mse_loss(current_q, target_q)
                loss = F.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                
                # for name, param in policy_net.named_parameters():
                #     if param.grad is not None:
                #         grad_norm = param.grad.data.norm(2).item()
                #         print('grad_norm before: ', grad_norm)
                
                # clip gradient
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)  # 按范数剪切
                
            
                # for name, param in policy_net.named_parameters():
                #     if param.grad is not None:
                #         grad_norm = param.grad.data.norm(2).item()
                #         print('grad_norm after: ', grad_norm)
                
                
                optimizer.step()
                scheduler.step(loss)
                epoch_losses.append(loss.item())
                
            if done:
                break
            state = next_state
        
        epoch_rewards.append(total_reward)
        # epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-epoch / EPS_DECAY)
        EPS_DECAY = 500  # 从200增加到500
        epsilon = max(EPS_END, EPS_START * (0.998 ** epoch))  # 指数衰减更平滑
    
    # 记录指标
    avg_reward = np.mean(epoch_rewards)
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0
    rewards_history.append(avg_reward)
    loss_history.append(avg_loss)
    with open('dqn_reward_training_log.txt', 'a') as f:
        f.write(f"{epoch+1}, {avg_reward:.2f}, {avg_loss:.4f}\n")
        
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Reward: {avg_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")
    
    # 更新目标网络
    # if epoch % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())
    
    TAU = 0.005
    # 每次训练后执行软更新
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU*policy_param.data + (1-TAU)*target_param.data)

# 设置科技感风格（深色背景+高对比色）
plt.style.use('seaborn-v0_8-darkgrid')  # 使用深色网格背景增强科技感

# 创建画布
plt.figure(figsize=(12, 6), facecolor='#f0f0f0')  # 浅灰背景避免刺眼

# 定义科技感配色
TECH_BLUE = '#1f77b4'  # 科技蓝 (类似Azure/AWS蓝)
TECH_ORANGE = '#ff7f0e'  # 活力橙 (类似Python橙)

# 左侧轴：奖励曲线（科技蓝）
ax1 = plt.gca()
line1 = ax1.plot(
    rewards_history, 
    color=TECH_BLUE, 
    linestyle='-', 
    linewidth=1,  # 加粗线条
    marker='',   # 无标记点
    label='Average Reward'
)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Reward', color=TECH_BLUE, fontsize=12)
ax1.tick_params(axis='y', labelcolor=TECH_BLUE)
ax1.grid(True, alpha=0.3)  # 半透明网格线

# 右侧轴：损失曲线（活力橙）
ax2 = ax1.twinx()
line2 = ax2.plot(
    loss_history, 
    color=TECH_ORANGE, 
    linestyle='-',  # 虚线增强区分度
    linewidth=1,
    marker='',
    label='Average Loss'
)
ax2.set_ylabel('Loss', color=TECH_ORANGE, fontsize=12)
ax2.tick_params(axis='y', labelcolor=TECH_ORANGE)

# 合并图例（统一放置在右上角）
lines = line1 + line2
labels = [l.get_label() for l in lines]
plt.legend(
    lines, labels, 
    loc='upper right', 
    framealpha=1,  # 不透明图例框
    edgecolor='#333333'  # 深色边框
)

# 标题和保存
plt.title('DuelingDQN Training Progress', 
          fontsize=14, 
          pad=20,  # 增加标题与图的间距
          color='#333333')  # 深灰标题
plt.tight_layout()  # 自动调整布局
plt.savefig('training_results_dqn_reward.png', dpi=300, bbox_inches='tight')  # 高分辨率保存

# 保存模型
torch.save(policy_net.state_dict(), 'cross_tictactoe_dqn_reward.pth')