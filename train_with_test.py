import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from utils import load_model, random_policy, greedy_policy, coord_to_action
from model import StabilizedDuelingDQN, ReplayBuffer
from env import CrossTicTacToe

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
def run_tests(model):
    ai_model = model  # 加载模型
    
    print("\n=== Testing AI vs Random Policy ===")
    random_results, random_game = test_ai_performance(ai_model, random_policy)
    print(f"Win: {random_results['win']}%, Lose: {random_results['lose']}%, Draw: {random_results['draw']}%")
    with open('dqn_reward_training_log.txt', 'a') as f:
        f.write(f"Random: Win: {random_results['win']}%, Lose: {random_results['lose']}%, Draw: {random_results['draw']}%")
    print("\n=== Testing AI vs Greedy Policy ===")
    greedy_results, greedy_game = test_ai_performance(ai_model, greedy_policy)
    print(f"Win: {greedy_results['win']}%, Lose: {greedy_results['lose']}%, Draw: {greedy_results['draw']}%")
    with open('dqn_reward_training_log.txt', 'a') as f:
        f.write(f"Greedy: Win: {greedy_results['win']}%, Lose: {greedy_results['lose']}%, Draw: {greedy_results['draw']}%")
    
    # # 保存对局记录
    # import json
    # with open("ai_vs_random_game.json", "w") as f:
    #     json.dump(random_game[0], f, indent=2)
    # with open("ai_vs_greedy_game.json", "w") as f:
    #     json.dump(greedy_game[0], f, indent=2)
    
    # print("\nSaved game records to JSON files")
    
    return random_results['win'], greedy_results['win']
    






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
GAMMA = 0.995  # 0.9 - 0.999 for gamma, larger for more stable 0.995
EPS_START = 0.9  # random action ratio  0.9
EPS_END = 0.1
EPS_DECAY = 200  # 200 - 1000 for exp decay  200
TARGET_UPDATE = 10  # 10
MEMORY_CAPACITY = 100000  # 100000
LR = 0.05  # 0.03
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


random_win_best, greedy_win_best = 0, 0

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
                
                # clip gradient
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)  # 按范数剪切
                
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
    
    
    if epoch % 10 ==0:
        random_win, greedy_win = run_tests(policy_net)
        if random_win > random_win_best:
            torch.save(policy_net.state_dict(), 'cross_tictactoe_dqn_reward_best.pth')
            random_win_best, greedy_win_best = random_win, greedy_win

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

