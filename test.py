from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

from utils import load_model, random_policy, greedy_policy, coord_to_action, visualize_game_animation
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
def run_tests(model_path="cross_tictactoe_dqn_reward_old.pth"):
    # 加载模型
    ai_model = load_model(model_path)
    
    print("\n=== Testing AI vs Random Policy ===")
    random_results, random_game = test_ai_performance(ai_model, random_policy)
    print(f"Win: {random_results['win']}%, Lose: {random_results['lose']}%, Draw: {random_results['draw']}%")
    
    print("\n=== Testing AI vs Greedy Policy ===")
    greedy_results, greedy_game = test_ai_performance(ai_model, greedy_policy)
    print(f"Win: {greedy_results['win']}%, Lose: {greedy_results['lose']}%, Draw: {greedy_results['draw']}%")
    
    visualize_game_animation(random_game[0], 'ai_vs_random.gif')
    visualize_game_animation(greedy_game[0], 'ai_vs_greedy.gif')
    
    # 保存对局记录
    import json
    with open("ai_vs_random_game.json", "w") as f:
        json.dump(random_game[0], f, indent=2)
    with open("ai_vs_greedy_game.json", "w") as f:
        json.dump(greedy_game[0], f, indent=2)
    
    print("\nSaved game records to JSON files")

if __name__ == "__main__":
    run_tests()