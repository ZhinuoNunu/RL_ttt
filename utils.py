import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import imageio
from model import StabilizedDuelingDQN


def visualize_game_animation(game_record, filename='game_animation.gif'):
    images = []
    
    for move_idx, move in enumerate(game_record['moves']):
        board = np.array(move['board'])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')
        
        # 绘制棋盘结构
        for i in range(12):
            for j in range(12):
                # 只绘制有效区域（十字形）
                if board[i][j] != -1:
                    # 绘制棋盘格子
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                      fill=False, ec='black', lw=0.8)
                    ax.add_patch(rect)
                    
                    # 绘制棋子
                    if board[i][j] == 1:  # 圆圈（Player 1）
                        circle = plt.Circle((j, i), 0.35,
                                         ec='blue', lw=3, fill=False)
                        ax.add_patch(circle)
                    elif board[i][j] == 2:  # 叉号（Player 2）
                        ax.plot([j-0.3, j+0.3], [i-0.3, i+0.3], 
                              'r-', lw=3)
                        ax.plot([j-0.3, j+0.3], [i+0.3, i-0.3], 
                              'r-', lw=3)

        # 设置坐标轴范围和样式
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(11.5, -0.5)  # 反转Y轴
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 高亮当前落子位置
        current_action = move['action']
        if current_action:
            ay, ax_pos = current_action  # 注意坐标顺序转换
            highlight = plt.Rectangle((ax_pos-0.5, ay-0.5), 1, 1,
                                    fill=False, ec='gold', lw=3)
            ax.add_patch(highlight)
        
        # 添加标题信息
        title_text = f"Move {move_idx+1}\nPlayer {move['player']} at {current_action}"
        plt.title(title_text, fontsize=12, pad=20)
        
        # 保存临时图片
        temp_file = f"temp_{move_idx}.png"
        plt.savefig(temp_file, bbox_inches='tight', dpi=100)
        plt.close()
        images.append(imageio.imread(temp_file))
        os.remove(temp_file)
    
    # 生成GIF
    imageio.mimsave(filename, images, duration=1)
    print(f"Saved game animation to {filename}")


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