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


    
    
# reward shaper: enhance the sparse reward signal of board games
class RewardShaper:
    def __init__(self, env):
        self.env = env
        self.board_size = env.board_size
        # board center coordinates
        self.center_r, self.center_c = self.board_size // 2, self.board_size // 2
        
    def shape_reward(self, state, action, next_state, done, reward, player):
        """enhance the reward signal based on the game state"""
        if done and reward != 0:  # game over and there is a winner
            # keep the reward unchanged
            return reward
        
        # base reward
        shaped_reward = reward
        
        # 1. position value reward - more important for center and key positions
        r, c = action
        position_value = self._position_value(r, c)
        
        # 2. connection potential evaluation
        connection_value = self._evaluate_connection(next_state, action, player)
        
        # 3. blocking potential evaluation
        blocking_value = self._evaluate_blocking(next_state, action, player)
        
        # 4. area control evaluation
        control_value = self._evaluate_area_control(next_state, player)
        
        # combine the rewards
        shaped_reward += (
            0.01 * position_value +  # position value
            0.03 * connection_value +  # connection potential
            0.02 * blocking_value +  # blocking potential
            0.01 * control_value  # area control
        )
        
        # adjust the reward based on the player
        if player == 2:  # opponent
            shaped_reward = -shaped_reward
            
        return shaped_reward
    
    def _position_value(self, r, c):
        """calculate the strategic value of the position, the center position has higher value"""
        # calculate the distance to the center
        distance_to_center = np.sqrt((r - self.center_r)**2 + (c - self.center_c)**2)
        # the closer to the center, the higher the value, use a non-linear transformation
        return max(0, 1.0 - (distance_to_center / (self.board_size/2)))
    
    def _evaluate_connection(self, board, action, player):
        """evaluate the potential of connecting pieces after the move"""
        r, c = action
        directions = [(0,1), (1,0), (1,1), (1,-1)]  # horizontal, vertical, main diagonal, anti-diagonal
        max_connection = 0
        
        for dr, dc in directions:
            # calculate the number of consecutive pieces in the current direction
            count = 1  # current position
            # check in the forward direction
            for i in range(1, 5):  # check up to 4 steps
                nr, nc = r + i*dr, c + i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                    count += 1
                else:
                    break
            # check in the reverse direction
            for i in range(1, 5):
                nr, nc = r - i*dr, c - i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                    count += 1
                else:
                    break
            
            # update the maximum number of connections
            max_connection = max(max_connection, count)
        
        # map the number of connections to the reward value
        connection_rewards = {
            1: 0.0,   # single block
            2: 0.2,   # two in a row
            3: 0.6,   # three in a row
            4: 1.0    # four in a row/five in a row
        }
        return connection_rewards.get(min(max_connection, 4), 0.0)
    
    def _evaluate_blocking(self, board, action, player):
        """evaluate the value of blocking the opponent's connections"""
        r, c = action
        opponent = 3 - player
        
        # temporarily set the current position to the opponent's piece to evaluate the opponent's connection potential
        temp_board = board.copy()
        temp_board[r, c] = opponent
        
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        max_blocked = 0
        
        for dr, dc in directions:
            # calculate the number of connections the opponent could form in this position
            count = 1
            # check in the forward direction
            for i in range(1, 5):
                nr, nc = r + i*dr, c + i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr, nc] == opponent:
                    count += 1
                else:
                    break
            # check in the reverse direction
            for i in range(1, 5):
                nr, nc = r - i*dr, c - i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and temp_board[nr, nc] == opponent:
                    count += 1
                else:
                    break
            
            max_blocked = max(max_blocked, count)
        
        # map the number of blocked connections to the reward value, blocking more connections has higher rewards
        blocking_rewards = {
            1: 0.0,   # single block
            2: 0.1,   # block two in a row
            3: 0.5,   # block three in a row
            4: 0.9    # block four in a row/five in a row
        }
        return blocking_rewards.get(min(max_blocked, 4), 0.0)
    
    def _evaluate_area_control(self, board, player):
        """evaluate the control of the key regions"""
        # calculate the difference in the number of pieces in each region
        control_score = 0
        
        # center region control
        center_region = board[4:8, 4:8]
        player_count = np.sum(center_region == player)
        opponent_count = np.sum(center_region == 3 - player)
        
        # region control score based on the difference between the player and the opponent
        control_score = 0.2 * (player_count - opponent_count) / 16  # normalize
        
        return max(-1.0, min(1.0, control_score))  # limit the score to [-1,1]