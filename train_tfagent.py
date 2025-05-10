import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

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

# 环境定义
class CrossTicTacToePyEnv(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self.board_size = 12
        self.valid_regions = self._get_valid_regions()
        
        # 动作空间定义为全部可能位置
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(self.valid_regions)-1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12, 12, 2), dtype=np.int32, minimum=0, maximum=1, name='observation')
        
        # 无效动作计数器（专门针对已有棋子的格子）
        self._occupied_position_count = 0
        self._state = None
        self._current_player = 1
        self._done = False
        self._winner = None
        self._valid_actions = None

    def _get_valid_regions(self):
        regions = [
            (slice(0, 4), slice(4, 8)),   # top
            (slice(8, 12), slice(4, 8)),  # bottom
            (slice(4, 8), slice(0, 4)),   # left
            (slice(4, 8), slice(8, 12)),  # right
            (slice(4, 8), slice(4, 8)),   # center
        ]
        valid = []
        for r_slice, c_slice in regions:
            for r in range(r_slice.start, r_slice.stop):
                for c in range(c_slice.start, c_slice.stop):
                    valid.append((r, c))
        return valid  # 假设返回的valid_regions是坐标列表

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((self.board_size, self.board_size), dtype=np.int32) - 1
        for (r, c) in self.valid_regions:
            self._state[r, c] = 0
        self._current_player = 1
        self._done = False
        self._winner = None
        self._valid_actions = self._get_current_valid_actions()
        self._occupied_position_count = 0  # 重置无效动作计数
        
        # 创建增强观察，包含有效动作掩码
        observation = self._get_observation()
        return ts.restart(observation)

    def _get_observation(self):
        # 创建一个包含棋盘状态和有效动作掩码的观察
        # 通道0: 棋盘状态
        # 通道1: 有效动作掩码（1表示有效，0表示无效）
        board_state = self._state.copy()  # 棋盘状态
        
        # 创建掩码层
        valid_mask = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        for i in self._valid_actions:
            r, c = self.valid_regions[i]
            valid_mask[r, c] = 1
            
        # 组合成多通道观察
        observation = np.stack([board_state, valid_mask], axis=2)
        return observation

    def _step(self, action):
        if self._done:
            return self._reset()

        # 检查是否在有效动作内
        if action not in self._valid_actions:
            # 检查是否是尝试落在已有棋子的位置
            r, c = self.valid_regions[action]
            if (r, c) in self.valid_regions and self._state[r, c] != 0:
                # 尝试下在已经有棋子的地方
                self._occupied_position_count += 1
                
                # 如果连续多次尝试下在已有棋子的位置，则终止游戏
                if self._occupied_position_count >= 5:
                    self._done = True
                    return ts.termination(self._get_observation(), reward=-1.0)
            
            # 其他无效动作（如边界外）不终止，只给予惩罚
            return ts.transition(self._get_observation(), reward=-0.5, discount=1.0)
            
        # 重置无效动作计数
        self._occupied_position_count = 0
        
        # 获取玩家选择的坐标
        selected_coord = self.valid_regions[action]
        r, c = selected_coord
        
        # 概率放置机制
        place_success = np.random.random() < 0.5  # 50%概率成功放置
        
        if not place_success:
            # 随机选择周围8个相邻位置之一
            offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
            random_offset = random.choice(offsets)
            new_r = r + random_offset[0]
            new_c = c + random_offset[1]
            
            # 检查是否在边界内且位置空闲
            if (0 <= new_r < self.board_size and 0 <= new_c < self.board_size and 
                (new_r, new_c) in self.valid_regions and self._state[new_r, new_c] == 0):
                r, c = new_r, new_c
            else:
                # 放置失败，跳过回合
                self._current_player = 3 - self._current_player
                self._valid_actions = self._get_current_valid_actions()
                
                # 检查游戏是否应该结束（无处可下）
                if not self._valid_actions:
                    self._done = True
                    return ts.termination(self._get_observation(), reward=0.0)  # 平局
                    
                return ts.transition(self._get_observation(), reward=-0.01, discount=1.0)
        
        # 执行放置动作
        self._state[r, c] = self._current_player
        
        # 检查胜利条件
        if self._check_win(self._current_player):
            reward = 1.0 if self._current_player == 1 else -1.0
            self._done = True
            self._winner = self._current_player
        else:
            reward = -0.01  # 步惩罚

        self._current_player = 3 - self._current_player
        self._valid_actions = self._get_current_valid_actions()
        
        # 检查是否还有有效动作
        if not self._valid_actions and not self._done:
            self._done = True
            return ts.termination(self._get_observation(), reward=0.0)  # 平局，没有赢家
        
        # 返回前生成新的观察
        observation = self._get_observation()
        
        if self._done:
            return ts.termination(observation, reward)
        else:
            return ts.transition(observation, reward=reward, discount=1.0)

    def _get_current_valid_actions(self):
        return [i for i, (r, c) in enumerate(self.valid_regions) if self._state[r, c] == 0]

    def _check_win(self, player):
        # check the 4-in-a-row win condition
        directions_hv = [(0,1), (1,0)]
        directions_diag = [(1,1), (1,-1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in directions_hv:
                    if all(0 <= r+i*dr < self.board_size and 
                           0 <= c+i*dc < self.board_size and
                           self._state[r+i*dr][c+i*dc] == player
                           for i in range(4)):
                        return True
                    
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dr, dc in directions_diag:
                    if all(0 <= r+i*dr < self.board_size and 
                           0 <= c+i*dc < self.board_size and
                           self._state[r+i*dr][c+i*dc] == player
                           for i in range(5)):
                        return True
        return False

# 奖励塑形（集成到环境）
class ShapedCrossTicTacToePyEnv(CrossTicTacToePyEnv):
    def __init__(self):
        super().__init__()
        # 因为 RewardShaper 被注释掉了，所以这里我们使用一个简单的替代方案
        # self.reward_shaper = RewardShaper(env=self)
        self.use_shaped_rewards = True
        
    def _step(self, action):
        original_state = self._state.copy()
        time_step = super()._step(action)
        
        # 简单的奖励塑形：如果游戏继续，给予小的正奖励
        shaped_reward = time_step.reward
        if not time_step.is_last() and self.use_shaped_rewards:
            shaped_reward += 0.01  # 鼓励探索
        
        return ts.TimeStep(
            step_type=time_step.step_type,
            reward=shaped_reward,
            discount=time_step.discount,
            observation=time_step.observation
        )

# 简化的网络创建函数
def create_networks(observation_spec, action_spec):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=(256, 128),
        activation_fn=tf.keras.activations.relu)
    
    value_net = value_network.ValueNetwork(
        observation_spec,
        fc_layer_params=(256, 128),
        activation_fn=tf.keras.activations.relu)
    
    return actor_net, value_net

def train_ppo_agent(env_name='shaped', num_iterations=10000):
    # 创建环境
    if env_name == 'shaped':
        py_env = ShapedCrossTicTacToePyEnv()
    else:
        py_env = CrossTicTacToePyEnv()
    
    train_env = tf_py_environment.TFPyEnvironment(py_env)
    eval_env = tf_py_environment.TFPyEnvironment(CrossTicTacToePyEnv())
    
    # 创建网络
    actor_net, value_net = create_networks(
        train_env.observation_spec(), 
        train_env.action_spec())
    
    # 创建优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    
    # 创建PPO Agent
    train_step_counter = tf.Variable(0)
    agent = ppo_clip_agent.PPOClipAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.01,
        importance_ratio_clipping=0.2,
        discount_factor=0.99,
        normalize_rewards=True,
        normalize_observations=True,
        use_gae=True,
        num_epochs=10,
        debug_summaries=True,
        summarize_grads_and_vars=False,
        train_step_counter=train_step_counter
    )
    agent.initialize()
    
    # 指标
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=10),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=10)
    ]
    
    # 定义策略评估函数
    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):
            print("开始评估...", num_episodes)
            time_step = environment.reset()
            episode_return = 0.0
            
            cnt = 0
            while not time_step.is_last():
                cnt += 1
                if cnt % 10 == 0:
                    print("评估中...", cnt)
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            
            total_return += episode_return
            
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    # 收集数据
    def collect_episode(environment, policy, num_episodes=1):
        driver = dynamic_step_driver.DynamicStepDriver(
            environment,
            policy,
            observers=[replay_buffer.add_batch],
            num_steps=200)  # 设置一个较大的步数，足够完成一个回合
        
        for _ in range(num_episodes):
            time_step = environment.reset()
            driver.run(time_step)
    
    # 创建回放缓冲区
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=1,
        max_length=1000)
    
    # 训练
    returns = []
    print("开始训练...")
    
    for iteration in range(num_iterations):
        try:
            # 收集数据
            start_time = time.time()
            collect_episode(train_env, agent.collect_policy, num_episodes=5)
            collect_time = time.time() - start_time
            
            # 从缓冲区获取数据
            experience = replay_buffer.gather_all()
            
            # 训练
            start_time = time.time()
            train_loss = agent.train(experience)
            train_time = time.time() - start_time
            
            replay_buffer.clear()  # 清除缓冲区
            
            if iteration % 2 == 0:
                # 评估
                avg_return = compute_avg_return(eval_env, agent.policy)
                print(f"迭代 {iteration} | avg_return: {avg_return:.2f} | 数据收集: {collect_time:.2f}s | 训练: {train_time:.2f}s")
                returns.append(avg_return)
                
        except Exception as e:
            print(f"迭代 {iteration} 出错: {e}")
            continue
    
    # 保存模型
    policy_saver.PolicySaver(agent.policy).save('saved_policy')
    
    # 绘制训练曲线
    iterations = range(0, num_iterations, 50)
    plt.figure(figsize=(12, 8))
    plt.plot(iterations[:len(returns)], returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.title('PPO Training Performance')
    plt.savefig('ppo_training_curve.png')
    
    print("训练完成!")
    return agent

if __name__ == "__main__":
    # 设置随机种子
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='shaped', 
                      help='Environment type: "shaped" or "original"')
    parser.add_argument('--iter', type=int, default=10000, 
                      help='Number of training iterations')
    args = parser.parse_args()
    
    # 开始训练
    train_ppo_agent(env_name=args.env, num_iterations=args.iter)