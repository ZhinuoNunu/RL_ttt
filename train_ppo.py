import random
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from utils import random_policy, greedy_policy, RewardShaper
from env import CrossTicTacToe
from model import PPONet



# PPO agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.action_dim = len(env.valid_regions)
        self.policy = PPONet(self.action_dim)
        # use a more common initialization strategy
        self._init_weights(self.policy)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-5, eps=1e-5)  # further reduce the learning rate
        self.buffer_size = 2048
        self.buffer = []
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_eps = 0.1
        self.epochs = 3  # reduce the number of training epochs
        self.batch_size = 128
        self.value_coef = 0.25  # reduce the weight of value loss
        self.entropy_coef = 0.02  # increase the entropy regularization coefficient
        self.max_grad_norm = 0.5
        
        # training record
        self.loss_history = {
            'actor': [],
            'critic': [],
            'entropy': [],
            'total': []
        }
        
    def _init_weights(self, module):
        """initialize the network weights, use orthogonal initialization and appropriate gain"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def get_action(self, state, valid_actions):
        # preprocess the state
        state_t = self._preprocess_state(state)
        with torch.no_grad():
            logits, value = self.policy(state_t)
        
        # create action mask
        action_mask = torch.zeros(self.action_dim)
        valid_indices = [self._coord_to_index(a) for a in valid_actions]
        action_mask[valid_indices] = 1.0
        
        # apply mask and sample
        masked_logits = logits + torch.log(action_mask + 1e-10)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action_idx = dist.sample()
        
        return action_idx.item(), dist.log_prob(action_idx), value.squeeze()
    
    def _preprocess_state(self, state):
        """preprocess the state, unify the input format"""
        if isinstance(state, np.ndarray):
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            if state.dim() == 2:
                state_t = state.unsqueeze(0).unsqueeze(0)
            elif state.dim() == 3:
                state_t = state.unsqueeze(0)
            else:
                state_t = state
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
        return state_t
    
    def _coord_to_index(self, coord):
        return self.env.valid_regions.index(coord)
    
    def store_transition(self, transition):
        self.buffer.append(transition)
        
    def compute_advantages(self, rewards, values, dones):
        advantages = []
        last_adv = 0
        next_value = 0
        
        # convert rewards to numpy array for statistical calculations
        rewards_np = np.array(rewards)
        if len(rewards_np) > 10:  # ensure enough samples
            reward_mean = rewards_np.mean()
            reward_std = rewards_np.std()
            # if the reward variance is too high, consider clipping the rewards
            if reward_std > 1.0:
                print(f"Warning: High reward variance {reward_std:.4f}, mean: {reward_mean:.4f}")
                # check for extreme values and handle them
                rewards = [max(min(r, reward_mean + 3*reward_std), reward_mean - 3*reward_std) for r in rewards]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_adv = delta + self.gamma * self.gae_lambda * last_adv * (1 - dones[t])
            advantages.insert(0, last_adv)
            next_value = values[t]
            
        # normalize advantages
        advantages = torch.tensor(advantages)
        if len(advantages) > 1:  # avoid errors when there's only one sample
            # check for extreme values
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            if adv_std > 10.0:
                print(f"Warning: High advantage variance {adv_std:.4f}, mean: {adv_mean:.4f}")
                # apply more robust normalization to avoid extreme values
                advantages = torch.clamp(
                    (advantages - advantages.mean()) / (advantages.std() + 1e-8),
                    min=-3.0, max=3.0
                )
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            # print(f"WARNING: Buffer size {len(self.buffer)} is less than batch size {self.batch_size}, skipping update")
            return
            
        states, actions, old_log_probs, rewards, dones, values = zip(*self.buffer)
        advantages = self.compute_advantages(rewards, values, dones)
        returns = advantages + torch.tensor(values)
        
        # normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        states = torch.FloatTensor(np.array(states)).unsqueeze(1)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # record the baseline KL divergence before update
        old_approx_kl = 0.0
        old_policy_loss = 0.0
        
        # early stopping mechanism parameters
        target_kl = 0.015  # KL divergence threshold
        early_stopping = False
        
        for epoch in range(self.epochs):
            indices = torch.randperm(len(states))
            
            approx_kl_divs = []
            policy_losses = []
            value_losses = []
            entropies = []
            
            for i in range(0, len(states), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                if len(batch_idx) < 8:  # too small batch may lead to instability
                    continue
                    
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]
                
                # forward calculation
                new_logits, new_values = self.policy(b_states)
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                
                # calculate the ratio
                logratio = new_log_probs - b_old_log_probs
                ratio = logratio.exp()
                
                # calculate the approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    approx_kl_divs.append(approx_kl)
                
                # check for abnormal ratio values
                with torch.no_grad():
                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std().item()
                    if ratio_mean > 2.0 or ratio_std > 1.5:
                        print(f"Warning: Ratio mean={ratio_mean:.2f}, std={ratio_std:.2f}")
                
                # PPO loss
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                policy_losses.append(policy_loss.item())
                
                # Critic loss - use Huber loss instead of MSE
                value_pred = new_values.squeeze()
                value_loss = F.smooth_l1_loss(value_pred, b_returns)
                value_losses.append(value_loss.item())
                
                entropies.append(entropy.item())
                
                # total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # calculate the average KL divergence for each epoch
            mean_approx_kl = np.mean(approx_kl_divs) if approx_kl_divs else 0
            mean_policy_loss = np.mean(policy_losses) if policy_losses else 0
            
            # if it's the first epoch, record the baseline KL
            if epoch == 0:
                old_approx_kl = mean_approx_kl
                old_policy_loss = mean_policy_loss
            # check if the KL divergence is too large, if so, stop early
            elif mean_approx_kl > 1.5 * target_kl:
                # print(f"Early stopping at epoch {epoch} due to reaching max kl: {mean_approx_kl:.5f}")
                early_stopping = True
                break
            # check if the policy loss is increasing (training divergence)
            elif mean_policy_loss > old_policy_loss * 1.5:
                # print(f"Early stopping at epoch {epoch} due to increasing policy loss: {mean_policy_loss:.5f} > {old_policy_loss:.5f}")
                early_stopping = True
                break
        
        self.buffer = []
        
        # record the loss values - use the last batch's loss
        self.loss_history['actor'].append(policy_loss.item())
        self.loss_history['critic'].append(value_loss.item())
        self.loss_history['entropy'].append(entropy.item())
        self.loss_history['total'].append(total_loss.item())

# strategy evaluation module
class PolicyEvaluator:
    @staticmethod
    def evaluate(policy_net, env, opponent_policy, num_games=20):
        wins = 0
        for _ in range(num_games):
            state = env.reset()
            done = False
            current_player = 1
            
            while not done:
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                if current_player == 1:  # agent's turn
                    action = PolicyEvaluator.get_policy_action(policy_net, state, valid_actions)
                else:  # opponent's turn
                    action = opponent_policy(env)
                    if action is None:
                        break
                
                next_state, reward, done, _ = env.step(action)
                if done:
                    if reward == 1 and current_player == 1:
                        wins += 1
                current_player = 3 - current_player
                state = next_state
        return wins / num_games

    @staticmethod
    def get_policy_action(net, state, valid_actions):
        state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(state_t)
        
        valid_indices = [env.valid_regions.index(a) for a in valid_actions]
        action_idx = valid_indices[torch.argmax(logits[0, valid_indices]).item()]
        return env.valid_regions[action_idx]


# training function
def train_ppo(env, episodes=1000, resume=False, checkpoint_path=None, use_reward_shaping=True):
    # initialize the reward shaper
    reward_shaper = RewardShaper(env) if use_reward_shaping else None
    
    # attempt to resume training
    start_episode = 0
    if resume and checkpoint_path:
        print(f"Attempting to resume training from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            agent = PPOAgent(env)
            agent.policy.load_state_dict(checkpoint['model_state'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            # load loss history
            agent.loss_history = checkpoint['loss_history']
            # restore training state
            start_episode = checkpoint['episode']
            metrics = checkpoint['metrics']
            all_rewards = checkpoint['all_rewards']
            best_win_rate = checkpoint['best_win_rate']
            print(f"Successfully resumed training, from episode {start_episode}")
        except Exception as e:
            print(f"Failed to resume training: {str(e)}, start training from scratch")
            agent = PPOAgent(env)
            metrics = {
                'random_win_rate': [],
                'greedy_win_rate': [],
                'episodes': [],
                'kl_divs': [],
                'policy_losses': [],
                'value_losses': [],
                'entropies': [],
                'shaped_rewards': [],  # record the shaped reward
                'original_rewards': []  # record the original reward
            }
            all_rewards = []
            best_win_rate = -1.0
    else:
        agent = PPOAgent(env)
        metrics = {
            'random_win_rate': [],
            'greedy_win_rate': [],
            'episodes': [],
            'kl_divs': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'shaped_rewards': [],  # record the shaped reward
            'original_rewards': []  # record the original reward
        }
        all_rewards = []
        best_win_rate = -1.0
    
    win_rates = []
    
    # reward normalizer
    class RunningMeanStd:
        def __init__(self, shape=(), epsilon=1e-4):
            self.mean = np.zeros(shape, dtype=np.float32)
            self.var = np.ones(shape, dtype=np.float32)
            self.count = epsilon
        
        def update(self, x):
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            
            self.update_from_moments(batch_mean, batch_var, batch_count)
        
        def update_from_moments(self, batch_mean, batch_var, batch_count):
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            
            new_mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
            new_var = M2 / total_count
            
            self.mean = new_mean
            self.var = new_var
            self.count = total_count
    
    reward_normalizer = RunningMeanStd()
    
    for ep in range(start_episode, episodes):
        state = env.reset()
        episode_rewards = []
        total_reward = 0
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            action_idx, log_prob, value = agent.get_action(state, valid_actions)
            action = env.valid_regions[action_idx]
            
            next_state, reward, done, _ = env.step(action)
            
            # apply reward shaping
            original_reward = reward
            if use_reward_shaping and reward_shaper:
                reward = reward_shaper.shape_reward(
                    state, action, next_state, done, reward, env.current_player
                )
            
            # collect the original reward and the shaped reward for statistics
            episode_rewards.append(reward)
            total_reward += reward
            
            # store the shaped reward in the buffer
            agent.store_transition((
                state, action_idx, log_prob.item(), reward, done, value.item()
            ))
            
            state = next_state
            
            if len(agent.buffer) >= agent.buffer_size:
                # update reward normalizer
                reward_normalizer.update(np.array(episode_rewards))
                # execute PPO update
                agent.update()
                episode_rewards = []
                # record training metrics
                if agent.loss_history['actor'] and agent.loss_history['critic']:
                    metrics['policy_losses'].append(agent.loss_history['actor'][-1])
                    metrics['value_losses'].append(agent.loss_history['critic'][-1])
                    metrics['entropies'].append(agent.loss_history['entropy'][-1])
        
        all_rewards.append(total_reward)
        
        if ep % 10 == 0 and use_reward_shaping:
            metrics['shaped_rewards'].append(total_reward)
            metrics['original_rewards'].append(original_reward if 'original_reward' in locals() else 0)
        
        # learning rate decay - more gradual decay
        if (ep+1) % 200 == 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.7
            print(f"Learning rate decayed to {agent.optimizer.param_groups[0]['lr']:.6f}")
        
        # save checkpoint periodically
        if (ep+1) % 1000 == 0:
            checkpoint = {
                'episode': ep + 1,
                'model_state': agent.policy.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'loss_history': agent.loss_history,
                'metrics': metrics,
                'all_rewards': all_rewards,
                'best_win_rate': best_win_rate
            }
            torch.save(checkpoint, f'./checkpoints/ppo_checkpoint_ep{ep+1}.pth')
            print(f"Checkpoint saved: ./checkpoints/ppo_checkpoint_ep{ep+1}.pth")
            
        # evaluate the win rate
        if (ep+1) % 50 == 0:
            wins = 0
            for _ in range(20):
                s = env.reset()
                ep_done = False
                while not ep_done:
                    valid = env.get_valid_actions()
                    if not valid:
                        break
                    a_idx, _, _ = agent.get_action(s, valid)
                    a = env.valid_regions[a_idx]
                    s, r, ep_done, _ = env.step(a)
                    if r == 1:
                        wins += 1
            win_rate = wins / 20
            win_rates.append(win_rate)
            
            # evaluate the random strategy opponent
            random_wins = PolicyEvaluator.evaluate(
                agent.policy, env, 
                random_policy
            )
            
            # evaluate the greedy strategy opponent
            greedy_wins = PolicyEvaluator.evaluate(
                agent.policy, env,
                greedy_policy
            )
            
            # calculate the stability index of the strategy
            stability_score = 0
            if len(metrics['greedy_win_rate']) >= 3:
                # calculate the volatility of the last three evaluation win rates
                recent_wins = metrics['greedy_win_rate'][-2:] + [greedy_wins]
                stability_score = np.std(recent_wins)
                
            # win rate and stability evaluation
            if greedy_wins > best_win_rate:
                best_win_rate = greedy_wins
                best_model = agent.policy.state_dict()
                torch.save(best_model, './checkpoints/cross_tictactoe_ppo_best_model.pth')
                print(f"save the new best model, vs Greedy: {greedy_wins:.2f}")
            
            
            metrics['random_win_rate'].append(random_wins)
            metrics['greedy_win_rate'].append(greedy_wins)
            metrics['episodes'].append(ep+1)
            
            # print detailed training diagnostic information
            status_msg = f"Episode {ep+1}: Win Rate: {win_rate:.2f}, " \
                         f"vs Random: {random_wins:.2f}, " \
                         f"vs Greedy: {greedy_wins:.2f}, " \
                         f"Stability: {stability_score:.4f}, " \
                         f"loss: {agent.loss_history['total'][-1]:.4f}, " \
                         f"policy_loss: {agent.loss_history['actor'][-1]:.4f}, " \
                         f"value_loss: {agent.loss_history['critic'][-1]:.4f}, " \
                         f"entropy: {agent.loss_history['entropy'][-1]:.4f}, " \
                         f"total_reward: {total_reward:.2f}"
            
            if use_reward_shaping:
                status_msg += f" (reward shaping enabled)"
                
            print(status_msg)
            
    # visualization module
    plt.style.use('seaborn-v0_8-darkgrid') 
    plt.figure(figsize=(20, 10), facecolor='#f0f0f0')
    
    TECH_BLUE = '#1f77b4'  # tech blue
    TECH_ORANGE = '#ff7f0e'  # tech orange
    TECH_GREEN = '#2ca02c'  # tech green
    TECH_RED = '#d62728'    # tech red
    TECH_PURPLE = '#9467bd' # tech purple
    

    # loss curve
    plt.subplot(2, 2, 1)
    loss_keys = ['critic']
    colors = [TECH_RED, TECH_BLUE, TECH_ORANGE]
    for i in range(len(loss_keys)):
        plt.plot(agent.loss_history[loss_keys[i]], label=loss_keys[i], color=colors[i], linestyle='-', linewidth=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
        
    # entropy curve
    plt.subplot(2, 2, 2)
    plt.plot(agent.loss_history['entropy'], color=TECH_PURPLE, label='Policy Entropy', linewidth=0.5)
    plt.xlabel('Training Steps')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.legend()
        
    # reward curve
    plt.subplot(2, 2, 3)
    window_size = 10
    if len(all_rewards) > window_size:
        smoothed_rewards = np.convolve(all_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_rewards, color=TECH_BLUE, label='Smoothed Reward', linewidth=0.5)
    else:
        plt.plot(all_rewards, color=TECH_BLUE, label='Reward', linewidth=0.5)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
        
    # 胜率曲线
    plt.subplot(2, 2, 4)
    plt.plot(metrics['episodes'], metrics['random_win_rate'], label='vs Random', color=TECH_BLUE, linewidth=0.5)
    plt.plot(metrics['episodes'], metrics['greedy_win_rate'], label='vs Greedy', color=TECH_ORANGE, linewidth=0.5)
    plt.xlabel('Training Episodes')
    plt.ylabel('Win Rate')
    plt.title('Policy Performance')
    plt.legend()
        
    plt.tight_layout()
    plt.savefig('./output/ppo_training_metrics.png', dpi=300)  # 增加DPI获得更高分辨率
    plt.close()
    
    # Save training metrics for later analysis
    import json
    with open('./output/ppo_training_metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            else:
                metrics_json[k] = v
        # Also handle loss history
        loss_history_json = {}
        for k, v in agent.loss_history.items():
            loss_history_json[k] = v
            
        json.dump({
            'metrics': metrics_json,
            'loss_history': loss_history_json,
            'settings': {
                'gamma': agent.gamma,
                'gae_lambda': agent.gae_lambda,
                'clip_eps': agent.clip_eps,
                'value_coef': agent.value_coef,
                'entropy_coef': agent.entropy_coef,
                'learning_rate': agent.optimizer.param_groups[0]['lr'],
                'batch_size': agent.batch_size,
                'epochs': agent.epochs,
                'use_reward_shaping': use_reward_shaping
            }
        }, f, indent=2)
    
    print("Training end, saved metrics to ppo_training_metrics.json")

if __name__ == "__main__":
    print('start training')
    # Set random seeds for reproducibility
    SEED = 42

    random.seed(SEED)  # Python random module
    np.random.seed(SEED)  # Numpy
    torch.manual_seed(SEED)  # PyTorch
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set the number of threads
    torch.set_num_threads(16)
    
    parser = argparse.ArgumentParser(description='Train PPO to solve cross tic-tac-toe')
    parser.add_argument('--resume', action='store_true', help='Whether to resume training from a checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--no_reward_shaping', action='store_true', help='Disable reward shaping')
    args = parser.parse_args()
    
    env = CrossTicTacToe()
    train_ppo(
        env, 
        episodes=args.episodes, 
        resume=args.resume, 
        checkpoint_path=args.checkpoint,
        use_reward_shaping=not args.no_reward_shaping
    )