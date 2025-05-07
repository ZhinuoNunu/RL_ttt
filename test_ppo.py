import numpy as np
import torch
import imageio
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from model import PPONet    
from env import CrossTicTacToe
from utils import random_policy, greedy_policy

def visualize_matchup(model_path, opponent_policy, output_gif="matchup.gif"):
    # initialize the environment and model
    env = CrossTicTacToe()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the trained PPO model
    policy_net = PPONet(len(env.valid_regions)).to(device)
    
    # support two formats: direct model weight and checkpoint format
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        # load from checkpoint
        policy_net.load_state_dict(checkpoint['model_state'])
        print(f"load the model from the checkpoint, training episode: {checkpoint.get('episode', 'unknown')}")
    else:
        # 直接加载模型权重
        policy_net.load_state_dict(checkpoint)
        print("load the model weight directly")
    
    policy_net.eval()
    
    # initialize the record data
    game_record = {
        "moves": [],
        "board_states": [],
        "winner": None
    }
    
    # run the game
    state = env.reset()
    done = False
    game_record["board_states"].append(state.copy())
    
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        # Player 1 (PPO Agent)
        if env.current_player == 1:
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = policy_net(state_t)
            
            # create the action mask
            action_mask = torch.zeros(len(env.valid_regions))
            valid_indices = [env.valid_regions.index(a) for a in valid_actions]
            action_mask[valid_indices] = 1.0
            
            # apply the mask and select the action
            masked_logits = logits + torch.log(action_mask.to(device) + 1e-10)
            action_idx = torch.argmax(masked_logits).item()
            action = env.valid_regions[action_idx]
        else:  # Player 2 (specified policy)
            action = opponent_policy(env)
        
        # execute the action
        next_state, reward, done, _ = env.step(action)
        
        # record the game process
        game_record["moves"].append({
            "player": env.current_player,
            "action": action,
            "reward": reward
        })
        game_record["board_states"].append(next_state.copy())
        
        if done:
            game_record["winner"] = env.winner
        
        state = next_state
    
    # visualization function
    def plot_board_state(board, move_info=None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # draw the board
        for (r, c) in env.valid_regions:
            color = 'lightgray' if board[r, c] == 0 else 'white'
            rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                               facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            
            if board[r, c] == 1:
                plt.text(c, r, 'X', ha='center', va='center', fontsize=30, color='purple')
            elif board[r, c] == 2:
                plt.text(c, r, 'O', ha='center', va='center', fontsize=30, color='blue')
        
        # highlight the latest move
        if move_info:
            r, c = move_info["action"]
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, 
                           fill=False, edgecolor='orange', linewidth=4))
            
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(11.5, -0.5)  # reverse the Y axis
        ax.set_aspect('equal')
        ax.axis('off')
        if move_info:
            if move_info['player'] == 2:
                plt.title(f"Player PPO at {move_info['action']}" if move_info else "Initial Board")
            else:
                plt.title(f"Player specified policy at {move_info['action']}" if move_info else "Initial Board")
        return fig
    
    # generate the GIF
    images = []
    temp_files = []
    
    # initial state
    fig = plot_board_state(game_record["board_states"][0])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    images.append(img)
    plt.close(fig)
    
    # each step state
    for i, move in enumerate(game_record["moves"]):
        fig = plot_board_state(game_record["board_states"][i+1], move)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(img)
        plt.close(fig)
    
    # save the GIF
    imageio.mimsave(output_gif, images, duration=80, palettesize=256)
    
    # add the result page
    result_img = np.zeros((800, 800, 3), dtype=np.uint8) + 255
    fig = plt.figure(figsize=(10, 10))
    plt.text(0.5, 0.6, f"Game Result: {'PPO Wins!' if game_record['winner'] == 1 else 'Opponent Wins!' if game_record['winner'] else 'Draw'}", 
             ha='center', va='center', fontsize=20)
    plt.text(0.5, 0.4, f"Total Moves: {len(game_record['moves'])}", 
             ha='center', va='center', fontsize=16)
    plt.axis('off')
    plt.savefig("result_frame.png")
    plt.close()
    images.append(imageio.imread("result_frame.png"))
    os.remove("result_frame.png")
    
    imageio.mimsave(output_gif, images, duration=800, palettesize=256)
    
    print(f"visualization of the game has been saved to {output_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test the PPO model')
    parser.add_argument('--model', type=str, default="./checkpoints/cross_tictactoe_ppo_best_model.pth", 
                        help='model weight file path')
    parser.add_argument('--output_dir', type=str, default="./output", 
                        help='directory to save the GIF')
    args = parser.parse_args()
    
    # ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # test the game with random policy
    visualize_matchup(
        model_path=args.model,
        opponent_policy=random_policy,
        output_gif=os.path.join(args.output_dir, "vs_random.gif")
    )
    
    # test the game with greedy policy
    visualize_matchup(
        model_path=args.model,
        opponent_policy=greedy_policy,
        output_gif=os.path.join(args.output_dir, "vs_greedy.gif")
    )