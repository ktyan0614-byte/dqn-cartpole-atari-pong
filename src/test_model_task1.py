import torch
import gymnasium as gym
import numpy as np
import argparse
from dqn import DQN  # 確保從你的 dqn.py 匯入模型類別

def test(model_path):
    # 1. 初始化環境
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    num_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 載入模型
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切換至評估模式

    total_rewards = []

    # 3. 連續測試 20 個 Episode (使用 seed 0 ~ 19)
    for i in range(20):
        obs, info = env.reset(seed=i) # 固定種子 
        state = obs # CartPole 不需要特別 preprocessor
        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            obs, reward, done, truncated, info = env.step(action)
            state = obs
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {i}: Reward = {episode_reward}")

    # 4. 計算平均
    mean_reward = np.mean(total_rewards)
    print("-" * 30)
    print(f"Average Reward over 20 episodes: {mean_reward:.2f}")
    
    if mean_reward >= 480:
        print("Status: FULL SCORE (15%) ACHIEVED!")
    else:
        print(f"Status: Estimated Score = {(min(mean_reward, 480) / 480 * 15):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    test(args.model_path)