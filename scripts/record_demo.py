# Records a greedy-policy rollout of the Task 3 (Enhanced DQN) agent on Pong-v5
# and saves it as a GIF for the README. Usage: python scripts/record_demo.py [--seed N]

import argparse
import os
import sys

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from enhanced_dqn import AtariPreprocessor, DuelingDQN

gym.register_envs(ale_py)


def record(model_path, out_path, seed, max_frames, frame_skip, fps, scale, window):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    num_actions = env.action_space.n

    model = DuelingDQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preprocessor = AtariPreprocessor()
    obs, _ = env.reset(seed=seed)
    state = preprocessor.reset(obs)

    frames = []
    rewards = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_frames:
        frames.append(env.render())
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        rewards.append(reward)
        state = preprocessor.step(obs)
        step += 1

    env.close()
    print(f"seed={seed} steps={step} total_reward={total_reward:.0f}")

    # pick the contiguous window with the best net score for the agent (a "highlight reel")
    if window and window < len(frames):
        cumsum = np.cumsum([0.0] + rewards)
        window_gain = cumsum[window:] - cumsum[:-window]
        start = int(np.argmax(window_gain))
        end = start + window
        print(f"highlight window: steps {start}-{end}, net reward {window_gain[start]:.0f}")
        frames = frames[start:end]

    kept = frames[::frame_skip]
    if scale != 1:
        kept = [
            np.array(Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST))
            for f in kept
        ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.mimsave(out_path, kept, fps=fps, loop=0)
    print(f"Saved GIF -> {out_path} ({len(kept)} frames, {kept[0].shape[1]}x{kept[0].shape[0]})")
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/LAB5_H24111269_task3_best.pt")
    parser.add_argument("--out_path", type=str, default="assets/pong_demo.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1200)
    parser.add_argument("--frame_skip", type=int, default=2, help="keep 1 of every N rendered frames")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--scale", type=int, default=3, help="nearest-neighbor upscale factor")
    parser.add_argument("--window", type=int, default=600, help="highlight window length in raw env steps (0 disables)")
    args = parser.parse_args()
    record(
        args.model_path, args.out_path, args.seed, args.max_frames,
        args.frame_skip, args.fps, args.scale, args.window,
    )
