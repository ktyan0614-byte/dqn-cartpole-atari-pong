# Spring 2026, 535518 Deep Learning
# Lab5: Task 3 evaluation script
# Usage: python test_model_task3.py --model_path <path> [--env_steps <N>]

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import cv2
import ale_py
import argparse
from collections import deque

gym.register_envs(ale_py)


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DuelingDQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        features = self.features(x / 255.0)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DuelingDQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    rewards = []
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)
        print(f"Environment steps: {args.env_steps}, seed: {seed}, eval reward: {int(total_reward)}")

    env.close()
    print(f"Average reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_steps", type=int, default=0, help="Training steps (for display only)")
    args = parser.parse_args()
    evaluate(args)
