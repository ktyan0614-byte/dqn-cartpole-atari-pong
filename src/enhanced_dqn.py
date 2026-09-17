# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL - Task 3: Enhanced DQN (Double DQN + PER + Multi-step + Dueling)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


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


# ── Atari Wrappers (CleanRL style) ──────────────────────────────────────────

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        return self._obs_buffer.max(axis=0), total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(float(reward))


def wrap_env(env):
    """Apply standard Atari training wrappers."""
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    return env


def wrap_test_env(env):
    """Wrappers for evaluation — no EpisodicLife, no ClipReward."""
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env


# ── Atari Preprocessor (grayscale + resize + frame stack) ────────────────────

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (Schaul et al., 2016)."""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_steps=2_000_000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        priority = (abs(error) + 1e-5) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        priorities = self.priorities[:N]
        probs = priorities / priorities.sum()
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (N * probs[indices]) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha

    def update_beta(self, step):
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * step / self.beta_steps)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = wrap_env(gym.make(env_name, render_mode="rgb_array"))
        self.test_env = wrap_test_env(gym.make(env_name, render_mode="rgb_array"))
        self.num_actions = self.env.unwrapped.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DuelingDQN(4, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DuelingDQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.n_steps = args.n_steps

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.per = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=args.per_alpha,
            beta_start=args.per_beta,
            beta_steps=args.total_steps
        )
        self.n_step_buffer = deque(maxlen=self.n_steps)

        self.milestone_steps = [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
        self.saved_milestones = set()
        self.reached_19 = False
        self.student_id = "H24111269"
        self.best_model_state = None  # best eval model weights kept in memory

        if args.resume:
            self._load_checkpoint(args.resume)

    def _save_checkpoint(self):
        path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_count": self.env_count,
            "train_count": self.train_count,
            "epsilon": self.epsilon,
            "per_beta": self.per.beta,
            "best_reward": self.best_reward,
            "saved_milestones": list(self.saved_milestones),
            "reached_19": self.reached_19,
        }, path)
        print(f"[Checkpoint] Saved at {self.env_count} steps -> {path}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.env_count = ckpt["env_count"]
        self.train_count = ckpt["train_count"]
        self.epsilon = ckpt["epsilon"]
        self.per.beta = ckpt["per_beta"]
        self.best_reward = ckpt["best_reward"]
        self.saved_milestones = set(ckpt["saved_milestones"])
        self.reached_19 = ckpt["reached_19"]
        best_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(best_path):
            self.best_model_state = torch.load(best_path, map_location=self.device)
        print(f"[Checkpoint] Resumed from {path} at {self.env_count} steps, epsilon={self.epsilon:.4f}")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def _get_n_step_info(self):
        state_0, action_0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        n_step_reward = 0.0
        final_state = self.n_step_buffer[-1][3]
        final_done = False

        for i, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                final_state = ns
                final_done = True
                break

        return state_0, action_0, n_step_reward, final_state, final_done

    def _add_to_per(self, transition):
        max_p = self.per.priorities[:len(self.per.buffer)].max() if len(self.per) > 0 else 1.0
        if len(self.per.buffer) < self.per.capacity:
            self.per.buffer.append(transition)
        else:
            self.per.buffer[self.per.pos] = transition
        self.per.priorities[self.per.pos] = max_p
        self.per.pos = (self.per.pos + 1) % self.per.capacity

    def run(self, episodes=20000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_steps:
                    self._add_to_per(self._get_n_step_info())

                for _ in range(self.train_per_step):
                    self.train()

                for milestone in self.milestone_steps:
                    if milestone not in self.saved_milestones and self.env_count >= milestone:
                        state = self.best_model_state if self.best_model_state is not None else self.q_net.state_dict()
                        model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_{milestone}.pt")
                        torch.save(state, model_path)
                        self.saved_milestones.add(milestone)
                        print(f"Saved milestone at {milestone} steps (best so far) -> {model_path}")

                if self.env_count >= 2_500_000:
                    print(f"Reached 2.5M steps, stopping.")
                    return

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 5000 == 0:
                    wandb.log({
                        "Env Step Count": self.env_count,
                        "Epsilon": self.epsilon,
                        "PER Beta": self.per.beta,
                    })

                if self.env_count % 50000 == 0 and self.env_count > 0:
                    self._save_checkpoint()

            # Flush remaining n-step transitions at episode end
            while len(self.n_step_buffer) > 1:
                self.n_step_buffer.popleft()
                if len(self.n_step_buffer) > 0:
                    self._add_to_per(self._get_n_step_info())

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count
            })

            if ep % 20 == 0:
                eval_reward = self.evaluate()

                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.q_net.state_dict().items()}
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.best_model_state, model_path)
                    print(f"Saved new best model with reward {eval_reward}")

                if eval_reward >= 19 and not self.reached_19:
                    self.reached_19 = True
                    model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_best.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"First reached score 19 at {self.env_count} env steps!")
                    wandb.log({"Steps to Score 19": self.env_count, "Env Step Count": self.env_count})

                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self, num_episodes=5):
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            preprocessor = AtariPreprocessor()
            state = preprocessor.reset(obs)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = preprocessor.step(next_obs)

            total_rewards.append(total_reward)
        return np.mean(total_rewards)

    def train(self):
        if len(self.per) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.per.update_beta(self.train_count)
        self.train_count += 1

        samples, indices, is_weights = self.per.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        is_weights = torch.from_numpy(is_weights).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: q_net selects action, target_net evaluates
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + (self.gamma ** self.n_steps) * next_q_values * (1 - dones)

        td_errors = (targets - q_values).detach().cpu().numpy()
        loss = (is_weights * F.smooth_l1_loss(q_values, targets.detach(), reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        self.per.update_priorities(indices, np.abs(td_errors))

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 5000 == 0:
            wandb.log({
                "Train Loss": loss.item(),
                "Q Mean": q_values.mean().item(),
                "Update Count": self.train_count
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results_task3")
    parser.add_argument("--wandb-run-name", type=str, default="pong-enhanced-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=16000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=27000)
    parser.add_argument("--train-per-step", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--total-steps", type=int, default=2_500_000)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong-Enhanced", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run()
