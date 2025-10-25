import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt


from Graph import GridWorld
from QNN import NeuralNetwork


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_2d_state_vec(state: Tuple[int, int], rows: int, cols: int) -> np.ndarray:
    """Encode (row, col) as 2 floats normalized to [0,1]."""
    r, c = state
    r_norm = r / max(rows - 1, 1)
    c_norm = c / max(cols - 1, 1)
    return np.array([r_norm, c_norm], dtype=np.float32)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 10000
    min_replay: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000
    target_update_every: int = 500
    max_steps_per_episode: int = 200
    train_episodes: int = 10000
    test_episodes: int = 20
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r.astype(np.float32), ns, d.astype(np.float32)

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, n_actions: int, cfg: DQNConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.n_actions = n_actions

        self.q_net = NeuralNetwork().to(self.device)
        self.target_net = NeuralNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(cfg.replay_size)
        self.global_step = 0
        self.eps_step = 0

    def epsilon(self):
        t = self.eps_step
        start, end, decay = self.cfg.epsilon_start, self.cfg.epsilon_end, self.cfg.epsilon_decay_steps
        return end + (start - end) * max(0.0, (decay - t) / decay)

    @torch.no_grad()
    def act(self, state_vec: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        s = torch.tensor(state_vec[None, :], dtype=torch.float32, device=self.device)
        q = self.q_net(s)
        return int(torch.argmax(q, dim=1).item())

    def push(self, s, a, r, ns, d):
        self.replay.push(s, a, r, ns, d)

    def update(self):
        if len(self.replay) < self.cfg.min_replay:
            return None

        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.long, device=self.device)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device)
        ns_t = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(d, dtype=torch.float32, device=self.device)

        q_values = self.q_net(s_t)                    # [B, 4]
        q_sa = q_values.gather(1, a_t[:, None]).squeeze(1)

        with torch.no_grad():
            target_q = self.target_net(ns_t).max(dim=1).values
            y = r_t + (1.0 - d_t) * self.cfg.gamma * target_q

        loss = self.loss_fn(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1
        self.eps_step += 1
        if self.global_step % self.cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def run_episode(env: GridWorld, agent: DQNAgent, rows: int, cols: int, train: bool, cfg: DQNConfig):
    s = env.reset()
    s_vec = to_2d_state_vec(s, rows, cols)
    total_reward, steps = 0.0, 0
    losses = []

    for _ in range(cfg.max_steps_per_episode):
        a = agent.act(s_vec, explore=train)
        ns, r, done = env.step(a)
        ns_vec = to_2d_state_vec(ns, rows, cols)

        if train:
            agent.push(s_vec, a, r, ns_vec, float(done))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

        total_reward += r
        s_vec = ns_vec
        steps += 1
        if done:
            break

    return total_reward, steps, (np.mean(losses) if losses else None)

def train_and_test(penalty: float, start_state: Tuple[int, int], cfg: DQNConfig):
    env = GridWorld(penalty=penalty, start_state=start_state)
    rows, cols = env.grid.shape
    nA = len(env.actions)

    set_seed(cfg.seed)
    agent = DQNAgent(n_actions=nA, cfg=cfg)

    print("=== Training DQN ===")
    rewards, losses = [], []

    for ep in range(1, cfg.train_episodes + 1):
        ep_ret, ep_steps, ep_loss = run_episode(env, agent, rows, cols, train=True, cfg=cfg)
        rewards.append(ep_ret)
        losses.append(ep_loss if ep_loss is not None else np.nan)

        if ep % 50 == 0:
            avg100 = np.nanmean(rewards[-100:])
            avg_loss = np.nanmean(losses[-50:])
            print(f"Episode {ep:4d}: return={ep_ret:7.3f}  steps={ep_steps:3d}  "
                  f"avg100={avg100:7.3f}  eps={agent.epsilon():.3f}  "
                  f"loss(avg50)={avg_loss:.5f}")

    # === Plot reward & loss ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Reward curve
    ax1.plot(rewards, color="tab:green", alpha=0.7, label="Episode Return")
    if len(rewards) > 10:
        win = 20
        smooth_r = np.convolve(rewards, np.ones(win)/win, mode='valid')
        ax1.plot(range(win-1, len(smooth_r)+win-1), smooth_r, color="tab:blue", linewidth=2, label=f"{win}-ep avg")
    ax1.set_title("DQN Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.legend()

    # Loss curve
    ax2.plot(losses, color="tab:red", alpha=0.6, label="Episode Loss")
    if len(losses) > 10:
        win = 20
        smooth_l = np.convolve([l for l in losses if not np.isnan(l)], np.ones(win)/win, mode='valid')
        ax2.plot(range(win-1, len(smooth_l)+win-1), smooth_l, color="tab:purple", linewidth=2, label=f"{win}-ep avg")
    ax2.set_title("DQN Training Loss")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    print("Saved training curves to training_curves.png")

    # === Testing ===
    print("\n=== Testing (greedy) ===")
    test_returns = []
    for ep in range(1, cfg.test_episodes + 1):
        ep_ret, ep_steps, _ = run_episode(env, agent, rows, cols, train=False, cfg=cfg)
        test_returns.append(ep_ret)
        print(f"[Test] Episode {ep:2d}: return={ep_ret:7.3f}  steps={ep_steps:3d}")

    print(f"\nTest average return over {cfg.test_episodes} episodes: {np.mean(test_returns):.3f}")
    torch.save(agent.q_net.state_dict(), "dqn_gridworld.pt")
    print("Saved model weights and training plots.")




def parse_args():
    p = argparse.ArgumentParser(description="DQN on GridWorld (2D state → 4 actions)")
    p.add_argument("--penalty", type=float, default=-1.0, help="terminal negative reward")
    p.add_argument("--start_r", type=int, default=0, help="start row index")
    p.add_argument("--start_c", type=int, default=0, help="start col index")
    p.add_argument("--episodes", type=int, default=2000, help="training episodes")
    p.add_argument("--test_episodes", type=int, default=20, help="testing episodes")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = DQNConfig(
        lr=args.lr,
        batch_size=args.batch,
        train_episodes=args.episodes,
        test_episodes=args.test_episodes,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_and_test(
        penalty=args.penalty,
        start_state=(args.start_r, args.start_c),
        cfg=cfg
    )
