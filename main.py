import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from Graph import GridWorld
from QNN import NeuralNetwork


# ======== always-on script (no argparse) ========
BASE_EPISODES     = 1000
TEST_EPISODES     = 20
HIDDEN_SIZE       = 128      # QNN.NeuralNetwork(h_size=...)
BASE_LR           = 1e-3
BASE_BATCH        = 64
BASE_GAMMA        = 0.99
PENALTY           = -1.0
START_STATE       = (0, 0)
MAX_STEPS         = 50       # hard cap per episode (as requested)

LR_SWEEP          = [1e-2, 1e-3, 1e-5]
BATCH_SWEEP       = [8, 32, 256]
GAMMA_SWEEP       = [0.99, 0.95, 0.5]
# =================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_2d_state_vec(state: Tuple[int, int], rows: int, cols: int) -> np.ndarray:
    r, c = state
    r_norm = r / max(rows - 1, 1)
    c_norm = c / max(cols - 1, 1)
    return np.array([r_norm, c_norm], dtype=np.float32)


# ---------- plotting helpers (cleaner figs) ----------
def _smooth_ma(y, frac: float = 0.015) -> np.ndarray:
    """Moving average; window = ~1.5% of length, clamped to [5, 200]."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    win = int(max(5, min(200, round(y.size * frac))))
    if win <= 1 or win > y.size:
        return y
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="valid")


def _pretty_axes(ax, title: str, xlabel: str = "Episode", ylabel: str = "Return"):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.8)
# -----------------------------------------------------


@dataclass
class DQNConfig:
    gamma: float = BASE_GAMMA
    lr: float = BASE_LR
    batch_size: int = BASE_BATCH
    replay_size: int = 10000
    min_replay: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000
    target_update_every: int = 500
    max_steps_per_episode: int = MAX_STEPS  # 50-step cap per episode
    train_episodes: int = BASE_EPISODES
    test_episodes: int = TEST_EPISODES
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size: int = HIDDEN_SIZE


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

        # Use your updated QNN with hidden_size
        self.q_net = NeuralNetwork(h_size=cfg.hidden_size).to(self.device)
        self.target_net = NeuralNetwork(h_size=cfg.hidden_size).to(self.device)
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

    rewards: List[float] = []
    losses: List[float] = []

    for ep in range(1, cfg.train_episodes + 1):
        ep_ret, ep_steps, ep_loss = run_episode(env, agent, rows, cols, train=True, cfg=cfg)
        rewards.append(ep_ret)
        losses.append(ep_loss if ep_loss is not None else np.nan)

        if ep % 50 == 0:
            avg100 = np.nanmean(rewards[-100:])
            avg_loss = np.nanmean(losses[-50:])
            print(f"Episode {ep:4d}: return={ep_ret:7.3f}  steps={ep_steps:3d}  "
                  f"avg100={avg100:7.3f}  eps={agent.epsilon():.3f}  loss(avg50)={avg_loss:.5f}")

    # === Clean training curves (smoothed only) ===
    plt.rcParams.update({"figure.dpi": 140})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    r_sm = _smooth_ma(rewards, frac=0.015)
    ax1.plot(range(len(r_sm)), r_sm, linewidth=2, label="Reward (MA)")
    _pretty_axes(ax1, "DQN Episode Rewards (smoothed)")
    ax1.legend()

    losses_clean = [l for l in losses if not (l is None or np.isnan(l))]
    if len(losses_clean) > 0:
        l_sm = _smooth_ma(losses_clean, frac=0.03)
        ax2.plot(range(len(l_sm)), l_sm, linewidth=2, label="Loss (MA)")
    _pretty_axes(ax2, "DQN Training Loss (smoothed)", ylabel="Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", bbox_inches="tight")
    plt.close(fig)

    # Greedy test
    test_returns = []
    for ep in range(1, cfg.test_episodes + 1):
        ep_ret, ep_steps, _ = run_episode(env, agent, rows, cols, train=False, cfg=cfg)
        test_returns.append(ep_ret)
        print(f"[Test] Episode {ep:2d}: return={ep_ret:7.3f}  steps={ep_steps:3d}")
    print(f"\nTest average return over {cfg.test_episodes} episodes: {np.mean(test_returns):.3f}")

    torch.save(agent.q_net.state_dict(), "dqn_gridworld.pt")

    return agent, rewards


@torch.no_grad()
def plot_policy_arrows(agent: DQNAgent, env: GridWorld, filename: str = "policy_arrows.png"):
    """Argmax(Q) arrow in each non-terminal cell; row 0 rendered at bottom."""
    grid = env.grid
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(6, 4))

    base = np.where(np.isnan(grid), np.nan, 0.0)
    ax.imshow(base, cmap=plt.cm.Greys, vmin=0, vmax=1)

    # cell borders
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color='k', linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color='k', linewidth=0.5)

    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if np.isnan(val):
                continue
            cx, cy = (c, r)

            # Terminals: label G / P
            if val == 1.0:
                ax.text(cx, cy, "G", ha='center', va='center', fontsize=16, fontweight='bold', color='green')
                continue
            if val < 0 and val != 0.0:
                ax.text(cx, cy, "P", ha='center', va='center', fontsize=16, fontweight='bold', color='red')
                continue

            # Non-terminal: arrow = argmax Q
            s_vec = to_2d_state_vec((r, c), rows, cols)
            s_t = torch.tensor(s_vec[None, :], dtype=torch.float32, device=agent.device)
            q = agent.q_net(s_t).cpu().numpy().squeeze()
            a = int(np.argmax(q))
            ax.text(cx, cy, arrows[a], ha='center', va='center', fontsize=16)

    ax.set_title("Policy (argmax Q) by cell")
    ax.set_xticks(range(cols)); ax.set_yticks(range(rows))
    # Flip vertical so row 0 is bottom, row (rows-1) is top
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def sweep_and_plot(base_cfg: DQNConfig, penalty: float, start_state: Tuple[int, int],
                   param_name: str, values: Sequence, out_file: str):
    """Run separate trainings for each value, plot ALL reward curves on ONE figure."""
    plt.rcParams.update({"figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    first_agent = None
    for v in values:
        cfg = DQNConfig(**vars(base_cfg))
        setattr(cfg, param_name, v)

        print(f"\n=== SWEEP {param_name}={v} ===")
        agent, rewards = train_and_test(penalty=penalty, start_state=start_state, cfg=cfg)
        if first_agent is None:
            first_agent = agent  # representative for policy snapshot

        # --- Plot only ONE clean line per hyperparameter value ---
        if len(rewards) > 20:
            win = max(5, min(50, len(rewards)//20))
            sm = np.convolve(rewards, np.ones(win)/win, mode='valid')
            ax.plot(np.arange(win, len(rewards)+1), sm, linewidth=2, label=f"{param_name}={v}")
        else:
            ax.plot(np.arange(len(rewards)), rewards, linewidth=2, label=f"{param_name}={v}")

    # Save one policy diagram from the first run
    if first_agent is not None:
        plot_policy_arrows(first_agent, GridWorld(penalty=penalty, start_state=start_state), filename="policy_arrows.png")

    ax.set_title(f"Rewards vs Episode — varying {param_name}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.legend(title=param_name, frameon=True)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)



def main():
    base_cfg = DQNConfig()
    print("\n=== Baseline training (always) ===")
    agent, _ = train_and_test(penalty=PENALTY, start_state=START_STATE, cfg=base_cfg)
    plot_policy_arrows(agent, GridWorld(penalty=PENALTY, start_state=START_STATE))

    # Three consolidated sweep plots (all runs on one figure each)
    sweep_and_plot(base_cfg, PENALTY, START_STATE, "lr",         LR_SWEEP,    "curves_lr.png")
    sweep_and_plot(base_cfg, PENALTY, START_STATE, "batch_size", BATCH_SWEEP, "curves_batch.png")
    sweep_and_plot(base_cfg, PENALTY, START_STATE, "gamma",      GAMMA_SWEEP, "curves_gamma.png")

    print("\nSaved: training_curves.png, policy_arrows.png, curves_lr.png, curves_batch.png, curves_gamma.png")


if __name__ == "__main__":
    main()
