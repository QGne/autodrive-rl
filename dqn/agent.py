# dqn/agent.py

from collections import deque
import random
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)
       
        
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            np.array, zip(*batch)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        target_update_freq: int = 1000,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=replay_capacity)
        self.loss_fn = nn.MSELoss()

        self.total_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        ε-greedy action selection.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        """
        One gradient update step from replay buffer.
        Returns loss value or None if not enough data yet.
        """
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        states_t = torch.tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions_t = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(
            dones.astype(np.float32), dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # Q(s, a)
        q_values = self.q_net(states_t).gather(1, actions_t)

        # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(
                dim=1, keepdim=True
            )[0]
            target_q = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.update_target()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
