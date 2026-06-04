import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict


class BCPolicy(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs_dict: Dict[str, np.ndarray], device: str = "cpu") -> np.ndarray:
        flat = self._flatten_obs(obs_dict)
        t = torch.FloatTensor(flat).unsqueeze(0).to(device)
        action = self.forward(t)
        return action.squeeze(0).cpu().numpy()

    @staticmethod
    def _flatten_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]])