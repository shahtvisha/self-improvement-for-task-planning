"""
diffusion_dataset.py — Action-chunk dataset for Diffusion Policy training.

Standard BC training operates on individual (obs, action) pairs.
Diffusion Policy training operates on (obs, action_CHUNK) pairs where
each chunk is chunk_size consecutive actions from the same episode.

This module wraps a RoboticsDataset and produces the chunk-level samples
needed for diffusion training, while respecting episode boundaries (we
never stitch actions from two different episodes into the same chunk).

Why chunks matter for diffusion
────────────────────────────────
A diffusion model predicts a whole trajectory segment at once. This means:
  1. The model "plans" chunk_size steps coherently rather than greedily
  2. At inference time the robot executes the full chunk before re-querying —
     reducing compounding errors from frequent policy re-calls
  3. The model can learn that "after the approach phase, I must push" as a
     single predicted motion, rather than two separate decisions

Episode boundary tracking
─────────────────────────
RoboticsDataset.add_episode() now records the index of the last transition
in each episode (see dataset.py). DiffusionDataset uses those boundaries
to build a list of valid chunk start indices — indices where a complete
chunk of chunk_size consecutive same-episode actions exists.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List

from dataset import RoboticsDataset


class DiffusionDataset(Dataset):
    """
    Wraps RoboticsDataset → (obs, flat_action_chunk) pairs.

    flat_action_chunk: chunk_size consecutive actions from a single episode,
                       flattened into a 1-D vector of length chunk_size × action_dim.

    Boundary handling: if index i is within chunk_size-1 steps of an episode
    end, it is excluded from valid_indices. We never pad or repeat actions
    across boundaries.
    """

    def __init__(
        self,
        base:       RoboticsDataset,
        chunk_size: int = 4,
        action_dim: int = 4,
    ):
        self.base       = base
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Build set of episode-end indices for O(1) boundary checks
        end_set = set(base._episode_ends)

        # Valid start indices: no episode end within the next chunk_size-1 steps
        n = len(base)
        self.valid_indices: List[int] = []

        for i in range(n - chunk_size + 1):
            crosses_boundary = any(
                (i + j) in end_set for j in range(chunk_size - 1)
            )
            if not crosses_boundary:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        i = self.valid_indices[idx]

        obs = torch.FloatTensor(self.base._obs[i])

        # Concatenate chunk_size consecutive actions
        chunk = np.concatenate(
            [self.base._actions[i + j] for j in range(self.chunk_size)]
        )
        action_chunk = torch.FloatTensor(chunk)

        return obs, action_chunk

    def __repr__(self) -> str:
        return (
            f"DiffusionDataset("
            f"valid_chunks={len(self)}, "
            f"chunk_size={self.chunk_size}, "
            f"base_transitions={len(self.base)})"
        )