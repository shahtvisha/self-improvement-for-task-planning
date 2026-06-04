"""
dataset.py — Replay buffer / dataset for BC and Diffusion Policy training.

Key addition vs v1: _episode_ends tracks the index of the last transition
in each episode, enabling DiffusionDataset to create valid action chunks
that never cross episode boundaries.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Optional


class Episode:
    """
    A rollout trajectory storing raw components for HER relabeling.
    Keeps obs_body, action, achieved_goal, desired_goal separately
    so that desired_goal can be substituted post-hoc.
    """

    def __init__(self, source: str = "expert", iteration: int = 0):
        self.obs_body_list:      List[np.ndarray] = []
        self.action_list:        List[np.ndarray] = []
        self.achieved_goal_list: List[np.ndarray] = []
        self.desired_goal:       Optional[np.ndarray] = None

        self.success:   bool = False
        self.source:    str  = source
        self.iteration: int  = iteration

    def add_transition(
        self,
        obs_body:      np.ndarray,
        action:        np.ndarray,
        achieved_goal: np.ndarray,
        desired_goal:  Optional[np.ndarray] = None,
    ):
        self.obs_body_list.append(obs_body.copy())
        self.action_list.append(action.copy())
        self.achieved_goal_list.append(achieved_goal.copy())
        if desired_goal is not None and self.desired_goal is None:
            self.desired_goal = desired_goal.copy()

    def finalize(self, success: bool):
        self.success = success

    def flat_obs_iter(self):
        if self.desired_goal is None:
            raise ValueError("desired_goal not set on Episode")
        for obs_body, action in zip(self.obs_body_list, self.action_list):
            yield np.concatenate([obs_body, self.desired_goal]), action

    def __len__(self):
        return len(self.obs_body_list)


class RoboticsDataset(Dataset):
    """
    Growing replay buffer of (flat_obs, action) transitions.

    New field: _episode_ends — list of the index of the last transition
    in each episode added. Used by DiffusionDataset to enforce chunk
    boundaries so action chunks never straddle two episodes.
    """

    def __init__(self):
        self._obs:          List[np.ndarray] = []
        self._actions:      List[np.ndarray] = []
        self._episode_ends: List[int]        = []   # ← NEW: boundary tracking

        self.n_episodes_total:     int = 0
        self.n_episodes_success:   int = 0
        self.n_expert_transitions: int = 0
        self.n_self_transitions:   int = 0
        self.n_her_transitions:    int = 0
        self.n_dagger_transitions: int = 0

    def add_episode(self, episode: Episode, success_only: bool = False) -> bool:
        if success_only and not episode.success:
            return False

        self.n_episodes_total += 1
        if episode.success:
            self.n_episodes_success += 1

        for flat_obs, action in episode.flat_obs_iter():
            self._obs.append(flat_obs)
            self._actions.append(action)
            if episode.source == "expert":
                self.n_expert_transitions += 1
            elif episode.source == "her":
                self.n_her_transitions += 1
            elif episode.source == "dagger":
                self.n_dagger_transitions += 1
            else:
                self.n_self_transitions += 1

        # Record the index of the last transition in this episode
        if len(self._obs) > 0:
            self._episode_ends.append(len(self._obs) - 1)

        return True

    def add_episodes(self, episodes: List[Episode], success_only: bool = False) -> int:
        return sum(self.add_episode(ep, success_only=success_only) for ep in episodes)

    # ── PyTorch Dataset ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self._obs[idx]),
            torch.FloatTensor(self._actions[idx]),
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        if self.n_episodes_total == 0:
            return 0.0
        return self.n_episodes_success / self.n_episodes_total

    @property
    def source_breakdown(self) -> str:
        total = len(self)
        if total == 0:
            return "empty"
        parts = []
        for name, count in [
            ("exp", self.n_expert_transitions),
            ("her", self.n_her_transitions),
            ("self", self.n_self_transitions),
            ("dag", self.n_dagger_transitions),
        ]:
            if count > 0:
                parts.append(f"{name}={count/total:.0%}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"RoboticsDataset("
            f"transitions={len(self)}, "
            f"episodes={self.n_episodes_total}, "
            f"success={self.success_rate:.1%}, "
            f"[{self.source_breakdown}])"
        )