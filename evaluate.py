"""
evaluate.py — Policy evaluation and rollout collection.

Updated to support chunked action execution for DiffusionPolicy:
  - If the policy has act_chunk() and .chunk_size, execute the full chunk
    before re-querying (action chunking)
  - Otherwise, fall back to single-step execution (BCPolicy)

Action chunking in rollouts
────────────────────────────
With chunk_size=4 the robot:
  1. Queries policy once → gets 4 planned actions
  2. Executes all 4 in sequence without re-querying
  3. Queries again for the next 4
This reduces compounding errors from 50 sequential decisions to ~12,
and forces coherent sub-trajectories (approach, then push).
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import imageio
from typing import List, Tuple

from policy  import BCPolicy
from dataset import Episode


# ── Single-step or chunked execution ─────────────────────────────────────────

def _is_diffusion(policy) -> bool:
    return hasattr(policy, "act_chunk") and hasattr(policy, "chunk_size")


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_policy(
    policy,
    env_id:          str,
    n_episodes:      int   = 50,
    render:          bool  = False,
    goal_axis:       int   = 0,
    goal_threshold:  float = 1.35,
    device:          str   = "cpu",
    seed_offset:     int   = 9999,
) -> Tuple[float, float]:
    """
    Evaluate policy success rate overall and on OOD goals.
    Supports both BCPolicy (single-step) and DiffusionPolicy (chunked).

    Returns: (overall_success_rate, ood_success_rate)
    """
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    policy.eval()

    all_success: List[float] = []
    ood_success: List[float] = []
    use_chunks = _is_diffusion(policy)

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep_idx)
        is_ood  = obs["desired_goal"][goal_axis] <= goal_threshold

        done = truncated = False
        episode_success  = False

        while not (done or truncated):
            if use_chunks:
                chunk = policy.act_chunk(obs, device=device)   # (chunk_size, 4)
                for action in chunk:
                    if done or truncated:
                        break
                    obs, _, done, truncated, info = env.step(action)
                    if info.get("is_success", False):
                        episode_success = True
            else:
                action = policy.act(obs, device=device)
                obs, _, done, truncated, info = env.step(action)
                if info.get("is_success", False):
                    episode_success = True

        all_success.append(float(episode_success))
        if is_ood:
            ood_success.append(float(episode_success))

    env.close()
    overall = float(np.mean(all_success))
    ood     = float(np.mean(ood_success)) if ood_success else 0.0
    return overall, ood


# ── Self-rollout collection ───────────────────────────────────────────────────

def collect_self_rollouts(
    policy,
    env_id:     str,
    n_episodes: int = 200,
    device:     str = "cpu",
    iteration:  int = 0,
) -> List[Episode]:
    """
    Collect self-rollout episodes. Supports chunked execution for DiffusionPolicy.

    For DiffusionPolicy: policy is queried every chunk_size steps.
    For BCPolicy:        policy is queried every step.

    In both cases every individual transition is stored (not just chunk-starts),
    so HER relabeling still operates on individual transitions.
    """
    env = gym.make(env_id, render_mode=None)
    policy.eval()
    episodes: List[Episode] = []
    use_chunks = _is_diffusion(policy)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode = Episode(source="self", iteration=iteration)
        done = truncated = False
        episode_success  = False

        while not (done or truncated):
            if use_chunks:
                chunk = policy.act_chunk(obs, device=device)   # (chunk_size, 4)
                for action in chunk:
                    if done or truncated:
                        break
                    obs_body      = obs["observation"].copy()
                    achieved_goal = obs["achieved_goal"].copy()
                    desired_goal  = obs["desired_goal"].copy()

                    obs, _, done, truncated, info = env.step(action)
                    if info.get("is_success", False):
                        episode_success = True

                    episode.add_transition(obs_body, action, achieved_goal, desired_goal)
            else:
                obs_body      = obs["observation"].copy()
                achieved_goal = obs["achieved_goal"].copy()
                desired_goal  = obs["desired_goal"].copy()

                action = policy.act(obs, device=device)
                obs, _, done, truncated, info = env.step(action)
                if info.get("is_success", False):
                    episode_success = True

                episode.add_transition(obs_body, action, achieved_goal, desired_goal)

        episode.finalize(episode_success)
        episodes.append(episode)

    env.close()
    return episodes


# ── Video saving ──────────────────────────────────────────────────────────────

def save_rollout_gif(
    policy,
    env_id:     str,
    path:       str = "rollout.gif",
    n_episodes: int = 3,
    fps:        int = 30,
    device:     str = "cpu",
):
    """Render n_episodes and save as GIF. Supports chunked policies."""
    env = gym.make(env_id, render_mode="rgb_array")
    policy.eval()
    frames = []
    use_chunks = _is_diffusion(policy)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        done = truncated = False

        while not (done or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if use_chunks:
                chunk = policy.act_chunk(obs, device=device)
                for action in chunk:
                    if done or truncated:
                        break
                    obs, _, done, truncated, _ = env.step(action)
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            else:
                action = policy.act(obs, device=device)
                obs, _, done, truncated, _ = env.step(action)

    env.close()
    if frames:
        imageio.mimsave(path, frames, fps=fps)
        print(f"  Rollout GIF saved → {path} ({len(frames)} frames)")