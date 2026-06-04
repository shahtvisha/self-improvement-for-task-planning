"""
expert.py — Scripted expert for FetchPush-v4.

Improved 3-phase expert with explicit height alignment before approach.
The previous 2-phase version failed because the gripper would arrive at
the approach position from above, inadvertently pushing the block sideways
before descending to contact height.

3-phase strategy
────────────────
Phase 1 — Align height: move gripper to block height (z) while hovering
           above the approach position (no xy contact with block yet).

Phase 2 — Approach from behind: with height correct, slide laterally to
           the position directly behind the block (opposite side from goal).
           At this point gripper is in contact position.

Phase 3 — Push: drive gripper toward goal. Block is carried by contact force.

The gripper stays open (action[3] = 1.0) throughout — this is a pure push.

FetchPush-v4 observation layout
─────────────────────────────────
obs['observation'] shape: (25,)
  [0:3]  grip_pos      — gripper xyz
  [3:6]  object_pos    — block xyz (same as achieved_goal)
  [6:9]  object_rel    — block relative to gripper
  [9:11] grip_fingers  — finger widths
  ...
obs['achieved_goal'] shape: (3,) — block position
obs['desired_goal']  shape: (3,) — target position

Policy input: obs[0:25] ‖ desired_goal → 28-dim
Action: [dx, dy, dz, gripper] ∈ [-1, 1]^4
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
from typing import Dict

from policy  import BCPolicy
from dataset import RoboticsDataset, Episode


# ── Expert action ─────────────────────────────────────────────────────────────

def scripted_expert_action(obs_dict: Dict[str, np.ndarray], noise: float = 0.0) -> np.ndarray:
    """
    3-phase P-controller expert for FetchPush.

    Phase 1: Align height above approach point (no lateral contact with block)
    Phase 2: Slide into approach position (gripper now behind block at push height)
    Phase 3: Push straight toward goal

    Thresholds are in metres. The FetchPush workspace has blocks at ~z=0.42.
    """
    obs      = obs_dict["observation"]
    grip_pos = obs[:3].copy()
    obj_pos  = obs_dict["achieved_goal"].copy()   # block position
    goal_pos = obs_dict["desired_goal"].copy()

    obj_to_goal     = goal_pos - obj_pos
    dist_obj_goal   = np.linalg.norm(obj_to_goal)
    obj_to_goal_dir = obj_to_goal / (dist_obj_goal + 1e-8)

    # Target approach: 8 cm behind block along push axis, at block height
    approach_offset   = -obj_to_goal_dir * 0.08
    approach_xy_z     = np.array([
        obj_pos[0] + approach_offset[0],
        obj_pos[1] + approach_offset[1],
        obj_pos[2],                          # block height
    ])

    # Hover position: same xy as approach, but 15 cm above block
    hover_pos = approach_xy_z.copy()
    hover_pos[2] = obj_pos[2] + 0.15

    action = np.zeros(4, dtype=np.float32)

    # ── Phase 1: descend to approach height while staying away laterally ──────
    xy_dist_to_approach = np.linalg.norm(grip_pos[:2] - approach_xy_z[:2])
    z_error             = abs(grip_pos[2] - approach_xy_z[2])

    if xy_dist_to_approach > 0.04 and z_error > 0.01:
        # Move to hover position first (above approach, safe from block contact)
        delta = hover_pos - grip_pos
        action[:3] = np.clip(delta * 10.0, -1.0, 1.0)

    # ── Phase 2: approach block from behind (height is now correct) ───────────
    elif xy_dist_to_approach > 0.025:
        delta = approach_xy_z - grip_pos
        action[:3] = np.clip(delta * 12.0, -1.0, 1.0)

    # ── Phase 3: push toward goal ─────────────────────────────────────────────
    else:
        delta = goal_pos - grip_pos
        action[:3] = np.clip(delta * 12.0, -1.0, 1.0)

    action[3] = 1.0   # keep gripper open

    if noise > 0.0:
        action[:3] += np.random.normal(0.0, noise, 3).astype(np.float32)
        action[:3]  = np.clip(action[:3], -1.0, 1.0)

    return action


# ── Demo collection ───────────────────────────────────────────────────────────

def collect_expert_demos(
    env_id:          str,
    n_demos:         int,
    noise:           float = 0.03,
    restrict_goals:  bool  = False,
    goal_axis:       int   = 0,
    goal_threshold:  float = 1.35,
    verbose:         bool  = True,
) -> RoboticsDataset:
    """
    Collect successful expert demonstrations.

    Also applies HER to the expert demos themselves: for each collected demo,
    we re-run the same trajectory but record it as a demo for a nearby goal
    (the block's final position). This "demo augmentation" doubles the effective
    dataset size without any additional rollouts.

    Goal restriction: demos only cover goals where desired_goal[goal_axis] > threshold.
    The OOD region (other side) is left entirely for self-improvement to discover.
    """
    from her import relabel_episode

    env = gym.make(env_id, render_mode=None)
    dataset  = RoboticsDataset()

    collected    = 0
    attempts     = 0
    max_attempts = n_demos * 25

    if verbose:
        label = (f"restricted: axis={goal_axis} > {goal_threshold}"
                 if restrict_goals else "full goal distribution")
        print(f"Collecting {n_demos} expert demos ({label})")

    while collected < n_demos and attempts < max_attempts:
        obs, _ = env.reset()
        attempts += 1

        if restrict_goals:
            tries = 0
            while obs["desired_goal"][goal_axis] <= goal_threshold and tries < 40:
                obs, _ = env.reset()
                tries += 1
            if obs["desired_goal"][goal_axis] <= goal_threshold:
                continue

        episode = Episode(source="expert", iteration=0)
        done = truncated = False
        episode_success  = False

        while not (done or truncated):
            obs_body      = obs["observation"].copy()
            achieved_goal = obs["achieved_goal"].copy()
            desired_goal  = obs["desired_goal"].copy()
            action        = scripted_expert_action(obs, noise=noise)

            next_obs, _, done, truncated, info = env.step(action)
            if info.get("is_success", False):
                episode_success = True

            episode.add_transition(obs_body, action, achieved_goal, desired_goal)
            obs = next_obs

        episode.finalize(episode_success)

        if episode_success:
            dataset.add_episode(episode)

            # Demo augmentation: also add HER-relabeled version of this demo
            her_ep = relabel_episode(episode, min_block_movement=0.02)
            if her_ep is not None:
                dataset.add_episode(her_ep)

            collected += 1
            if verbose and collected % 25 == 0:
                print(f"  {collected}/{n_demos} demos collected")

    env.close()

    if verbose:
        print(f"Expert collection complete → {dataset}\n")

    return dataset