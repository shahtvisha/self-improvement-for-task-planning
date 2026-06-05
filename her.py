"""
her.py — Hindsight Experience Replay (HER) relabeling for BC.

Original paper: Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017 (OpenAI).
That work applied HER to RL (DDPG). We apply the same relabeling idea to BC
in a self-improvement loop — a combination that is not in the original paper.

The key insight
───────────────
In goal-conditioned tasks, a failed episode (didn't reach desired_goal) still
contains useful information: the robot successfully reached *some* position —
namely wherever the block ended up at the end of the episode.

Rather than discarding failed rollouts, we relabel them:
  - new_goal = final achieved_goal (where the block actually ended up)
  - relabeled transitions: obs_body ‖ new_goal → action (same actions)
  - This creates a new, valid "successful" episode for the relabeled goal.

The relabeled goal is genuinely reached — by definition, because we chose it
to be the block's final position. So the (relabeled_state, action) pairs are
correct demonstrations of how to push to that goal.

Why this helps
──────────────
Without HER, a failed rollout generates 0 training signal.
With HER, every rollout generates at least one training episode.

Empirically (from the RL HER paper): HER turns tasks with 0% success rate
into tasks that are learnable at all, because the agent can always learn
something from where it actually went, even if it wasn't the desired goal.

For BC self-improvement, this means:
  - Early iterations (low success rate): HER dramatically increases training data
  - Later iterations: success rate is high anyway, HER provides less marginal gain
  - Net effect: much faster convergence to high success rate

HER strategy: "final" (we use the final achieved_goal as the relabeled goal).
The original paper also proposes "future" and "episode" strategies, which
relabel with goals encountered *during* the episode. "final" is simplest and
most conservative — avoids overfitting to intermediate states.
"""

import numpy as np
from typing import List, Optional

from dataset import Episode


def relabel_episode(
    episode:            Episode,
    min_block_movement: float          = 0.05,
    goal_filter_axis:   Optional[int]  = None,
    goal_filter_min:    Optional[float] = None,
) -> Optional[Episode]:
    """
    Create a HER-relabeled version of an episode using the "final" strategy.

    The relabeled goal is the block's position at the *last* step of the episode.

    Args:
        episode:            the original (typically failed) episode
        min_block_movement: block must move ≥ this far (metres) to be worth relabeling.
                            Default 0.05 m (5 cm).
        goal_filter_axis:   if set, only relabel when the relabeled goal's coordinate
                            along this axis satisfies goal ≥ goal_filter_min.
                            Use this to keep HER goals within the test distribution —
                            e.g. axis=0 (x), min=1.35 matches restrict_goals in config.
        goal_filter_min:    minimum value for goal_filter_axis (ignored if axis is None).

    Returns:
        A new Episode with success=True and source="her", or None if filtered out.
    """
    if len(episode) == 0 or not episode.achieved_goal_list:
        return None

    final_achieved_goal   = episode.achieved_goal_list[-1].copy()
    initial_achieved_goal = episode.achieved_goal_list[0].copy()

    # Filter 1: block must have moved meaningfully
    block_movement = np.linalg.norm(final_achieved_goal - initial_achieved_goal)
    if block_movement < min_block_movement:
        return None

    # Filter 2: relabeled goal must fall within the test goal distribution
    if goal_filter_axis is not None and goal_filter_min is not None:
        if final_achieved_goal[goal_filter_axis] <= goal_filter_min:
            return None

    # Build the relabeled episode
    her_episode = Episode(source="her", iteration=episode.iteration)
    for obs_body, action, achieved_goal in zip(
        episode.obs_body_list,
        episode.action_list,
        episode.achieved_goal_list,
    ):
        her_episode.add_transition(
            obs_body=obs_body,
            action=action,
            achieved_goal=achieved_goal,
            desired_goal=final_achieved_goal,
        )

    her_episode.finalize(success=True)
    return her_episode


def apply_her(
    episodes:           List[Episode],
    relabel_failed_only: bool           = True,
    min_block_movement:  float          = 0.05,
    goal_filter_axis:    Optional[int]  = None,
    goal_filter_min:     Optional[float] = None,
) -> List[Episode]:
    """
    Apply HER relabeling to a batch of rollout episodes.

    Args:
        episodes:            list of rollout episodes (mix of success/failure)
        relabel_failed_only: only relabel failed episodes (default True)
        min_block_movement:  block movement threshold
        goal_filter_axis:    if set, drop HER episodes whose relabeled goal doesn't
                             satisfy goal[axis] >= goal_filter_min. Keeps HER goals
                             within the test distribution when restrict_goals is active.
        goal_filter_min:     minimum threshold for goal_filter_axis

    Returns:
        List of new HER-relabeled Episodes (does not include originals).
    """
    her_episodes = []

    for ep in episodes:
        if relabel_failed_only and ep.success:
            continue

        her_ep = relabel_episode(
            ep,
            min_block_movement=min_block_movement,
            goal_filter_axis=goal_filter_axis,
            goal_filter_min=goal_filter_min,
        )
        if her_ep is not None:
            her_episodes.append(her_ep)

    return her_episodes


def her_stats(
    episodes:         List[Episode],
    goal_filter_axis: Optional[int]   = None,
    goal_filter_min:  Optional[float] = None,
) -> str:
    """Return a summary string of HER relabeling yield."""
    failed    = [ep for ep in episodes if not ep.success]
    relabeled = apply_her(
        episodes,
        goal_filter_axis=goal_filter_axis,
        goal_filter_min=goal_filter_min,
    )
    total = len(episodes)
    return (
        f"HER: {len(relabeled)}/{len(failed)} failed episodes relabeled "
        f"({len(relabeled)/max(1,total):.0%} of {total} total rollouts)"
    )