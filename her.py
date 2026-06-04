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


def relabel_episode(episode: Episode, min_block_movement: float = 0.05) -> Optional[Episode]:
    """
    Create a HER-relabeled version of an episode using the "final" strategy.

    The relabeled goal is the block's position at the *last* step of the episode
    (i.e., where the block actually ended up). All transitions in the episode
    are then valid demonstrations for that goal.

    Args:
        episode:            the original (typically failed) episode
        min_block_movement: minimum distance the block must have moved from
                            start to end for relabeling to be meaningful.
                            0.05m (5 cm) avoids flooding the dataset with
                            near-zero-push episodes that teach the policy to
                            barely touch the block rather than push to a goal.

    Returns:
        A new Episode with success=True and source="her", or None if the
        block didn't move enough to be worth relabeling.
    """
    if len(episode) == 0 or not episode.achieved_goal_list:
        return None

    # Where did the block end up?
    final_achieved_goal = episode.achieved_goal_list[-1].copy()
    initial_achieved_goal = episode.achieved_goal_list[0].copy()

    # Skip if block barely moved — the relabeled episode would be trivial
    block_movement = np.linalg.norm(final_achieved_goal - initial_achieved_goal)
    if block_movement < min_block_movement:
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
            desired_goal=final_achieved_goal,   # ← the relabeled goal
        )

    her_episode.finalize(success=True)   # by construction, this episode "succeeds"
    return her_episode


def apply_her(
    episodes: List[Episode],
    relabel_failed_only: bool = True,
    min_block_movement: float = 0.05,
) -> List[Episode]:
    """
    Apply HER relabeling to a batch of rollout episodes.

    Args:
        episodes:           list of rollout episodes (mix of success/failure)
        relabel_failed_only: if True, only relabel failed episodes (default).
                             if False, also relabel successful ones with their
                             final achieved goal (rarely useful but possible).
        min_block_movement: threshold for meaningful relabeling

    Returns:
        List of new HER-relabeled Episodes (does not include originals).
        Caller is responsible for adding both originals and HER episodes
        to the dataset as desired.
    """
    her_episodes = []

    for ep in episodes:
        if relabel_failed_only and ep.success:
            continue   # success — no relabeling needed

        her_ep = relabel_episode(ep, min_block_movement=min_block_movement)
        if her_ep is not None:
            her_episodes.append(her_ep)

    return her_episodes


def her_stats(episodes: List[Episode]) -> str:
    """Return a summary string of HER relabeling yield."""
    failed    = [ep for ep in episodes if not ep.success]
    relabeled = [ep for ep in apply_her(episodes) if ep is not None]
    total     = len(episodes)
    return (
        f"HER: {len(relabeled)}/{len(failed)} failed episodes relabeled "
        f"({len(relabeled)/max(1,total):.0%} of {total} total rollouts)"
    )