"""
dagger.py — DAgger (Dataset Aggregation) comparison baseline.

Paper: Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and Structured
       Prediction to No-Regret Online Learning", AISTATS 2011.

DAgger is the gold-standard solution to BC's covariate shift problem. It serves
as our upper-bound comparison: a method that *requires* an online expert but
achieves better sample efficiency as a result.

Algorithm
─────────
  D₀ ← expert_demos
  π₀ ← train_BC(D₀)
  for i in 1..K:
      Roll out πᵢ₋₁ in the environment (policy chooses actions for stepping)
      At every visited state s: query expert → a* = expert(s)
      Dᵢ ← Dᵢ₋₁ ∪ {(s, a*)}    ← label with EXPERT action, not policy action
      πᵢ ← train_BC(Dᵢ)

The critical difference from self-improvement:
  - Self-improvement: label states with the POLICY's own actions (from successful rollouts)
  - DAgger: label states with the EXPERT's actions (always available, always correct)

This means DAgger never suffers from incorrect labels — even on states far from
the demonstrated distribution, the expert can provide a good action. The self-
improvement loop cannot do this; it must rely on the policy occasionally succeeding
on OOD states to generate training signal for those regions.

Why DAgger is an upper bound
─────────────────────────────
  - DAgger has access to the expert on every step of every rollout
  - The self-improvement loop only gets signal when the policy succeeds
  - If the policy has 0% success on some goal region, self-improvement cannot
    bootstrap, but DAgger still generates training data for those states

Why we still care about self-improvement
─────────────────────────────────────────
  - DAgger requires an online expert — not available for most real tasks
  - Human demonstrators cannot label every state of a 200-episode rollout
  - Our method is fully offline after the initial demo collection phase

The comparison plot (self-improvement vs. DAgger) makes this trade-off explicit:
"We achieve X% of DAgger's performance without requiring the online expert."
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
from typing import List, Callable

from policy   import BCPolicy
from dataset  import Episode, RoboticsDataset
from expert   import scripted_expert_action


def collect_dagger_rollouts(
    policy:     BCPolicy,
    env_id:     str,
    n_episodes: int,
    device:     str = "cpu",
    iteration:  int = 0,
    noise:      float = 0.0,
) -> List[Episode]:
    """
    Collect one DAgger iteration of rollouts.

    The *policy* steps the environment (determining which states are visited),
    but every state is labelled with the *scripted expert's* action.
    This is the DAgger protocol: on-policy state distribution, expert labels.

    Args:
        policy:     current BC policy (used to step the environment)
        env_id:     gymnasium environment id
        n_episodes: number of rollout episodes
        device:     torch device
        iteration:  self-improvement iteration index (stored on Episode)
        noise:      noise added to expert labels (matches demo collection)

    Returns:
        List of Episodes — all labelled with expert actions, all marked success=True
        (because the labels are correct regardless of whether the policy would have
        succeeded on its own).
    """
    env = gym.make(env_id, render_mode=None)
    policy.eval()
    episodes: List[Episode] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode = Episode(source="dagger", iteration=iteration)
        done = truncated = False
        policy_success = False

        while not (done or truncated):
            obs_body      = obs["observation"].copy()
            achieved_goal = obs["achieved_goal"].copy()
            desired_goal  = obs["desired_goal"].copy()

            # Expert labels this state (the key DAgger step)
            expert_action = scripted_expert_action(obs, noise=noise)

            # Policy steps the environment (determines next state visited)
            policy_action = policy.act(obs, device=device)

            # Store (state, EXPERT_action) — not policy action
            episode.add_transition(obs_body, expert_action, achieved_goal, desired_goal)

            next_obs, _, done, truncated, info = env.step(policy_action)
            if info.get("is_success", False):
                policy_success = True

            obs = next_obs

        # Mark whether the policy itself succeeded (for metrics)
        episode.finalize(success=policy_success)
        episode.source = "dagger"   # reaffirm source after finalize
        episodes.append(episode)

    env.close()
    return episodes


def run_dagger_loop(
    env_id:        str,
    obs_dim:       int,
    action_dim:    int,
    demo_dataset:  RoboticsDataset,
    config,
    device:        str = "cpu",
) -> dict:
    """
    Run the full DAgger training loop and return metrics.

    Starts from the same expert demos as self-improvement for a fair comparison.
    Each iteration: collect on-policy rollouts with expert labels → retrain.

    Returns:
        dict with keys: iteration, success_rate, ood_success_rate, dataset_size
    """
    from policy  import BCPolicy
    from trainer import train_bc
    from evaluate import evaluate_policy

    print("\n" + "═" * 60)
    print("  DAGGER BASELINE")
    print("═" * 60)

    # Start from same demo dataset
    dataset = RoboticsDataset()
    dataset.add_episodes([ep for ep in _iter_episodes_from_dataset(demo_dataset)])

    # Initial BC on demos
    policy = BCPolicy(obs_dim, action_dim, config.hidden_dims)
    train_bc(policy, dataset, n_epochs=config.n_epochs, batch_size=config.batch_size,
             lr=config.lr, weight_decay=config.weight_decay, device=device,
             verbose=config.verbose)

    sr0, ood0 = evaluate_policy(
        policy, env_id, config.n_eval_episodes,
        goal_axis=config.goal_restrict_axis,
        goal_threshold=config.goal_restrict_threshold,
        device=device,
    )
    print(f"  Iteration 0 (DAgger cold start) → success={sr0:.1%}  OOD={ood0:.1%}\n")

    metrics = {
        "iteration":         [0],
        "success_rate":      [sr0],
        "ood_success_rate":  [ood0],
        "dataset_size":      [len(dataset)],
    }

    for iteration in range(1, config.n_iterations + 1):
        print(f"  DAgger iteration {iteration}/{config.n_iterations}...")

        rollouts = collect_dagger_rollouts(
            policy, env_id,
            n_episodes=config.n_rollouts_per_iter,
            device=device,
            iteration=iteration,
            noise=config.expert_noise,
        )

        # Add ALL DAgger episodes (all are expert-labelled, all are valid)
        dataset.add_episodes(rollouts, success_only=False)

        # Retrain from scratch
        policy = BCPolicy(obs_dim, action_dim, config.hidden_dims)
        train_bc(policy, dataset, n_epochs=config.n_epochs, batch_size=config.batch_size,
                 lr=config.lr, weight_decay=config.weight_decay, device=device,
                 verbose=config.verbose)

        sr, ood_sr = evaluate_policy(
            policy, env_id, config.n_eval_episodes,
            goal_axis=config.goal_restrict_axis,
            goal_threshold=config.goal_restrict_threshold,
            device=device,
        )
        print(f"  ✓  success={sr:.1%}  OOD={ood_sr:.1%}  dataset={len(dataset)}\n")

        metrics["iteration"].append(iteration)
        metrics["success_rate"].append(sr)
        metrics["ood_success_rate"].append(ood_sr)
        metrics["dataset_size"].append(len(dataset))

    return metrics, policy


def _iter_episodes_from_dataset(dataset: RoboticsDataset):
    """
    Reconstruct Episode objects from a flat RoboticsDataset.
    Used to share the demo dataset between DAgger and self-improvement runs.

    Note: reconstructed episodes have source="expert" and no achieved_goal
    (sufficient for training, but HER cannot be applied to them).
    """
    # We can't reconstruct episodes perfectly from a flat dataset,
    # so we create one big "episode" with all transitions.
    # This is fine for DAgger — it just needs the (obs, action) pairs.
    ep = Episode(source="expert", iteration=0)
    ep.obs_body_list  = [obs[:-3].copy() for obs in dataset._obs]   # strip goal
    ep.action_list    = [a.copy() for a in dataset._actions]
    ep.achieved_goal_list = [np.zeros(3)] * len(dataset._obs)       # placeholder
    # desired_goal: just use zeros — flat_obs already has it baked in
    # Instead, override flat_obs_iter to return stored flat obs directly
    ep._flat_obs = [(obs.copy(), act.copy())
                    for obs, act in zip(dataset._obs, dataset._actions)]
    ep.finalize(True)

    class _FlatEpisode:
        """Thin wrapper that yields pre-computed flat obs directly."""
        success = True
        source  = "expert"
        iteration = 0
        desired_goal = np.zeros(3)   # dummy — flat_obs_iter is overridden

        def flat_obs_iter(self_inner):
            return iter(ep._flat_obs)

    yield _FlatEpisode()