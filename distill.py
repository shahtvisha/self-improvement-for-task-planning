"""
distill.py — SAC → Diffusion Policy distillation.

The SAC+HER model achieved 85% success. We use it as an "expert" to collect
high-quality demonstrations, then train Diffusion Policy on those demos.

Why this works when the original diffusion training didn't:
  - Original: 200 scripted demos → policy at 8% → HER relabeling noise → plateau
  - Now: SAC at 85% → ~850 successes per 1000 rollouts → clean, rich dataset
  - Diffusion Policy finally has the data it needs to learn well.

Usage:
    # Collect 500 SAC rollouts, train diffusion, evaluate:
    python distill.py

    # More demos for a stronger policy (~20 min):
    python distill.py --n-demos 1000

    # Quick smoke test:
    python distill.py --n-demos 100 --n-epochs 80
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
torch.set_num_threads(2)   # cap CPU cores — keeps system responsive

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

from config           import Config
from dataset          import RoboticsDataset
from diffusion_policy import DiffusionPolicy
from trainer          import train_policy
from evaluate         import evaluate_policy


SAC_MODEL_PATH    = "results/sac_best/best_model"
DIFFUSION_OUT     = "results/distilled_diffusion.pt"
RESULTS_DIR       = "results"


# ── Collect SAC rollouts into RoboticsDataset ─────────────────────────────────

def collect_sac_demos(
    sac_model,
    env_id:      str,
    n_episodes:  int = 500,
    obs_horizon: int = 2,
    verbose:     bool = True,
) -> RoboticsDataset:
    """
    Roll out the SAC policy and store transitions in a RoboticsDataset.

    SAC uses raw obs dict — no stacking needed for inference.
    But we store stacked obs_body (obs_horizon frames) so the dataset is
    compatible with DiffusionPolicy training.
    """
    env = gym.make(env_id, render_mode=None)
    dataset = RoboticsDataset()
    n_success = 0

    if verbose:
        print(f"\n  Collecting {n_episodes} SAC rollouts (obs_horizon={obs_horizon})...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7 + 1000)

        # Rolling buffer for obs stacking (diffusion needs stacked obs)
        obs_buf = deque(
            [obs["observation"].copy()] * obs_horizon,
            maxlen=obs_horizon,
        )

        episode_obs     = []
        episode_actions = []
        episode_achieved = []
        done = truncated = False
        ep_success = False

        while not (done or truncated):
            # SAC predicts from raw obs dict (no stacking)
            action, _ = sac_model.predict(obs, deterministic=True)

            # Store stacked obs_body for diffusion training
            stacked_body = np.concatenate(list(obs_buf))
            episode_obs.append(stacked_body)
            episode_actions.append(action.copy())
            episode_achieved.append(obs["achieved_goal"].copy())

            desired_goal = obs["desired_goal"].copy()
            obs, _, done, truncated, info = env.step(action)
            obs_buf.append(obs["observation"].copy())

            if info.get("is_success", False):
                ep_success = True

        # Build an Episode and add to dataset
        from dataset import Episode
        ep_obj = Episode(source="sac", iteration=0)
        for ob, ac, ag in zip(episode_obs, episode_actions, episode_achieved):
            ep_obj.add_transition(
                obs_body=ob,
                action=ac,
                achieved_goal=ag,
                desired_goal=desired_goal,
            )
        ep_obj.finalize(success=ep_success)
        if ep_success:   # BC distillation: only learn from successful demos
            dataset.add_episode(ep_obj)

        if ep_success:
            n_success += 1

        if verbose and (ep + 1) % 100 == 0:
            print(f"    {ep+1}/{n_episodes} episodes  |  "
                  f"success so far: {n_success}/{ep+1} "
                  f"({n_success/(ep+1):.0%})")

    env.close()
    sr = n_success / n_episodes
    print(f"\n  ✓  Collected {n_episodes} episodes  |  "
          f"success rate: {sr:.1%}  |  "
          f"total transitions: {len(dataset)}")
    return dataset


# ── Train Diffusion Policy on SAC demos ───────────────────────────────────────

def train_distilled_diffusion(
    dataset:  RoboticsDataset,
    obs_dim:  int,
    act_dim:  int,
    config:   Config,
    device:   str = "cpu",
) -> DiffusionPolicy:
    print(f"\n  Training Diffusion Policy on SAC demos "
          f"({len(dataset)} transitions, {config.n_epochs} epochs)...")

    policy = DiffusionPolicy(
        obs_dim           = obs_dim,
        action_dim        = act_dim,
        chunk_size        = config.chunk_size,
        T                 = config.diffusion_T,
        n_inference_steps = config.diffusion_inf_steps,
        hidden_dim        = config.diffusion_hidden,
        n_layers          = config.diffusion_layers,
        time_emb_dim      = config.diffusion_time_emb,
    )

    train_policy(policy, dataset, config, device=device, verbose=True)
    return policy


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from stable_baselines3 import SAC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    config.obs_horizon = 2

    # Override from args
    config.n_epochs = args.n_epochs

    env_id = config.env_id
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Get dims ──────────────────────────────────────────────────────────────
    env = gym.make(env_id, render_mode=None)
    obs, _ = env.reset()
    obs_body_dim = obs["observation"].shape[0]
    goal_dim     = obs["desired_goal"].shape[0]
    act_dim      = env.action_space.shape[0]
    env.close()

    obs_dim = obs_body_dim * config.obs_horizon + goal_dim  # 25*2+3 = 53

    print(f"\n{'═'*55}")
    print(f"  SAC → Diffusion Policy Distillation")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  device={device}")
    print(f"{'═'*55}")

    # ── Load SAC model ────────────────────────────────────────────────────────
    if not os.path.exists(SAC_MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"SAC model not found at {SAC_MODEL_PATH}.zip\n"
            "Run: python main.py --method rl  first."
        )
    print(f"\n  Loading SAC model: {SAC_MODEL_PATH}.zip")
    _load_env = gym.make(env_id, render_mode=None)
    sac_model = SAC.load(SAC_MODEL_PATH, env=_load_env)
    _load_env.close()
    print(f"  ✓  SAC loaded.")

    # ── Collect SAC demos ─────────────────────────────────────────────────────
    dataset = collect_sac_demos(
        sac_model, env_id,
        n_episodes=args.n_demos,
        obs_horizon=config.obs_horizon,
    )

    # ── Train Diffusion Policy ────────────────────────────────────────────────
    policy = train_distilled_diffusion(dataset, obs_dim, act_dim, config, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n  Evaluating distilled Diffusion Policy (50 episodes)...")
    sr, ood = evaluate_policy(
        policy, env_id, n_episodes=50,
        device=device, obs_horizon=config.obs_horizon,
    )
    print(f"  ✓  Distilled Diffusion Policy → success={sr:.1%}  OOD={ood:.1%}")

    # ── Save (include norm stats so inference works without retraining) ────────
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim":    obs_dim,
        "act_dim":    act_dim,
        "config":     vars(config),
        "success_rate": sr,
        "obs_mean":   policy.obs_mean.cpu() if policy.obs_mean is not None else None,
        "obs_std":    policy.obs_std.cpu()  if policy.obs_std  is not None else None,
        "act_mean":   policy.act_mean.cpu() if policy.act_mean is not None else None,
        "act_std":    policy.act_std.cpu()  if policy.act_std  is not None else None,
    }, DIFFUSION_OUT)
    print(f"  ✓  Diffusion Policy saved → {DIFFUSION_OUT}")

    print(f"\n{'═'*55}")
    print(f"  DONE")
    print(f"  SAC+HER:              ~85%")
    print(f"  Distilled Diffusion:  {sr:.1%}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-demos",  type=int, default=500,
                        help="SAC rollout episodes to collect (default 500)")
    parser.add_argument("--n-epochs", type=int, default=400,
                        help="Diffusion training epochs (default 400)")
    args = parser.parse_args()
    main(args)