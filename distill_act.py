"""
distill_act.py — SAC → ACT distillation on FetchPickAndPlace-v4.

Uses the same SAC expert as distill.py, but trains an ACT policy instead
of a Diffusion Policy. Run AFTER training SAC with rl_finetune.py.

Usage:
    python distill_act.py                        # 500 demos, 400 epochs
    python distill_act.py --n-demos 1000         # more demos (~20 min)
    python distill_act.py --n-demos 100 --n-epochs 80  # quick smoke test
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
torch.set_num_threads(2)   # cap CPU cores — keeps system responsive

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

from config    import Config
from dataset   import RoboticsDataset, Episode
from act_policy import ACTPolicy
from trainer   import train_policy
from evaluate  import evaluate_policy


SAC_MODEL_PATH = "results/sac_best/best_model"
ACT_OUT        = "results/distilled_act.pt"
RESULTS_DIR    = "results"


# ── Collect SAC rollouts ───────────────────────────────────────────────────────

def collect_sac_demos(
    sac_model,
    env_id:      str,
    n_episodes:  int  = 500,
    obs_horizon: int  = 2,
    verbose:     bool = True,
) -> RoboticsDataset:
    """Roll out the SAC expert and store only successful episodes."""
    env     = gym.make(env_id, render_mode=None)
    dataset = RoboticsDataset()
    n_success = 0

    if verbose:
        print(f"\n  Collecting {n_episodes} SAC rollouts (obs_horizon={obs_horizon})...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7 + 2000)

        obs_buf = deque(
            [obs["observation"].copy()] * obs_horizon,
            maxlen=obs_horizon,
        )

        episode_obs      = []
        episode_actions  = []
        episode_achieved = []
        done = truncated = False
        ep_success = False

        while not (done or truncated):
            action, _ = sac_model.predict(obs, deterministic=True)

            stacked_body = np.concatenate(list(obs_buf))
            episode_obs.append(stacked_body)
            episode_actions.append(action.copy())
            episode_achieved.append(obs["achieved_goal"].copy())

            desired_goal = obs["desired_goal"].copy()
            obs, _, done, truncated, info = env.step(action)
            obs_buf.append(obs["observation"].copy())

            if info.get("is_success", False):
                ep_success = True

        ep_obj = Episode(source="sac", iteration=0)
        for ob, ac, ag in zip(episode_obs, episode_actions, episode_achieved):
            ep_obj.add_transition(
                obs_body=ob, action=ac,
                achieved_goal=ag, desired_goal=desired_goal,
            )
        ep_obj.finalize(success=ep_success)

        if ep_success:
            dataset.add_episode(ep_obj)
            n_success += 1

        if verbose and (ep + 1) % 100 == 0:
            print(f"    {ep+1}/{n_episodes} episodes  |  "
                  f"successes: {n_success}/{ep+1} ({n_success/(ep+1):.0%})")

    env.close()
    print(f"\n  ✓  Collected {n_episodes} episodes  |  "
          f"success rate: {n_success/n_episodes:.1%}  |  "
          f"transitions: {len(dataset)}")
    return dataset


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from stable_baselines3 import SAC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    config.obs_horizon = 2
    config.n_epochs    = args.n_epochs

    env_id = config.env_id
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Dims ──────────────────────────────────────────────────────────────────
    env  = gym.make(env_id, render_mode=None)
    obs, _ = env.reset()
    obs_body_dim = obs["observation"].shape[0]
    goal_dim     = obs["desired_goal"].shape[0]
    act_dim      = env.action_space.shape[0]
    env.close()

    obs_dim = obs_body_dim * config.obs_horizon + goal_dim   # 25*2+3 = 53

    print(f"\n{'═'*55}")
    print(f"  SAC → ACT Distillation  ({env_id})")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  device={device}")
    print(f"{'═'*55}")

    # ── Load SAC ──────────────────────────────────────────────────────────────
    if not os.path.exists(SAC_MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"SAC model not found: {SAC_MODEL_PATH}.zip\n"
            "Run: python rl_finetune.py  first."
        )
    print(f"\n  Loading SAC: {SAC_MODEL_PATH}.zip")
    _env      = gym.make(env_id, render_mode=None)
    sac_model = SAC.load(SAC_MODEL_PATH, env=_env)
    _env.close()
    print("  ✓  SAC loaded.")

    # ── Collect demos ─────────────────────────────────────────────────────────
    dataset = collect_sac_demos(
        sac_model, env_id,
        n_episodes=args.n_demos,
        obs_horizon=config.obs_horizon,
    )

    # ── Build ACT ─────────────────────────────────────────────────────────────
    print(f"\n  Building ACT policy...")
    policy = ACTPolicy(
        obs_dim      = obs_dim,
        action_dim   = act_dim,
        chunk_size   = config.chunk_size,
        hidden_dim   = config.act_hidden,
        n_heads      = config.act_n_heads,
        n_enc_layers = config.act_enc_layers,
        n_dec_layers = config.act_dec_layers,
        latent_dim   = config.act_latent_dim,
        beta         = config.act_beta,
        dropout      = config.act_dropout,
    )
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  ACT parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n  Training ACT ({len(dataset)} transitions, {config.n_epochs} epochs)...")
    train_policy(policy, dataset, config, device=device, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\n  Evaluating ACT (50 episodes)...")
    sr, ood = evaluate_policy(
        policy, env_id, n_episodes=50,
        device=device, obs_horizon=config.obs_horizon,
    )
    print(f"  ✓  ACT → success={sr:.1%}  OOD={ood:.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict":   policy.state_dict(),
        "obs_dim":      obs_dim,
        "act_dim":      act_dim,
        "config":       vars(config),
        "success_rate": sr,
        "obs_mean": policy.obs_mean.cpu() if policy.obs_mean is not None else None,
        "obs_std":  policy.obs_std.cpu()  if policy.obs_std  is not None else None,
        "act_mean": policy.act_mean.cpu() if policy.act_mean is not None else None,
        "act_std":  policy.act_std.cpu()  if policy.act_std  is not None else None,
    }, ACT_OUT)
    print(f"  ✓  ACT saved → {ACT_OUT}")

    print(f"\n{'═'*55}")
    print(f"  DONE")
    print(f"  SAC+HER:   ~85%  (expert)")
    print(f"  ACT:        {sr:.1%}  (distilled)")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-demos",  type=int, default=500)
    parser.add_argument("--n-epochs", type=int, default=400)
    args = parser.parse_args()
    main(args)