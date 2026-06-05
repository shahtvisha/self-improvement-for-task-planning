"""
main.py — Ensemble Self-Improving Diffusion Policy + HER + SAC online RL

Policy options:
  --policy diffusion   DiffusionPolicy x0-prediction + DDIM inference (default)
  --policy mlp         BCPolicy MLP (fast, for comparison)

Method options:
  --method rl          SAC + HER online RL — THE solution (~70-80% success)
  --method ensemble    Offline ensemble self-improvement + HER (cold-start BC)
  --method dagger      DAgger baseline (needs live expert)
  --method bc          BC-only baseline (no self-improvement)
  --method all         Run all methods (very slow)

Recommended runs:
  # Real solution — SAC+HER (~30-60 min, achieves 70-80%):
  python main.py --method rl

  # Quick sanity check on offline BC (5-10 min):
  python main.py --policy diffusion --method ensemble --n-iter 2 --n-demos 50

  # Full offline comparison (60-90 min):
  python main.py --policy diffusion --method all
"""

import argparse
import json
import os
import random

import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch

from config            import Config
from dataset           import RoboticsDataset
from evaluate          import collect_self_rollouts, evaluate_policy, save_rollout_gif
from expert            import collect_expert_demos
from her               import apply_her, her_stats
from policy            import BCPolicy
from diffusion_policy  import DiffusionPolicy
from trainer           import train_policy
from visualize         import comparison_plot, heatmap_series, dataset_growth_plot
from rl_finetune       import run_sac_her, sac_her_rollout_gif


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_env_dims(env_id: str, obs_horizon: int = 1):
    """
    obs_dim = obs_body_dim * obs_horizon + goal_dim
    With obs_horizon=2: 25*2 + 3 = 53 (vs 28 for horizon=1).
    """
    env = gym.make(env_id, render_mode=None)
    obs, _ = env.reset()
    obs_body_dim = obs["observation"].shape[0]
    goal_dim     = obs["desired_goal"].shape[0]
    obs_dim      = obs_body_dim * obs_horizon + goal_dim
    action_dim   = env.action_space.shape[0]
    env.close()
    return obs_dim, action_dim


def make_policy(obs_dim: int, action_dim: int, config: Config):
    """Construct a fresh policy of the configured type."""
    if config.policy_type == "diffusion":
        return DiffusionPolicy(
            obs_dim           = obs_dim,
            action_dim        = action_dim,
            chunk_size        = config.chunk_size,
            T                 = config.diffusion_T,
            n_inference_steps = config.diffusion_inf_steps,
            hidden_dim        = config.diffusion_hidden,
            n_layers          = config.diffusion_layers,
            time_emb_dim      = config.diffusion_time_emb,
        )
    else:
        return BCPolicy(obs_dim, action_dim, config.hidden_dims)


def clone_dataset(src: RoboticsDataset) -> RoboticsDataset:
    dst = RoboticsDataset()
    dst._obs           = list(src._obs)
    dst._actions       = list(src._actions)
    dst._episode_ends  = list(src._episode_ends)    # ← carry over boundaries
    dst.n_episodes_total     = src.n_episodes_total
    dst.n_episodes_success   = src.n_episodes_success
    dst.n_expert_transitions = src.n_expert_transitions
    dst.n_her_transitions    = src.n_her_transitions
    return dst


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def quick_eval(policy, env_id, config, device, n=20) -> float:
    sr, _ = evaluate_policy(policy, env_id, n_episodes=n,
                             goal_axis=config.goal_restrict_axis,
                             goal_threshold=config.goal_restrict_threshold,
                             device=device,
                             obs_horizon=config.obs_horizon)
    return sr


# ── Experiment 1: BC / Diffusion baseline (no self-improvement) ───────────────

def run_baseline(env_id, obs_dim, action_dim, demo_dataset, config, device):
    label = "Diffusion" if config.policy_type == "diffusion" else "BC"
    print(f"\n{'═'*60}")
    print(f"  {label.upper()} BASELINE (no self-improvement)")
    print(f"{'═'*60}")

    policy = make_policy(obs_dim, action_dim, config)
    train_policy(policy, demo_dataset, config, device=device, verbose=config.verbose)

    sr, ood = evaluate_policy(
        policy, env_id, config.n_eval_episodes,
        goal_axis=config.goal_restrict_axis,
        goal_threshold=config.goal_restrict_threshold,
        device=device, obs_horizon=config.obs_horizon,
    )
    print(f"  {label} Baseline → success={sr:.1%}  OOD={ood:.1%}\n")
    return {"iteration": [0], "success_rate": [sr], "ood_success_rate": [ood],
            "dataset_size": [len(demo_dataset)]}, policy


# ── Experiment 2: Ensemble Self-Improve + HER ─────────────────────────────────

def run_ensemble_self_improve_her(env_id, obs_dim, action_dim, demo_dataset, config, device):
    N = config.n_ensemble
    rollouts_per_policy = config.n_rollouts_per_iter // N
    label = "Diffusion" if config.policy_type == "diffusion" else "MLP"

    print(f"\n{'═'*60}")
    print(f"  ENSEMBLE SELF-IMPROVEMENT + HER  [{label} policy, N={N}]")
    print(f"{'═'*60}")

    dataset = clone_dataset(demo_dataset)

    # Cold start: train N policies (different seeds → different failure modes)
    print(f"\n  Cold start: training {N} × {label} policies...")
    policies = []
    for i in range(N):
        torch.manual_seed(config.seed + i * 100)
        p = make_policy(obs_dim, action_dim, config)
        train_policy(p, dataset, config, device=device, verbose=False)
        policies.append(p)
    print(f"  {N} policies trained.")

    bp = max(policies, key=lambda p: quick_eval(p, env_id, config, device))
    sr0, ood0 = evaluate_policy(
        bp, env_id, config.n_eval_episodes,
        goal_axis=config.goal_restrict_axis,
        goal_threshold=config.goal_restrict_threshold,
        device=device, obs_horizon=config.obs_horizon,
    )
    print(f"  Iteration 0 → success={sr0:.1%}  OOD={ood0:.1%}\n")

    metrics = {
        "iteration": [0], "success_rate": [sr0], "ood_success_rate": [ood0],
        "dataset_size": [len(dataset)], "n_success_added": [0], "n_her_added": [0],
        "n_expert_transitions": [dataset.n_expert_transitions],
        "n_success_transitions": [0], "n_her_transitions": [0],
    }
    policy_checkpoints = [bp]
    heatmap_iters      = [0]

    for iteration in range(1, config.n_iterations + 1):
        print(f"\n  {'─'*58}")
        print(f"  Iteration {iteration}/{config.n_iterations}  "
              f"[{N} policies × {rollouts_per_policy} rollouts each]")
        print(f"  {'─'*58}")

        # Each policy rolls out independently
        all_rollouts = []
        for i, policy in enumerate(policies):
            rollouts = collect_self_rollouts(
                policy, env_id, n_episodes=rollouts_per_policy,
                device=device, iteration=iteration,
                obs_horizon=config.obs_horizon,
            )
            n_s = sum(ep.success for ep in rollouts)
            print(f"  Policy {i+1}: {n_s}/{rollouts_per_policy} successes ({n_s/rollouts_per_policy:.0%})")
            all_rollouts.extend(rollouts)

        total_s = sum(ep.success for ep in all_rollouts)
        print(f"  Pooled: {total_s}/{len(all_rollouts)} successes ({total_s/len(all_rollouts):.0%})")

        # HER relabeling — filter goals to match test distribution when goals are restricted.
        # Without this filter, 89% of failures get relabeled with goals that land outside
        # the test distribution, flooding the dataset with off-distribution training signal.
        goal_axis = config.goal_restrict_axis    if config.restrict_goals else None
        goal_min  = config.goal_restrict_threshold if config.restrict_goals else None

        her_episodes = apply_her(
            all_rollouts, relabel_failed_only=True,
            goal_filter_axis=goal_axis, goal_filter_min=goal_min,
        )
        print(f"  {her_stats(all_rollouts, goal_filter_axis=goal_axis, goal_filter_min=goal_min)}")

        n_success_added = dataset.add_episodes(all_rollouts, success_only=True)
        n_her_added     = dataset.add_episodes(her_episodes, success_only=False)
        print(f"  Added: {n_success_added} success + {n_her_added} HER  |  {dataset}")

        # Retrain all N from scratch (different seeds = maintained diversity)
        print(f"  Retraining {N} × {label} policies...")
        policies = []
        for i in range(N):
            torch.manual_seed(config.seed + iteration * 1000 + i * 100)
            p = make_policy(obs_dim, action_dim, config)
            train_policy(p, dataset, config, device=device, verbose=False)
            policies.append(p)

        bp = max(policies, key=lambda p: quick_eval(p, env_id, config, device))
        sr, ood = evaluate_policy(
            bp, env_id, config.n_eval_episodes,
            goal_axis=config.goal_restrict_axis,
            goal_threshold=config.goal_restrict_threshold,
            device=device, obs_horizon=config.obs_horizon,
        )
        print(f"\n  ✓  Best policy: success={sr:.1%}  OOD={ood:.1%}  dataset={len(dataset)}")

        metrics["iteration"].append(iteration)
        metrics["success_rate"].append(sr)
        metrics["ood_success_rate"].append(ood)
        metrics["dataset_size"].append(len(dataset))
        metrics["n_success_added"].append(n_success_added)
        metrics["n_her_added"].append(n_her_added)
        metrics["n_expert_transitions"].append(dataset.n_expert_transitions)
        metrics["n_success_transitions"].append(
            metrics["n_success_transitions"][-1] + n_success_added * 50)
        metrics["n_her_transitions"].append(
            metrics["n_her_transitions"][-1] + n_her_added * 50)

        if iteration % 2 == 0 or iteration == config.n_iterations:
            policy_checkpoints.append(bp)
            heatmap_iters.append(iteration)

    return metrics, bp, policy_checkpoints, heatmap_iters


# ── Experiment 3: SAC + HER online RL ────────────────────────────────────────

def run_rl(env_id, config, device):
    """
    Online RL with SAC + HER. This is the recommended method for FetchPush.

    No expert demos needed for SAC — it learns entirely from scratch via
    environment interaction. The HER relabeling is handled internally by
    stable-baselines3's HerReplayBuffer.

    Expected outcome: ~70-80% success on FetchPush-v4 in 500K steps.
    Compare to offline BC/diffusion plateau at ~8-15%.
    """
    print(f"\n{'═'*60}")
    print(f"  SAC + HER  (online RL — the real solution)")
    print(f"  Expected: ~70-80% success in {config.sac_timesteps:,} steps")
    print(f"{'═'*60}")

    sac_model, metrics = run_sac_her(
        env_id          = env_id,
        total_timesteps = config.sac_timesteps,
        n_sampled_goal  = config.sac_n_her_goals,
        learning_starts = config.sac_learning_starts,
        batch_size      = config.batch_size,
        learning_rate   = config.sac_lr,
        tau             = config.sac_tau,
        gamma           = config.sac_gamma,
        buffer_size     = config.sac_buffer_size,
        device          = device,
        log_dir         = config.log_dir,
        verbose         = config.verbose,
        seed            = config.seed,
    )

    if sac_model is not None and config.save_video:
        gif_path = os.path.join(config.log_dir, "sac_rollout.gif")
        sac_her_rollout_gif(sac_model, env_id, path=gif_path, n_episodes=5)

    return sac_model, metrics


# ── Experiment 4: DAgger ──────────────────────────────────────────────────────

def run_dagger(env_id, obs_dim, action_dim, demo_dataset, config, device):
    from dagger import collect_dagger_rollouts
    label = "Diffusion" if config.policy_type == "diffusion" else "MLP"

    print(f"\n{'═'*60}")
    print(f"  DAGGER BASELINE [{label} policy]")
    print(f"{'═'*60}")

    dataset = clone_dataset(demo_dataset)
    policy  = make_policy(obs_dim, action_dim, config)
    train_policy(policy, dataset, config, device=device, verbose=config.verbose)

    sr0, ood0 = evaluate_policy(
        policy, env_id, config.n_eval_episodes,
        goal_axis=config.goal_restrict_axis,
        goal_threshold=config.goal_restrict_threshold, device=device,
    )
    print(f"  Iteration 0 → success={sr0:.1%}  OOD={ood0:.1%}\n")
    metrics = {"iteration": [0], "success_rate": [sr0], "ood_success_rate": [ood0],
               "dataset_size": [len(dataset)]}

    for iteration in range(1, config.n_iterations + 1):
        print(f"\n  DAgger iteration {iteration}/{config.n_iterations}...")
        rollouts = collect_dagger_rollouts(
            policy, env_id, n_episodes=config.n_rollouts_per_iter,
            device=device, iteration=iteration, noise=config.expert_noise,
        )
        dataset.add_episodes(rollouts, success_only=False)

        policy = make_policy(obs_dim, action_dim, config)
        train_policy(policy, dataset, config, device=device, verbose=config.verbose)

        sr, ood = evaluate_policy(
            policy, env_id, config.n_eval_episodes,
            goal_axis=config.goal_restrict_axis,
            goal_threshold=config.goal_restrict_threshold, device=device,
        )
        print(f"  ✓  success={sr:.1%}  OOD={ood:.1%}  dataset={len(dataset)}")
        metrics["iteration"].append(iteration)
        metrics["success_rate"].append(sr)
        metrics["ood_success_rate"].append(ood)
        metrics["dataset_size"].append(len(dataset))

    return metrics, policy


# ── Main orchestration ────────────────────────────────────────────────────────

def run(config: Config, methods: list, render: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(config.seed)
    os.makedirs(config.log_dir, exist_ok=True)

    label = "Diffusion" if config.policy_type == "diffusion" else "MLP"
    print(f"\n{'='*60}")
    print(f"  Robotics Self-Improvement: {label} + SAC+HER")
    print(f"  Env: {config.env_id}  |  Device: {device}  |  Policy: {label}")
    print(f"  Methods: {methods}")
    print(f"{'='*60}")

    # ── SAC+HER path: no expert demos needed ──────────────────────────────────
    sac_model = None
    sac_metrics = None
    if "rl" in methods or "all" in methods:
        sac_model, sac_metrics = run_rl(config.env_id, config, device)
        if sac_metrics:
            save_metrics(sac_metrics, os.path.join(config.log_dir, "metrics_sac.json"))

    # ── Offline BC methods: need expert demos ─────────────────────────────────
    bc_metrics = ensemble_metrics = dagger_metrics = None
    policy_checkpoints = []
    heatmap_iters = []
    final_policy = None

    offline_needed = any(m in methods for m in ["bc", "ensemble", "dagger", "all"])
    if offline_needed:
        obs_dim, action_dim = get_env_dims(config.env_id, obs_horizon=config.obs_horizon)
        print(f"  obs_dim={obs_dim} (obs_horizon={config.obs_horizon}×body+goal), action_dim={action_dim}")

        print(f"\n[Phase 1]  Collecting {config.n_expert_demos} expert demos...")
        demo_dataset = collect_expert_demos(
            env_id=config.env_id, n_demos=config.n_expert_demos,
            noise=config.expert_noise, restrict_goals=config.restrict_goals,
            goal_axis=config.goal_restrict_axis,
            goal_threshold=config.goal_restrict_threshold,
            obs_horizon=config.obs_horizon,
            verbose=config.verbose,
        )

        if "bc" in methods or "all" in methods:
            bc_metrics, _ = run_baseline(
                config.env_id, obs_dim, action_dim, demo_dataset, config, device)
            save_metrics(bc_metrics, os.path.join(config.log_dir, "metrics_bc.json"))

        if "ensemble" in methods or "all" in methods:
            ensemble_metrics, final_policy, policy_checkpoints, heatmap_iters = \
                run_ensemble_self_improve_her(
                    config.env_id, obs_dim, action_dim, demo_dataset, config, device)
            save_metrics(ensemble_metrics, os.path.join(config.log_dir, "metrics_ensemble.json"))
            if final_policy:
                torch.save(final_policy.state_dict(),
                           os.path.join(config.log_dir, "final_policy.pt"))

        if "dagger" in methods or "all" in methods:
            dagger_metrics, _ = run_dagger(
                config.env_id, obs_dim, action_dim, demo_dataset, config, device)
            save_metrics(dagger_metrics, os.path.join(config.log_dir, "metrics_dagger.json"))

        # Figures for offline methods
        print(f"\n{'='*60}\n  Generating figures...")
        _iters = list(range(config.n_iterations + 1))
        _zero  = [0.0] * (config.n_iterations + 1)
        _dummy = {"iteration": _iters, "success_rate": _zero, "ood_success_rate": _zero}

        comparison_plot(
            bc_metrics or _dummy, dagger_metrics or _dummy, ensemble_metrics or _dummy,
            out_dir=config.log_dir,
        )

        if policy_checkpoints:
            heatmap_series(
                policies=policy_checkpoints, iterations=heatmap_iters,
                env_id=config.env_id, device=device, n_episodes=500, grid_size=7,
                goal_axis=config.goal_restrict_axis,
                goal_threshold=config.goal_restrict_threshold,
                out_dir=config.log_dir,
                obs_horizon=config.obs_horizon,
            )
        if ensemble_metrics:
            dataset_growth_plot(ensemble_metrics, out_dir=config.log_dir)

        if config.save_video and final_policy:
            save_rollout_gif(final_policy, config.env_id,
                             path=os.path.join(config.log_dir, "final_policy.gif"),
                             n_episodes=5, device=device,
                             obs_horizon=config.obs_horizon)

        if render and final_policy:
            evaluate_policy(final_policy, config.env_id, n_episodes=10,
                            render=True, device=device,
                            obs_horizon=config.obs_horizon)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}\n  FINAL RESULTS")
    if sac_metrics:
        print(f"  SAC+HER          success={sac_metrics['success_rate']:.1%}  "
              f"(trained {config.sac_timesteps:,} steps)")
    if bc_metrics:
        print(f"  {label} Baseline    success={bc_metrics['success_rate'][0]:.1%}  "
              f"OOD={bc_metrics['ood_success_rate'][0]:.1%}")
    if dagger_metrics:
        print(f"  DAgger (final)   success={dagger_metrics['success_rate'][-1]:.1%}  "
              f"OOD={dagger_metrics['ood_success_rate'][-1]:.1%}")
    if ensemble_metrics:
        print(f"  Ensemble+HER     success={ensemble_metrics['success_rate'][-1]:.1%}  "
              f"OOD={ensemble_metrics['ood_success_rate'][-1]:.1%}")
    print(f"\n  Results → {os.path.abspath(config.log_dir)}/\n{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robotics self-improvement: Diffusion BC + SAC+HER online RL",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--policy",      default="diffusion", choices=["diffusion", "mlp"],
                        help="Policy architecture (diffusion = recommended)")
    parser.add_argument("--method",      default="rl",
                        help=(
                            "rl        → SAC+HER online RL (recommended, ~70-80%%)\n"
                            "ensemble  → Offline ensemble + HER self-improvement\n"
                            "dagger    → DAgger baseline\n"
                            "bc        → Pure BC baseline\n"
                            "all       → Run all methods"
                        ))
    parser.add_argument("--render",       action="store_true",
                        help="Render final policy in GUI window")
    parser.add_argument("--no-restrict",  action="store_true",
                        help="Don't restrict goal space during expert collection")
    parser.add_argument("--n-demos",      type=int, default=None,
                        help="Override n_expert_demos")
    parser.add_argument("--n-iter",       type=int, default=None,
                        help="Override n_iterations")
    parser.add_argument("--n-ensemble",   type=int, default=None,
                        help="Override ensemble size")
    parser.add_argument("--n-epochs",     type=int, default=None,
                        help="Override training epochs per policy (default 400, use 80 for quick test)")
    parser.add_argument("--n-rollouts",   type=int, default=None,
                        help="Override rollouts per iteration (default 300)")
    parser.add_argument("--n-eval",       type=int, default=None,
                        help="Override eval episodes (default 50)")
    parser.add_argument("--sac-steps",    type=int, default=None,
                        help="Override SAC total_timesteps")
    parser.add_argument("--quick",        action="store_true",
                        help=(
                            "Quick sanity-check mode (~5 min): "
                            "50 demos, 80 epochs, 1 ensemble, 100 rollouts, 2 iters, 20 eval"
                        ))
    parser.add_argument("--seed",         type=int, default=None)
    args = parser.parse_args()

    config = Config()
    config.policy_type = args.policy

    # --quick sets everything low for a fast smoke-test
    if args.quick:
        config.n_expert_demos     = 50
        config.n_epochs           = 80
        config.n_ensemble         = 1
        config.n_rollouts_per_iter = 100
        config.n_iterations       = 2
        config.n_eval_episodes    = 20
        config.save_video         = False
        print("  [--quick mode: 50 demos, 80 epochs, 1 policy, 100 rollouts, 2 iters]")

    if args.no_restrict:   config.restrict_goals      = False
    if args.n_demos:       config.n_expert_demos      = args.n_demos
    if args.n_iter:        config.n_iterations        = args.n_iter
    if args.n_ensemble:    config.n_ensemble          = args.n_ensemble
    if args.n_epochs:      config.n_epochs            = args.n_epochs
    if args.n_rollouts:    config.n_rollouts_per_iter = args.n_rollouts
    if args.n_eval:        config.n_eval_episodes     = args.n_eval
    if args.sac_steps:     config.sac_timesteps       = args.sac_steps
    if args.seed:          config.seed                = args.seed

    methods = ["all"] if args.method == "all" else [args.method]
    run(config, methods=methods, render=args.render)