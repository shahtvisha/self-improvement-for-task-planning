"""
rl_finetune.py — SAC + HER online RL fine-tuning via stable-baselines3.

Why SAC+HER succeeds where offline diffusion BC fails
──────────────────────────────────────────────────────
Offline BC (including Diffusion Policy) is a self-improvement dead-end:
  - Start at 8% success → only ~24 successes per 300 rollouts
  - Those 24 episodes add ~1200 useful transitions vs thousands of HER noise
  - Dataset grows but signal-to-noise stays low → zero improvement

The fundamental issue is covariate shift: BC learns the expert's distribution,
but the policy visits states the expert never saw. Without correction, errors
compound exponentially across a 50-step episode.

SAC + HER breaks this cycle:
  - SAC is an off-policy actor-critic RL algorithm with entropy maximisation
  - It actively explores its own mistake states and learns from them via reward
  - HER (Hindsight Experience Replay) relabels every episode automatically —
    even total failures provide useful signal (the goal-reaching sub-problem)
  - Together they achieve 70-80% on FetchPush-v4 in ~500K environment steps

This is the canonical approach:
  Plappert et al. "Multi-Goal Reinforcement Learning: Challenging Robotics
  Environments and Request for Research" (2018) — achieved 80%+ on FetchPush
  with DDPG+HER. SAC converges faster and is more stable than DDPG.

Usage
──────
# From command line (recommended entry point):
  python main.py --method rl

# From Python:
  from rl_finetune import run_sac_her
  model, metrics = run_sac_her("FetchPush-v4", total_timesteps=500_000)

Install stable-baselines3 if needed:
  pip install stable-baselines3[extra]
"""

import os
import time
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401


def _check_sb3():
    try:
        import stable_baselines3  # noqa: F401
        return True
    except ImportError:
        return False


def run_sac_her(
    env_id:           str   = "FetchPush-v4",
    total_timesteps:  int   = 500_000,
    n_sampled_goal:   int   = 4,
    learning_starts:  int   = 1_000,
    batch_size:       int   = 256,
    learning_rate:    float = 1e-3,
    tau:              float = 0.05,
    gamma:            float = 0.98,
    buffer_size:      int   = 1_000_000,
    device:           str   = "cpu",
    log_dir:          str   = "results",
    verbose:          bool  = True,
    seed:             int   = 42,
) -> tuple:
    """
    Train SAC + HER from scratch on a goal-conditioned robotics environment.

    Hyperparameters are the standard ones used in the HER paper and SB3 zoo:
      - tau=0.05, gamma=0.98 for goal-conditioned tasks (shorter horizon)
      - n_sampled_goal=4 ("future" strategy) — each transition relabeled with
        4 randomly sampled future achieved goals from the same episode
      - learning_rate=1e-3 (higher than typical; works well for SAC+HER)

    Args:
        env_id:           Gymnasium environment ID
        total_timesteps:  total environment interactions (~500K for FetchPush)
        n_sampled_goal:   HER: how many relabeled goals per real transition
        learning_starts:  steps of random exploration before learning begins
        batch_size:       SAC mini-batch size
        learning_rate:    Adam LR for actor and critic
        tau:              soft target update rate
        gamma:            discount factor (0.98 = ~50 step effective horizon)
        buffer_size:      replay buffer capacity
        device:           "cpu" or "cuda"
        log_dir:          directory for eval logs and best model checkpoint
        verbose:          print training progress
        seed:             random seed

    Returns:
        (model, metrics_dict)
        model: the trained SB3 SAC model (or None if SB3 not installed)
        metrics: {"success_rate": float, "training_time_s": float}
    """
    if not _check_sb3():
        print(
            "\n  ╔══════════════════════════════════════════════════════╗\n"
            "  ║  stable-baselines3 not installed.                    ║\n"
            "  ║  Run: pip install stable-baselines3[extra]           ║\n"
            "  ╚══════════════════════════════════════════════════════╝\n"
        )
        return None, {}

    from stable_baselines3 import SAC
    from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

    # ── Environments ─────────────────────────────────────────────────────────
    env      = gym.make(env_id, render_mode=None)
    eval_env = gym.make(env_id, render_mode=None)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SAC + HER  [{env_id}]")
    print(f"  steps={total_timesteps:,}  lr={learning_rate}  gamma={gamma}  tau={tau}")
    print(f"  HER: n_sampled_goal={n_sampled_goal}  strategy=future")
    print(f"{'═'*60}\n")

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy="future",   # relabel with future goals in episode
        ),
        verbose=1 if verbose else 0,
        learning_starts=learning_starts,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tau=tau,
        gamma=gamma,
        buffer_size=buffer_size,
        device=device,
        seed=seed,
        tensorboard_log=os.path.join(log_dir, "tb_sac"),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "sac_best"),
        log_path=os.path.join(log_dir, "sac_eval"),
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
        verbose=0,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed/60:.1f} min")

    env.close()

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_path = os.path.join(log_dir, "sac_best", "best_model")
    if os.path.exists(best_path + ".zip"):
        print(f"  Loading best checkpoint: {best_path}.zip")
        model = SAC.load(best_path, env=gym.make(env_id, render_mode=None))

    # ── Final evaluation (100 episodes) ──────────────────────────────────────
    print(f"\n  Final evaluation (100 episodes)...")
    successes = []
    for ep in range(100):
        obs, _ = eval_env.reset(seed=9000 + ep)
        done = truncated = False
        ep_success = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = eval_env.step(action)
            if info.get("is_success", False):
                ep_success = True
        successes.append(float(ep_success))

    eval_env.close()
    final_sr = float(np.mean(successes))
    print(f"  ✓  SAC+HER success rate: {final_sr:.1%}  (expected ~70-80%)\n")

    metrics = {
        "success_rate":    final_sr,
        "training_time_s": elapsed,
    }
    return model, metrics


def sac_her_rollout_gif(
    model,
    env_id:     str = "FetchPush-v4",
    path:       str = "sac_rollout.gif",
    n_episodes: int = 3,
    fps:        int = 30,
):
    """Save a GIF of the SAC+HER policy rolling out in the environment."""
    try:
        import imageio
    except ImportError:
        print("  imageio not installed — skipping GIF")
        return

    env = gym.make(env_id, render_mode="rgb_array")
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        done = truncated = False
        while not (done or truncated):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

    env.close()
    if frames:
        imageio.mimsave(path, frames, fps=fps)
        print(f"  SAC rollout GIF → {path} ({len(frames)} frames)")