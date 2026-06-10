"""
config.py — Hyperparameter configuration.

Three modes:
  policy_type = "mlp"       → BCPolicy (fast, ~15 min full run)
  policy_type = "diffusion" → DiffusionPolicy x0-prediction (recommended, ~60 min)

Three methods (--method flag):
  "ensemble"  → Offline ensemble + HER self-improvement (BC cold start)
  "rl"        → SAC + HER online RL (THE solution — achieves 70-80%)
  "dagger"    → DAgger baseline (upper bound for offline methods)
  "all"       → Run all three

Quick sanity check:
  python main.py --policy diffusion --method ensemble --n-iter 2 --n-demos 50

Full RL run (recommended, ~30-60 min):
  python main.py --method rl
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    # FetchPickAndPlace-v4: harder than FetchPush — robot must grasp the block,
    # lift it, and place it at a floating 3D goal (not just push on a table).
    env_id: str = "FetchPickAndPlace-v4"
    seed: int = 42

    # ── Policy type ──────────────────────────────────────────────────────────
    # "mlp"       → BCPolicy MLP (fast baseline)
    # "diffusion" → DiffusionPolicy with action chunking (recommended)
    policy_type: str = "diffusion"

    # ── Observation history ───────────────────────────────────────────────────
    # Stack the last obs_horizon observations before passing to policy.
    # Chi et al. ablation: obs_horizon=2 is the single biggest win for
    # state-based tasks — implicitly encodes velocity and task phase without
    # requiring an LSTM or recurrent architecture.
    # obs_dim passed to policy = obs_body_dim * obs_horizon + goal_dim
    obs_horizon: int = 2

    # ── Expert Demo Collection ───────────────────────────────────────────────
    n_expert_demos: int = 200
    expert_noise: float = 0.03
    restrict_goals: bool = True
    goal_restrict_axis: int = 0
    goal_restrict_threshold: float = 1.35

    # ── MLP Policy (BCPolicy) ────────────────────────────────────────────────
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])

    # ── Diffusion Policy ─────────────────────────────────────────────────────
    # chunk_size=8: empirically optimal for most manipulation tasks (Chi 2023 ablation)
    chunk_size:          int   = 16     # larger chunk for PickAndPlace's longer action sequences
    diffusion_T:         int   = 100    # forward diffusion steps (training)
    diffusion_inf_steps: int   = 5      # DDIM steps (inference) — 5 is fast, 10 is higher quality
    diffusion_hidden:    int   = 256    # denoiser hidden dim
    diffusion_layers:    int   = 4      # number of FiLM blocks
    diffusion_time_emb:  int   = 64     # sinusoidal time embedding dim
    diffusion_lr:        float = 1e-4   # lower LR than BC

    # EMA decay — 0.999 standard for robotics diffusion (slower than image diffusion)
    ema_decay:           float = 0.999

    # LR warmup before cosine decay
    warmup_epochs:       int   = 20

    # ── BC Training (shared) ─────────────────────────────────────────────────
    lr: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 400          # bump for diffusion with EMA + warmup
    weight_decay: float = 1e-5

    # ── SAC + HER (online RL — recommended) ──────────────────────────────────
    sac_timesteps:     int   = 500_000   # ~30 min CPU; 70-80% success
    sac_lr:            float = 1e-3
    sac_gamma:         float = 0.98
    sac_tau:           float = 0.05
    sac_buffer_size:   int   = 1_000_000
    sac_n_her_goals:   int   = 4        # HER "future" relabeling per transition
    sac_learning_starts: int = 1_000

    # ── Ensemble Self-Improvement ────────────────────────────────────────────
    n_ensemble: int = 3
    n_iterations: int = 6
    n_rollouts_per_iter: int = 300

    # ── Evaluation ───────────────────────────────────────────────────────────
    n_eval_episodes: int = 50
    render_eval: bool = False
    save_video: bool = True

    # ── ACT Policy ───────────────────────────────────────────────────────────
    # Zhao et al. 2023 — CVAE + Transformer encoder-decoder.
    # chunk_size shared with diffusion above.
    act_hidden:     int   = 256    # transformer hidden dim
    act_n_heads:    int   = 8      # attention heads (hidden must be divisible)
    act_enc_layers: int   = 4      # transformer encoder depth
    act_dec_layers: int   = 4      # transformer decoder depth
    act_latent_dim: int   = 32     # CVAE latent z dimension
    act_beta:       float = 1.0    # KL weight in ELBO loss
    act_dropout:    float = 0.1    # transformer dropout
    act_lr:         float = 1e-4   # AdamW lr (same as diffusion)

    # ── Output ───────────────────────────────────────────────────────────────
    log_dir: str = "results"
    verbose: bool = True