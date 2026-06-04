"""
config.py — Hyperparameter configuration.

Two policy modes:
  policy_type = "mlp"       → BCPolicy (fast, ~15 min full run)
  policy_type = "diffusion" → DiffusionPolicy (best quality, ~60 min full run)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    env_id: str = "FetchPush-v4"
    seed: int = 42

    # ── Policy type ──────────────────────────────────────────────────────────
    # "mlp"       → BCPolicy MLP (fast baseline)
    # "diffusion" → DiffusionPolicy with action chunking (recommended)
    policy_type: str = "diffusion"

    # ── Expert Demo Collection ───────────────────────────────────────────────
    n_expert_demos: int = 150
    expert_noise: float = 0.03
    restrict_goals: bool = True
    goal_restrict_axis: int = 0
    goal_restrict_threshold: float = 1.35

    # ── MLP Policy (BCPolicy) ────────────────────────────────────────────────
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])

    # ── Diffusion Policy ─────────────────────────────────────────────────────
    chunk_size:          int   = 4      # actions predicted per query
    diffusion_T:         int   = 100    # forward diffusion steps (training)
    diffusion_inf_steps: int   = 10     # DDIM steps (inference, fast)
    diffusion_hidden:    int   = 256    # denoiser hidden dim
    diffusion_layers:    int   = 4      # number of FiLM blocks
    diffusion_time_emb:  int   = 64     # sinusoidal time embedding dim
    diffusion_lr:        float = 1e-4   # lower LR than BC

    # ── BC Training (shared) ─────────────────────────────────────────────────
    lr: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 200
    weight_decay: float = 1e-5

    # ── Ensemble Self-Improvement ────────────────────────────────────────────
    n_ensemble: int = 3
    n_iterations: int = 6
    n_rollouts_per_iter: int = 300

    # ── Evaluation ───────────────────────────────────────────────────────────
    n_eval_episodes: int = 50
    render_eval: bool = False
    save_video: bool = True

    # ── Output ───────────────────────────────────────────────────────────────
    log_dir: str = "results"
    verbose: bool = True