"""
diffusion_policy.py — Diffusion Policy for robot manipulation.

Paper: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
       Action Diffusion", RSS 2023.

Why diffusion instead of MSE regression?
─────────────────────────────────────────
Behavioural cloning with MSE loss forces the policy to output the *mean*
of all expert actions for a given state. When the expert can legitimately
take multiple different actions from the same state (e.g., approach the
block from the left OR from the right), MSE predicts the average — going
straight through the middle — which is wrong in both cases.

Diffusion Policy instead models the full *distribution* p(action | obs).
It learns to sample from that distribution via iterative denoising:
  - Training: corrupt action with Gaussian noise, learn to denoise it
  - Inference: start from pure noise, denoise conditioned on observation

This naturally handles multimodality — the network can represent
"50% probability go left, 50% probability go right" without collapsing
to the mean.

Action Chunking
───────────────
Instead of predicting one action at a time, we predict a *chunk* of
CHUNK_SIZE=4 consecutive actions and execute them all before re-querying
the policy. Benefits:
  1. Reduces the number of policy queries per episode (50 → ~12)
  2. Forces the policy to commit to a coherent sub-trajectory
  3. Each query "sees" more of the planned motion, reducing jitter
  4. Directly inspired by ACT (Zhao et al. 2023)

Architecture
────────────
Denoiser: FiLM-conditioned MLP
  Input:  noisy action chunk  (chunk_size × action_dim, flattened)
  Cond:   observation + sinusoidal time embedding
  Output: predicted noise (same shape as input)

Training: DDPM (Ho et al. 2020)
  - Cosine noise schedule (Nichol & Dhariwal 2021) — better than linear
  - T=100 forward-diffusion steps
  - Loss: MSE(predicted_noise, actual_noise)

Inference: DDIM (Song et al. 2020)
  - 10 deterministic denoising steps (vs 100 DDPM steps)
  - ~10× faster than DDPM at inference time
  - eta=0 → fully deterministic given obs
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from policy import BCPolicy   # reuse _flatten_obs


# ── Noise schedule ────────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Produces betas that start small and increase gradually,
    avoiding sudden collapse of signal at early steps.
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alpha_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-4, 0.9999)


# ── Building blocks ───────────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal timestep embedding — standard in diffusion models.
    Maps integer timestep t → continuous vector.
    Same formulation as "Attention is All You Need" positional encoding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) block.
    Conditions the hidden state on observation + timestep via learned
    per-feature scale and shift:
        h_out = scale(cond) * LayerNorm(h) + shift(cond)  +  residual(h)

    This is more expressive than simple concatenation because the condition
    can modulate the *magnitude* and *sign* of every hidden dimension,
    not just add to it.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)   # → (scale, shift)
        self.residual  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Initialise condition projection to near-zero so early training
        # behaves like an unconditional network (stability trick)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm(x) * (1.0 + scale) + shift
        return x + self.residual(h)


class DiffusionDenoiser(nn.Module):
    """
    The denoising network f_θ(x_t, t, obs).

    Given a noisy action chunk x_t at timestep t and the current
    observation, directly predicts the clean action chunk x0
    (x0-prediction / "v-prediction" variant).

    Why x0-prediction over ε-prediction for robot control?
    ───────────────────────────────────────────────────────
    For low-dimensional action spaces (4-dim here), ε-prediction suffers
    from the "high-t instability": at large timesteps almost all signal
    is noise, so the gradient becomes very noisy. x0-prediction always
    targets the clean action — a stable, bounded target in [-1,1] —
    which yields much more consistent gradients throughout training.

    Papers: Ramesh et al. (DALL-E 2, 2022), Salimans & Ho (2022)
    show x0-prediction is strictly more stable for small diffusion models.
    """

    def __init__(
        self,
        action_dim:   int,
        obs_dim:      int,
        chunk_size:   int = 4,
        hidden_dim:   int = 256,
        n_layers:     int = 4,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        flat_action_dim = action_dim * chunk_size

        # Timestep → embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.Mish(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        cond_dim = obs_dim + time_emb_dim

        # Input projection: noisy action chunk → hidden
        self.input_proj = nn.Sequential(
            nn.Linear(flat_action_dim, hidden_dim),
            nn.Mish(),
        )

        # FiLM blocks
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, cond_dim) for _ in range(n_layers)
        ])

        # Output projection: hidden → predicted x0 (clean action chunk)
        self.output_proj = nn.Linear(hidden_dim, flat_action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_chunk: torch.Tensor,   # (B, chunk_size * action_dim)
        obs:         torch.Tensor,   # (B, obs_dim)
        t:           torch.Tensor,   # (B,) integer timesteps
    ) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        cond  = torch.cat([obs, t_emb], dim=-1)
        x     = self.input_proj(noisy_chunk)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)


# ── Main policy class ─────────────────────────────────────────────────────────

class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy: action distribution modeled as a denoising process.

    Interface matches BCPolicy:
      .act(obs_dict, device)       → single action np.ndarray (4,)
      .act_chunk(obs_dict, device) → action chunk np.ndarray (chunk_size, 4)
      .compute_loss(obs, chunk)    → scalar training loss

    Training loss: DDPM noise prediction MSE
    Inference:     DDIM with n_inference_steps=10 (deterministic)
    """

    def __init__(
        self,
        obs_dim:           int,
        action_dim:        int,
        chunk_size:        int   = 4,
        T:                 int   = 100,
        n_inference_steps: int   = 10,
        hidden_dim:        int   = 256,
        n_layers:          int   = 4,
        time_emb_dim:      int   = 64,
    ):
        super().__init__()
        self.obs_dim           = obs_dim
        self.action_dim        = action_dim
        self.chunk_size        = chunk_size
        self.T                 = T
        self.n_inference_steps = n_inference_steps
        self.flat_action_dim   = action_dim * chunk_size

        self.denoiser = DiffusionDenoiser(
            action_dim=action_dim, obs_dim=obs_dim,
            chunk_size=chunk_size, hidden_dim=hidden_dim,
            n_layers=n_layers, time_emb_dim=time_emb_dim,
        )

        # Pre-compute and register noise schedule buffers
        betas = cosine_beta_schedule(T)
        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)

        # Pad with 1.0 at index -1 for DDIM (alpha_bar at step -1 = 1)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",                    betas)
        self.register_buffer("alphas_cumprod",           alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev",      alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",      alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt())

    # ── Training ──────────────────────────────────────────────────────────────

    def q_sample(
        self,
        x_start: torch.Tensor,
        t:       torch.Tensor,
        noise:   Optional[torch.Tensor] = None,
    ):
        """Forward diffusion: corrupt x_start to x_t at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha  = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_start + sqrt_1minus * noise, noise

    def compute_loss(
        self,
        obs:          torch.Tensor,   # (B, obs_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size * action_dim)
    ) -> torch.Tensor:
        """
        x0-prediction loss: E_t [ || x0 - f_θ(x_t, t, obs) ||² ]

        We corrupt the clean action chunk to x_t at a random timestep,
        then ask the denoiser to reconstruct x0 directly.

        Advantages over ε-prediction:
          - Target is always bounded in [-1, 1] → stable gradients
          - No gradient blowup at large t where noise dominates
          - Faster convergence for small action spaces (4-dim)
        """
        B = obs.shape[0]
        t = torch.randint(0, self.T, (B,), device=obs.device)

        noisy_chunk, _  = self.q_sample(action_chunk, t)
        x0_pred         = self.denoiser(noisy_chunk, obs, t)

        return F.mse_loss(x0_pred, action_chunk)

    # ── Inference (DDIM) ──────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(self, obs: torch.Tensor) -> torch.Tensor:
        """
        DDIM deterministic sampling with x0-prediction denoiser.

        At each step:
          1. Denoiser directly predicts x0_pred from x_t
          2. Imply eps from x_t and x0_pred (DDIM consistency)
          3. Step to x_{t-1} using DDIM update rule

        Returns action chunk: (B, chunk_size, action_dim).
        """
        B      = obs.shape[0]
        device = obs.device
        x      = torch.randn(B, self.flat_action_dim, device=device)

        # Evenly spaced timestep subsequence from T-1 → 0
        timesteps = torch.linspace(
            self.T - 1, 0, self.n_inference_steps + 1
        ).long().to(device)

        for i in range(self.n_inference_steps):
            t      = timesteps[i].expand(B)
            t_next = timesteps[i + 1]

            # x0 directly from denoiser
            x0_pred = self.denoiser(x, obs, t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            alpha      = self.alphas_cumprod[timesteps[i]]
            alpha_next = self.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0, device=device)

            # Implied epsilon (for DDIM direction term): eps = (x_t - sqrt(alpha)*x0) / sqrt(1-alpha)
            eps_implied = (x - alpha.sqrt() * x0_pred) / (1.0 - alpha).sqrt().clamp(min=1e-8)

            # DDIM update: x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1-alpha_{t-1}) * eps_implied
            x = alpha_next.sqrt() * x0_pred + (1.0 - alpha_next).sqrt() * eps_implied

        return x.reshape(B, self.chunk_size, self.action_dim)

    # ── Inference interface ───────────────────────────────────────────────────

    @torch.no_grad()
    def act_chunk(
        self,
        obs_dict: Dict[str, np.ndarray],
        device:   str = "cpu",
    ) -> np.ndarray:
        """
        Generate a full action chunk from obs dict.
        Returns: np.ndarray of shape (chunk_size, action_dim).
        """
        self.eval()
        flat = BCPolicy._flatten_obs(obs_dict)
        obs_t = torch.FloatTensor(flat).unsqueeze(0).to(device)
        chunk = self.ddim_sample(obs_t)
        return chunk.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def act(
        self,
        obs_dict: Dict[str, np.ndarray],
        device:   str = "cpu",
    ) -> np.ndarray:
        """Return only the first action of the chunk (single-step interface)."""
        return self.act_chunk(obs_dict, device)[0]