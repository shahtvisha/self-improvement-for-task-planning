"""
act_policy.py — Action Chunking with Transformers (ACT)

Paper: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with
       Low-Cost Hardware", RSS 2023.

Why ACT over plain BC?
──────────────────────
BC with MSE predicts the mean action for a state — wrong when multiple
valid actions exist. ACT learns a *style* latent z via a CVAE:

  Training:  z = CVAE_encoder(obs, action_chunk)  ← captures "how" to do it
             action_chunk = Transformer_decoder(obs, z)
             Loss = L2(pred, target) + β · KL(z || N(0,1))

  Inference: z = 0  (prior mean → deterministic, no sampling noise)
             action_chunk = Transformer_decoder(obs, 0)

The Transformer decoder learns to produce temporally consistent action
chunks (same chunking as our Diffusion Policy) by attending to the
compressed obs + style context.

Architecture
────────────
  CVAE encoder:       MLP(obs ⊕ action_chunk) → (μ, σ²) → z ∈ ℝ³²
  Transformer encoder: [obs_token, z_token] → memory  (2 tokens, 4 layers)
  Transformer decoder: chunk_size query tokens attend to memory → actions

State-based (no vision): obs projected linearly to hidden_dim — no CNN.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from policy import BCPolicy   # reuse _flatten_obs


class ACTPolicy(nn.Module):
    """
    ACT: CVAE-conditioned Transformer for action chunk prediction.

    Interface matches DiffusionPolicy:
      .act(obs_dict, device)       → single action np.ndarray (action_dim,)
      .act_chunk(obs_dict, device) → action chunk np.ndarray (chunk_size, action_dim)
      .compute_loss(obs, chunk)    → scalar training loss
    """

    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        chunk_size:   int   = 8,
        hidden_dim:   int   = 256,
        n_heads:      int   = 8,
        n_enc_layers: int   = 4,
        n_dec_layers: int   = 4,
        latent_dim:   int   = 32,
        beta:         float = 1.0,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.beta       = beta

        # Normalisation stats — set by train_act() from dataset statistics.
        self.obs_mean = None
        self.obs_std  = None
        self.act_mean = None
        self.act_std  = None

        # ── CVAE encoder (training only) ──────────────────────────────────
        # Maps (obs, flattened action_chunk) → style latent z.
        # The style variable captures *how* the expert solves the task
        # (e.g. approach angle, speed profile) — independent of *what* goal is.
        self.cvae_enc = nn.Sequential(
            nn.Linear(obs_dim + action_dim * chunk_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cvae_mu     = nn.Linear(hidden_dim, latent_dim)
        self.cvae_logvar = nn.Linear(hidden_dim, latent_dim)

        # ── Input projections ─────────────────────────────────────────────
        # One token per input type; seq_len = 2 for the encoder.
        self.obs_proj    = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Learnable positional encoding for the 2-token encoder sequence
        self.pos_enc = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        nn.init.normal_(self.pos_enc, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.tf_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ── Transformer decoder ───────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)

        # One learnable query token per timestep in the chunk
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        # ── Output head ───────────────────────────────────────────────────
        self.action_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "ACTPolicy.forward() should not be called directly. "
            "Use compute_loss() for training or act_chunk() for inference."
        )

    # ── CVAE helpers ──────────────────────────────────────────────────────

    def _encode_z(
        self,
        obs:          torch.Tensor,   # (B, obs_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size * action_dim) flattened
    ):
        h = self.cvae_enc(torch.cat([obs, action_chunk], dim=-1))
        return self.cvae_mu(h), self.cvae_logvar(h)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    # ── Forward ───────────────────────────────────────────────────────────

    def _decode(
        self,
        obs: torch.Tensor,   # (B, obs_dim)  already normalised
        z:   torch.Tensor,   # (B, latent_dim)
    ) -> torch.Tensor:       # (B, chunk_size, action_dim)
        B = obs.shape[0]

        obs_tok  = self.obs_proj(obs).unsqueeze(1)       # (B, 1, H)
        z_tok    = self.latent_proj(z).unsqueeze(1)      # (B, 1, H)
        enc_in   = torch.cat([obs_tok, z_tok], dim=1)   # (B, 2, H)
        enc_in   = enc_in + self.pos_enc                 # add positional enc

        memory   = self.tf_encoder(enc_in)               # (B, 2, H)

        queries  = self.query_embed.weight               # (chunk_size, H)
        queries  = queries.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, H)
        decoded  = self.tf_decoder(queries, memory)      # (B, chunk_size, H)

        return self.action_head(decoded)                 # (B, chunk_size, action_dim)

    # ── Training loss ─────────────────────────────────────────────────────

    def compute_loss(
        self,
        obs:          torch.Tensor,   # (B, obs_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size * action_dim) flattened
    ) -> torch.Tensor:
        """
        CVAE loss: L2 reconstruction + β·KL divergence.

        KL = -½ Σ (1 + log σ² - μ² - σ²)
        This regularises z toward N(0,I), ensuring the prior z=0 at
        inference is a good stand-in for the posterior.

        β=1.0 (standard ELBO).  Lower β → less regularisation, richer z
        but inference with z=0 degrades.  Higher β → z collapses to 0
        which makes the model deterministic but loses style diversity.
        """
        B = obs.shape[0]

        mu, logvar = self._encode_z(obs, action_chunk)
        z          = self._reparameterize(mu, logvar)

        target = action_chunk.view(B, self.chunk_size, self.action_dim)
        pred   = self._decode(obs, z)                    # (B, chunk_size, action_dim)

        l2 = F.mse_loss(pred, target)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return l2 + self.beta * kl

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def act_chunk(
        self,
        obs_dict: Dict[str, np.ndarray],
        device:   str = "cpu",
    ) -> np.ndarray:
        """
        Predict action chunk from obs dict.

        z is set to zero (prior mean) → fully deterministic at inference.
        The trained decoder has learned to produce good chunks from z=0
        because the KL term in training pushed the posterior toward N(0,I).
        """
        self.eval()
        flat  = BCPolicy._flatten_obs(obs_dict)
        obs_t = torch.FloatTensor(flat).unsqueeze(0).to(device)

        if self.obs_mean is not None:
            obs_t = (obs_t - self.obs_mean.to(device)) / self.obs_std.to(device)

        z     = torch.zeros(1, self.latent_dim, device=device)
        chunk = self._decode(obs_t, z)                   # (1, chunk_size, action_dim)

        if self.act_mean is not None:
            act_m = self.act_mean.to(device).view(1, 1, -1)
            act_s = self.act_std.to(device).view(1, 1, -1)
            chunk = chunk * act_s + act_m

        return chunk.squeeze(0).cpu().numpy()            # (chunk_size, action_dim)

    @torch.no_grad()
    def act(
        self,
        obs_dict: Dict[str, np.ndarray],
        device:   str = "cpu",
    ) -> np.ndarray:
        """Return only the first action of the chunk (single-step interface)."""
        return self.act_chunk(obs_dict, device)[0]