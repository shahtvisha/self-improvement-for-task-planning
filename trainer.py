"""
trainer.py — Training loops for BCPolicy and DiffusionPolicy.

train_bc()        — MSE regression for BCPolicy (unchanged)
train_diffusion() — DDPM noise-prediction loss for DiffusionPolicy

The diffusion training loop is structurally identical to BC training
(DataLoader → forward → loss → backward) but:
  1. Uses DiffusionDataset instead of RoboticsDataset directly
     (so each sample is an action chunk, not a single action)
  2. Calls policy.compute_loss(obs, action_chunk) which internally
     adds noise and asks the denoiser to predict it
  3. Action normalisation: clips action chunks to [-1, 1] before training
     (actions should already be in this range for FetchPush, but clipping
     prevents rare outliers from destabilising the diffusion objective)
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple

from dataset          import RoboticsDataset
from diffusion_dataset import DiffusionDataset
from policy           import BCPolicy
from diffusion_policy  import DiffusionPolicy


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMAModel:
    """
    Exponential Moving Average of model weights.

    EMA weights produce smoother, more robust predictions at inference time —
    an implicit ensemble of all past checkpoints. Standard in every modern
    diffusion model pipeline (Ho 2020, Nichol & Dhariwal 2021, Chi 2023).

    decay=0.999 means the shadow weights update by 0.1% per step toward the
    current model — slow enough to average out gradient noise.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_p, m_p in zip(self.shadow.parameters(), model.parameters()):
            s_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        """Overwrite model parameters with EMA shadow (call after training)."""
        model.load_state_dict(self.shadow.state_dict())


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_normalization(dataset: RoboticsDataset) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Per-dimension mean and std of obs and actions from the replay buffer.

    FetchPush-v4 obs mixes gripper xyz (~1.3 m), velocities (~0.01 m/s),
    and finger widths (~0.05 m) — an unscaled mess for a linear FiLM layer.
    Normalising to zero-mean / unit-variance aligns all dimensions and
    dramatically improves gradient signal through the FiLM conditioning blocks.

    std is clamped to ≥ 0.01 so near-constant dims (gripper always open)
    don't blow up to inf.
    """
    obs_arr = np.stack(dataset._obs,     axis=0)
    act_arr = np.stack(dataset._actions, axis=0)
    obs_mean = torch.FloatTensor(obs_arr.mean(axis=0))
    obs_std  = torch.FloatTensor(obs_arr.std(axis=0)).clamp(min=0.01)
    act_mean = torch.FloatTensor(act_arr.mean(axis=0))
    act_std  = torch.FloatTensor(act_arr.std(axis=0)).clamp(min=0.01)
    return obs_mean, obs_std, act_mean, act_std


class NormDataset(torch.utils.data.Dataset):
    """Wrapper that returns normalised (obs, action) pairs at __getitem__."""
    def __init__(self, dataset, obs_mean, obs_std, act_mean, act_std):
        self._base   = dataset
        self.obs_mean = obs_mean;  self.obs_std  = obs_std
        self.act_mean = act_mean;  self.act_std  = act_std

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        obs, act = self._base[idx]
        return (obs - self.obs_mean) / self.obs_std, \
               (act - self.act_mean) / self.act_std

    @property
    def _episode_ends(self):
        return self._base._episode_ends


# ── BC training ───────────────────────────────────────────────────────────────

def train_bc(
    policy:       BCPolicy,
    dataset:      RoboticsDataset,
    n_epochs:     int   = 200,
    batch_size:   int   = 256,
    lr:           float = 3e-4,
    weight_decay: float = 1e-5,
    device:       str   = "cpu",
    verbose:      bool  = True,
) -> List[float]:
    """
    Behavioural Cloning: MSE regression on (obs, action) pairs.
    Returns per-epoch mean losses.
    """
    policy.to(device)
    policy.train()

    optimiser = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    drop_last = len(dataset) > batch_size
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    losses = []
    for epoch in range(n_epochs):
        batch_losses = []
        for obs_b, action_b in loader:
            obs_b    = obs_b.to(device)
            action_b = action_b.to(device)

            loss = criterion(policy(obs_b), action_b)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        scheduler.step()
        avg = sum(batch_losses) / len(batch_losses)
        losses.append(avg)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | BC loss: {avg:.5f}")

    policy.eval()
    return losses


# ── Diffusion training ────────────────────────────────────────────────────────

def train_diffusion(
    policy:        DiffusionPolicy,
    dataset:       RoboticsDataset,
    n_epochs:      int   = 200,
    batch_size:    int   = 256,
    lr:            float = 1e-4,
    weight_decay:  float = 1e-6,
    warmup_epochs: int   = 20,
    ema_decay:     float = 0.999,
    device:        str   = "cpu",
    verbose:       bool  = True,
) -> List[float]:
    """
    Diffusion Policy training with three key improvements over the baseline:

    1. Observation & action normalisation
       Computed from the dataset and applied before chunking, so both the
       FiLM conditioning signal (obs) and the denoising target (action chunk)
       live in a well-scaled space. Normalization stats are stored on the
       policy object for use at inference time.

    2. EMA (Exponential Moving Average) of weights  [decay=0.999]
       Updated every gradient step. At the end of training the EMA shadow
       weights are copied back to the policy — these are what inference uses.
       EMA noticeably reduces variance in the generated action distribution.

    3. Linear LR warmup + cosine decay
       Ramp from 0 → lr over warmup_epochs, then cosine anneal to 1e-6.
       Prevents large, destabilising gradient steps early in training when
       the denoiser is randomly initialised.
    """
    # ── Normalisation ────────────────────────────────────────────────────────
    obs_mean, obs_std, act_mean, act_std = compute_normalization(dataset)
    norm_dataset = NormDataset(dataset, obs_mean, obs_std, act_mean, act_std)

    # Store normalisation stats on the policy for use at inference
    policy.obs_mean = obs_mean
    policy.obs_std  = obs_std
    policy.act_mean = act_mean
    policy.act_std  = act_std

    # ── Chunk dataset ────────────────────────────────────────────────────────
    chunk_dataset = DiffusionDataset(
        norm_dataset,
        chunk_size=policy.chunk_size,
        action_dim=policy.action_dim,
    )

    if len(chunk_dataset) == 0:
        print("    Warning: DiffusionDataset has 0 valid chunks — skipping training.")
        return []

    if verbose:
        print(f"    DiffusionDataset: {chunk_dataset}")

    policy.to(device)
    # Move normalisation buffers to device
    policy.obs_mean = obs_mean.to(device)
    policy.obs_std  = obs_std.to(device)
    policy.act_mean = act_mean.to(device)
    policy.act_std  = act_std.to(device)
    policy.train()

    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
        return max(1e-6 / lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    ema = EMAModel(policy, decay=ema_decay)

    drop_last = len(chunk_dataset) > batch_size
    loader    = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    losses = []
    for epoch in range(n_epochs):
        policy.train()
        batch_losses = []
        for obs_b, chunk_b in loader:
            obs_b   = obs_b.to(device)
            chunk_b = chunk_b.to(device)
            # After normalisation, clip to ±3σ for numerical safety
            chunk_b = chunk_b.clamp(-3.0, 3.0)

            loss = policy.compute_loss(obs_b, chunk_b)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            ema.update(policy)      # ← EMA step after every gradient update
            batch_losses.append(loss.item())

        scheduler.step()
        avg = sum(batch_losses) / len(batch_losses)
        losses.append(avg)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | Diffusion loss: {avg:.5f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")

    # Copy EMA weights into policy — this is what evaluation and rollouts use
    ema.copy_to(policy)
    policy.eval()
    return losses


# ── Unified interface ─────────────────────────────────────────────────────────

def train_policy(policy, dataset, config, device: str = "cpu", verbose: bool = True):
    """
    Dispatch to the correct training function based on policy type.
    Keeps calling code clean — just call train_policy() regardless of type.
    """
    if isinstance(policy, DiffusionPolicy):
        return train_diffusion(
            policy, dataset,
            n_epochs=config.n_epochs, batch_size=config.batch_size,
            lr=config.diffusion_lr, weight_decay=config.weight_decay,
            warmup_epochs=getattr(config, "warmup_epochs", 20),
            ema_decay=getattr(config, "ema_decay", 0.999),
            device=device, verbose=verbose,
        )
    else:
        return train_bc(
            policy, dataset,
            n_epochs=config.n_epochs, batch_size=config.batch_size,
            lr=config.lr, weight_decay=config.weight_decay,
            device=device, verbose=verbose,
        )