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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from dataset          import RoboticsDataset
from diffusion_dataset import DiffusionDataset
from policy           import BCPolicy
from diffusion_policy  import DiffusionPolicy


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
    policy:       DiffusionPolicy,
    dataset:      RoboticsDataset,
    n_epochs:     int   = 200,
    batch_size:   int   = 256,
    lr:           float = 1e-4,          # lower LR than BC — diffusion is sensitive
    weight_decay: float = 1e-6,
    device:       str   = "cpu",
    verbose:      bool  = True,
) -> List[float]:
    """
    Diffusion Policy training: DDPM noise-prediction loss.

    Uses DiffusionDataset to produce (obs, action_chunk) pairs that
    respect episode boundaries. Each epoch iterates over all valid chunks.

    The loss is: E_t E_ε || ε - ε_θ(√ᾱ_t·chunk + √(1-ᾱ_t)·ε, obs, t) ||²

    Args:
        policy:       DiffusionPolicy to train
        dataset:      RoboticsDataset (boundaries already tracked)
        n_epochs:     training epochs
        batch_size:   mini-batch size
        lr:           Adam learning rate (lower than BC)
        weight_decay: L2 regularisation
        device:       torch device
        verbose:      print loss every 50 epochs
    """
    chunk_dataset = DiffusionDataset(
        dataset,
        chunk_size=policy.chunk_size,
        action_dim=policy.action_dim,
    )

    if len(chunk_dataset) == 0:
        print("    Warning: DiffusionDataset has 0 valid chunks — skipping training.")
        return []

    if verbose:
        print(f"    DiffusionDataset: {chunk_dataset}")

    policy.to(device)
    policy.train()

    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs, eta_min=1e-6)

    drop_last = len(chunk_dataset) > batch_size
    loader    = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    losses = []
    for epoch in range(n_epochs):
        batch_losses = []
        for obs_b, chunk_b in loader:
            obs_b   = obs_b.to(device)
            chunk_b = chunk_b.to(device)

            # Clip actions to [-1, 1] — diffusion expects normalised inputs
            chunk_b = chunk_b.clamp(-1.0, 1.0)

            loss = policy.compute_loss(obs_b, chunk_b)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            batch_losses.append(loss.item())

        scheduler.step()
        avg = sum(batch_losses) / len(batch_losses)
        losses.append(avg)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs} | Diffusion loss: {avg:.5f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")

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
            device=device, verbose=verbose,
        )
    else:
        return train_bc(
            policy, dataset,
            n_epochs=config.n_epochs, batch_size=config.batch_size,
            lr=config.lr, weight_decay=config.weight_decay,
            device=device, verbose=verbose,
        )