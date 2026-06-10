"""
make_comparison_gif.py — 3-panel GIF: SAC+HER | Diffusion Policy | ACT

All three policies run the same episodes (same seeds) for a fair comparison.
Each panel shows a coloured label bar: green = success, red = failure.

Usage:
    python make_comparison_gif.py

Output:
    results/comparison.gif
"""

import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import torch

try:
    import imageio
except ImportError:
    raise ImportError("Run: pip install imageio")


SAC_MODEL_PATH      = "results/sac_best/best_model"
DIFFUSION_MODEL_PATH = "results/distilled_diffusion.pt"
ACT_MODEL_PATH       = "results/distilled_act.pt"
OUT_GIF              = "results/comparison.gif"
ENV_ID               = "FetchPickAndPlace-v4"
N_EPISODES           = 5
FPS                  = 30


# ── Label bar ─────────────────────────────────────────────────────────────────

def add_label(frame: np.ndarray, text: str, success=None) -> np.ndarray:
    """Overlay a coloured text bar at the top of a frame using PIL."""
    if success:
        bg = (56, 142, 60)      # green
    elif success is not None:
        bg = (198, 40, 40)      # red
    else:
        bg = (50, 50, 50)       # grey (unknown)

    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    W, H = img.size
    BAR  = 24

    draw.rectangle([(0, 0), (W, BAR)], fill=bg)

    bbox = draw.textbbox((0, 0), text)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (BAR - th) // 2), text, fill=(255, 255, 255))

    return np.array(img)


# ── Load policies ─────────────────────────────────────────────────────────────

def load_sac():
    from stable_baselines3 import SAC
    if not os.path.exists(SAC_MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"SAC model not found: {SAC_MODEL_PATH}.zip")
    _env  = gym.make(ENV_ID, render_mode=None)
    model = SAC.load(SAC_MODEL_PATH, env=_env)
    _env.close()
    print(f"  ✓  SAC loaded")
    return model


def load_diffusion():
    from config           import Config
    from diffusion_policy import DiffusionPolicy

    if not os.path.exists(DIFFUSION_MODEL_PATH):
        raise FileNotFoundError(
            f"Diffusion model not found: {DIFFUSION_MODEL_PATH}\n"
            "Run: python distill.py  first."
        )
    ckpt   = torch.load(DIFFUSION_MODEL_PATH, map_location="cpu")
    config = Config()
    policy = DiffusionPolicy(
        obs_dim           = ckpt["obs_dim"],
        action_dim        = ckpt["act_dim"],
        chunk_size        = config.chunk_size,
        T                 = config.diffusion_T,
        n_inference_steps = config.diffusion_inf_steps,
        hidden_dim        = config.diffusion_hidden,
        n_layers          = config.diffusion_layers,
        time_emb_dim      = config.diffusion_time_emb,
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.obs_mean = ckpt.get("obs_mean")
    policy.obs_std  = ckpt.get("obs_std")
    policy.act_mean = ckpt.get("act_mean")
    policy.act_std  = ckpt.get("act_std")
    policy.eval()
    sr = ckpt.get("success_rate", "?")
    print(f"  ✓  Diffusion loaded  (trained success={sr:.1%})")
    return policy


def load_act():
    from config     import Config
    from act_policy import ACTPolicy

    if not os.path.exists(ACT_MODEL_PATH):
        raise FileNotFoundError(
            f"ACT model not found: {ACT_MODEL_PATH}\n"
            "Run: python distill_act.py  first."
        )
    ckpt   = torch.load(ACT_MODEL_PATH, map_location="cpu")
    config = Config()
    policy = ACTPolicy(
        obs_dim      = ckpt["obs_dim"],
        action_dim   = ckpt["act_dim"],
        chunk_size   = config.chunk_size,
        hidden_dim   = config.act_hidden,
        n_heads      = config.act_n_heads,
        n_enc_layers = config.act_enc_layers,
        n_dec_layers = config.act_dec_layers,
        latent_dim   = config.act_latent_dim,
        beta         = config.act_beta,
        dropout      = config.act_dropout,
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.obs_mean = ckpt.get("obs_mean")
    policy.obs_std  = ckpt.get("obs_std")
    policy.act_mean = ckpt.get("act_mean")
    policy.act_std  = ckpt.get("act_std")
    policy.eval()
    sr = ckpt.get("success_rate", "?")
    print(f"  ✓  ACT loaded  (trained success={sr:.1%})")
    return policy


# ── Episode runners ───────────────────────────────────────────────────────────

def run_sac_episode(model, env, seed: int):
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    final_success = False
    frames = []
    while not (done or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        final_success = info.get("is_success", False)
    return frames, final_success


def run_chunk_episode(policy, env, seed: int, obs_horizon: int = 2, device: str = "cpu"):
    """Works for both DiffusionPolicy and ACTPolicy — same act_chunk interface."""
    obs, _ = env.reset(seed=seed)
    obs_buf = deque([obs["observation"].copy()] * obs_horizon, maxlen=obs_horizon)
    done = truncated = False
    final_success = False
    frames = []
    chunk = None
    chunk_idx = 0

    while not (done or truncated):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if chunk is None or chunk_idx >= policy.chunk_size:
            stacked = {
                "observation":   np.concatenate(list(obs_buf)),
                "achieved_goal": obs["achieved_goal"],
                "desired_goal":  obs["desired_goal"],
            }
            chunk     = policy.act_chunk(stacked, device=device)
            chunk_idx = 0

        action    = chunk[chunk_idx]
        chunk_idx += 1

        obs, _, done, truncated, info = env.step(action)
        obs_buf.append(obs["observation"].copy())
        final_success = info.get("is_success", False)

    return frames, final_success


# ── Build 3-panel frames ──────────────────────────────────────────────────────

def make_three_panels(
    frames_sac:  list,
    frames_diff: list,
    frames_act:  list,
    sac_ok:  bool,
    diff_ok: bool,
    act_ok:  bool,
) -> list:
    n = max(len(frames_sac), len(frames_diff), len(frames_act))

    def pad(frames, n):
        if not frames:
            return [np.zeros((200, 200, 3), dtype=np.uint8)] * n
        return frames + [frames[-1]] * (n - len(frames))

    frames_sac  = pad(frames_sac,  n)
    frames_diff = pad(frames_diff, n)
    frames_act  = pad(frames_act,  n)

    divider_color = 60
    result = []
    for fs, fd, fa in zip(frames_sac, frames_diff, frames_act):
        ls = add_label(fs, "SAC + HER",             sac_ok)
        ld = add_label(fd, "Diffusion Policy",       diff_ok)
        la = add_label(fa, "ACT",                    act_ok)

        # Match heights
        h = ls.shape[0]
        def resize_h(arr, h):
            if arr.shape[0] == h:
                return arr
            pil = Image.fromarray(arr).resize((arr.shape[1], h))
            return np.array(pil)
        ld = resize_h(ld, h)
        la = resize_h(la, h)

        div = np.full((h, 4, 3), divider_color, dtype=np.uint8)
        result.append(np.hstack([ls, div, ld, div, la]))

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  3-panel comparison GIF: SAC+HER | Diffusion | ACT")
    print(f"  Task: {ENV_ID}")
    print(f"  {N_EPISODES} episodes  ·  same seeds for all policies")
    print(f"{'═'*60}\n")

    sac_model   = load_sac()
    diff_policy = load_diffusion()
    act_policy  = load_act()

    sac_env  = gym.make(ENV_ID, render_mode="rgb_array")
    diff_env = gym.make(ENV_ID, render_mode="rgb_array")
    act_env  = gym.make(ENV_ID, render_mode="rgb_array")

    seeds = [ep * 17 + 42 for ep in range(N_EPISODES)]

    all_frames = []
    sac_wins = diff_wins = act_wins = 0

    for i, seed in enumerate(seeds):
        print(f"  Episode {i+1}/{N_EPISODES}  (seed={seed})...")

        sac_frames,  sac_ok  = run_sac_episode(sac_model,   sac_env,  seed)
        diff_frames, diff_ok = run_chunk_episode(diff_policy, diff_env, seed)
        act_frames,  act_ok  = run_chunk_episode(act_policy,  act_env,  seed)

        if sac_ok:  sac_wins  += 1
        if diff_ok: diff_wins += 1
        if act_ok:  act_wins  += 1

        print(f"    SAC {'✓' if sac_ok else '✗'}  |  "
              f"Diffusion {'✓' if diff_ok else '✗'}  |  "
              f"ACT {'✓' if act_ok else '✗'}")

        frames = make_three_panels(
            sac_frames, diff_frames, act_frames,
            sac_ok, diff_ok, act_ok,
        )
        all_frames.extend(frames)

        if i < N_EPISODES - 1 and all_frames:
            all_frames.extend([all_frames[-1]] * FPS)   # 1-second pause

    sac_env.close()
    diff_env.close()
    act_env.close()

    print(f"\n  Results over {N_EPISODES} episodes:")
    print(f"    SAC+HER:          {sac_wins}/{N_EPISODES}")
    print(f"    Diffusion Policy: {diff_wins}/{N_EPISODES}")
    print(f"    ACT:              {act_wins}/{N_EPISODES}")

    if all_frames:
        imageio.mimsave(OUT_GIF, all_frames, fps=FPS)
        size_mb = os.path.getsize(OUT_GIF) / 1_000_000
        print(f"\n  ✓  Saved → {OUT_GIF}  ({len(all_frames)} frames, {size_mb:.1f} MB)")
    else:
        print("  ⚠  No frames rendered.")


if __name__ == "__main__":
    main()