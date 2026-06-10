"""
eval_gif.py — 3-panel annotated GIF: SAC+HER  |  Diffusion Policy  |  ACT

Annotations per frame:
  - Top bar:    policy name + episode counter, green/red on success
  - Bottom bar: task description, step counter, goal-distance progress bar
  - End card:   0.75 s SUCCESS / FAILED overlay

Usage:
    python eval_gif.py
    python eval_gif.py --n-episodes 4 --fps 15 --frame-repeat 2
    python eval_gif.py --out results/eval.gif
"""

import argparse
import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import torch
import imageio


SAC_MODEL_PATH       = "results/sac_best/best_model"
DIFFUSION_MODEL_PATH = "results/distilled_diffusion.pt"
ACT_MODEL_PATH       = "results/distilled_act.pt"
ENV_ID               = "FetchPickAndPlace-v4"
TASK_TEXT            = "Grasp block  →  move to floating target (red sphere)"
SUCCESS_THRESHOLD    = 0.05
DIST_MAX             = 0.40

LABEL_H   = 42
INFO_H    = 54
PAD       = 10
DIVIDER   = 14

def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Geneva.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

FONT_LABEL = _load_font(16)
FONT_INFO  = _load_font(13)

BLACK       = (0,   0,   0)
SUCCESS_COL = (39,  174,  96)
FAILURE_COL = (192,  57,  43)
NEUTRAL_COL = (50,   50,  50)
BAR_EMPTY   = (60,   60,  60)
WHITE       = (255, 255, 255)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_text_centred(draw, text, rect, fill=WHITE, font=None):
    x0, y0, x1, y1 = rect
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - tw) // 2, y0 + (y1 - y0 - th) // 2), text, fill=fill, font=font)


def add_top_label(frame: np.ndarray, label: str, success) -> np.ndarray:
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    W, _ = img.size
    bg   = SUCCESS_COL if success else (FAILURE_COL if success is not None else NEUTRAL_COL)
    draw.rectangle([(0, 0), (W, LABEL_H)], fill=bg)
    _draw_text_centred(draw, label, (0, 0, W, LABEL_H), font=FONT_LABEL)
    return np.array(img)


def add_info_bar(frame: np.ndarray, step: int, max_steps: int, dist: float) -> np.ndarray:
    W        = frame.shape[1]
    bar_img  = Image.new("RGB", (W, INFO_H), color=BLACK)
    draw     = ImageDraw.Draw(bar_img)
    ROW1_Y   = 4                      # top row: task description
    ROW2_Y   = INFO_H // 2 + 2        # bottom row: step + distance bar

    # Row 1 — task description
    draw.text((8, ROW1_Y), TASK_TEXT, fill=(150, 150, 150), font=FONT_INFO)

    # Row 2 left — step counter
    step_txt = f"Step {step:>3}/{max_steps}"
    draw.text((8, ROW2_Y), step_txt, fill=(200, 200, 200), font=FONT_INFO)

    # Row 2 right — distance bar + label
    bar_w   = 120
    bar_h   = 10
    bar_x   = W - bar_w - 70
    bar_y   = ROW2_Y + 2
    fill_px = int(bar_w * max(0.0, 1.0 - dist / DIST_MAX))
    col     = SUCCESS_COL if dist <= SUCCESS_THRESHOLD else (39, 174, 96)

    draw.rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], fill=BAR_EMPTY)
    if fill_px > 0:
        draw.rectangle([(bar_x, bar_y), (bar_x + fill_px, bar_y + bar_h)], fill=col)
    draw.rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], outline=(100, 100, 100))

    dist_lbl = f"{dist*100:.1f}cm to goal"
    draw.text((bar_x + bar_w + 6, ROW2_Y), dist_lbl, fill=(180, 180, 180), font=FONT_INFO)

    return np.vstack([frame, np.array(bar_img)])


def end_card(frame: np.ndarray, success: bool) -> np.ndarray:
    img   = Image.fromarray(frame.copy())
    W, H  = img.size
    text  = "SUCCESS" if success else "FAILED"
    color = SUCCESS_COL if success else FAILURE_COL

    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = (W - tw) // 2, (H - th) // 2

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [(tx - 16, ty - 10), (tx + tw + 16, ty + th + 10)],
        fill=(*color, 210)
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(img).text((tx, ty), text, fill=WHITE)
    return np.array(img)


def compose_panels(*panels: np.ndarray) -> np.ndarray:
    h = max(p.shape[0] for p in panels)

    def pad_h(arr):
        diff = h - arr.shape[0]
        top  = np.zeros((diff // 2,        arr.shape[1], 3), dtype=np.uint8)
        bot  = np.zeros((diff - diff // 2, arr.shape[1], 3), dtype=np.uint8)
        return np.vstack([top, arr, bot])

    panels  = [pad_h(p) for p in panels]
    border  = np.zeros((h, PAD,     3), dtype=np.uint8)
    divider = np.zeros((h, DIVIDER, 3), dtype=np.uint8)

    row = border.copy()
    for i, p in enumerate(panels):
        row = np.hstack([row, p])
        row = np.hstack([row, divider if i < len(panels) - 1 else border])

    hborder = np.zeros((PAD, row.shape[1], 3), dtype=np.uint8)
    return np.vstack([hborder, row, hborder])


# ── Load models ───────────────────────────────────────────────────────────────

def load_sac():
    from stable_baselines3 import SAC
    _env  = gym.make(ENV_ID, render_mode=None)
    model = SAC.load(SAC_MODEL_PATH, env=_env)
    _env.close()
    print("  ✓  SAC loaded")
    return model


def load_diffusion():
    from config           import Config
    from diffusion_policy import DiffusionPolicy
    ckpt   = torch.load(DIFFUSION_MODEL_PATH, map_location="cpu", weights_only=False)
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
    print(f"  ✓  Diffusion loaded  (success={ckpt.get('success_rate', '?'):.1%})")
    return policy


def load_act():
    from config     import Config
    from act_policy import ACTPolicy
    ckpt   = torch.load(ACT_MODEL_PATH, map_location="cpu", weights_only=False)
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
    print(f"  ✓  ACT loaded  (success={ckpt.get('success_rate', '?'):.1%})")
    return policy


# ── Episode runners ───────────────────────────────────────────────────────────

def run_sac_episode(model, env, seed: int):
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    success = False
    records = []
    step = 0
    while not (done or truncated):
        dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        f    = env.render()
        if f is not None:
            records.append((f, dist, step))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, info = env.step(action)
        success = info.get("is_success", success)
        step += 1
    return records, success, step


def run_chunk_episode(policy, env, seed: int, obs_horizon: int = 2):
    obs, _ = env.reset(seed=seed)
    obs_buf   = deque([obs["observation"].copy()] * obs_horizon, maxlen=obs_horizon)
    done = truncated = False
    success = False
    records = []
    chunk = None
    chunk_idx = 0
    step = 0
    while not (done or truncated):
        dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        f    = env.render()
        if f is not None:
            records.append((f, dist, step))

        if chunk is None or chunk_idx >= policy.chunk_size:
            stacked = {
                "observation":   np.concatenate(list(obs_buf)),
                "achieved_goal": obs["achieved_goal"],
                "desired_goal":  obs["desired_goal"],
            }
            chunk     = policy.act_chunk(stacked, device="cpu")
            chunk_idx = 0

        obs, _, done, truncated, info = env.step(chunk[chunk_idx])
        chunk_idx += 1
        obs_buf.append(obs["observation"].copy())
        success = info.get("is_success", success)
        step += 1
    return records, success, step


# ── Annotate a sequence ───────────────────────────────────────────────────────

def annotate(records, max_steps: int, policy_name: str, success: bool,
             frame_repeat: int, fps: int, ep_label: str = "") -> list:
    label = f"{policy_name}  |  {ep_label}" if ep_label else policy_name
    out   = []
    for (frame, dist, step) in records:
        f = add_top_label(frame, label, None)
        f = add_info_bar(f, step, max_steps, dist)
        out.extend([f] * frame_repeat)

    if out:
        last = add_top_label(records[-1][0], label, success)
        last = add_info_bar(last, records[-1][2], max_steps, records[-1][1])
        card = end_card(last, success)
        out.extend([card] * int(fps * 0.75))

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs("results", exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  Eval GIF: SAC+HER  |  Diffusion  |  ACT")
    print(f"  Task: {ENV_ID}  ·  {args.n_episodes} episode(s)")
    print(f"  fps={args.fps}  frame_repeat={args.frame_repeat}")
    print(f"{'═'*60}\n")

    sac_model   = load_sac()
    diff_policy = load_diffusion()
    act_policy  = load_act()

    sac_env  = gym.make(ENV_ID, render_mode="rgb_array")
    diff_env = gym.make(ENV_ID, render_mode="rgb_array")
    act_env  = gym.make(ENV_ID, render_mode="rgb_array")

    seeds      = [ep * 17 + 42 for ep in range(args.n_episodes)]
    all_frames = []

    for i, seed in enumerate(seeds):
        print(f"  Episode {i+1}/{args.n_episodes}  (seed={seed})")

        sac_recs,  sac_ok,  sac_steps  = run_sac_episode(sac_model,   sac_env,  seed)
        diff_recs, diff_ok, diff_steps = run_chunk_episode(diff_policy, diff_env, seed)
        act_recs,  act_ok,  act_steps  = run_chunk_episode(act_policy,  act_env,  seed)

        print(f"    SAC {'✓' if sac_ok else '✗'} ({sac_steps}s)  |  "
              f"Diffusion {'✓' if diff_ok else '✗'} ({diff_steps}s)  |  "
              f"ACT {'✓' if act_ok else '✗'} ({act_steps}s)")

        max_steps = max(sac_steps, diff_steps, act_steps)
        ep_label  = f"Episode {i+1}/{args.n_episodes}"

        sac_ann  = annotate(sac_recs,  max_steps, "SAC + HER",       sac_ok,  args.frame_repeat, args.fps, ep_label)
        diff_ann = annotate(diff_recs, max_steps, "Diffusion Policy", diff_ok, args.frame_repeat, args.fps, ep_label)
        act_ann  = annotate(act_recs,  max_steps, "ACT",              act_ok,  args.frame_repeat, args.fps, ep_label)

        n = max(len(sac_ann), len(diff_ann), len(act_ann))
        def pad_seq(seq):
            return seq + [seq[-1]] * (n - len(seq)) if seq else \
                   [np.zeros((200, 200, 3), dtype=np.uint8)] * n

        for fs, fd, fa in zip(pad_seq(sac_ann), pad_seq(diff_ann), pad_seq(act_ann)):
            all_frames.append(compose_panels(fs, fd, fa))

    sac_env.close()
    diff_env.close()
    act_env.close()

    if all_frames:
        imageio.mimsave(args.out, all_frames, fps=args.fps)
        size_mb = os.path.getsize(args.out) / 1_000_000
        print(f"\n  ✓  Saved → {args.out}  ({len(all_frames)} frames, {size_mb:.1f} MB)")
    else:
        print("  ⚠  No frames rendered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes",   type=int, default=4)
    parser.add_argument("--fps",          type=int, default=15)
    parser.add_argument("--frame-repeat", type=int, default=2)
    parser.add_argument("--out",          type=str, default="results/eval.gif")
    main(parser.parse_args())
