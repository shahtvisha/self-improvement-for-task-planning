"""
visualize.py — All plotting for the self-improvement project.

Three figures:
  1. comparison_plot()   — success rate curves for BC / DAgger / Self-Improve+HER
  2. heatmap_series()    — 2D goal-space success heatmaps at iterations 0,2,4,6
  3. dataset_growth()    — training data composition over iterations

The heatmap is the most visually striking output. It shows the policy's
competence literally expanding across the goal space as self-improvement runs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional
import os

import gymnasium as gym
import gymnasium_robotics  # noqa: F401

from policy import BCPolicy

# Custom colourmap: grey (0% success) → orange → green (100%)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "success", ["#BDBDBD", "#FF7043", "#66BB6A"], N=256
)

METHOD_COLORS = {
    "bc":           ("#9E9E9E", "--"),   # grey dashed
    "dagger":       ("#1565C0", "-"),    # blue solid
    "self_her":     ("#2E7D32", "-"),    # green solid
}
METHOD_LABELS = {
    "bc":       "BC Baseline (no self-improvement)",
    "dagger":   "DAgger (online expert required)",
    "self_her": "Self-Improve + HER (ours, no online expert)",
}


# ── Goal-space heatmap evaluation ─────────────────────────────────────────────

def evaluate_goal_space(
    policy:     BCPolicy,
    env_id:     str,
    n_episodes: int = 400,
    device:     str = "cpu",
    seed_offset: int = 0,
):
    """
    Run n_episodes with random goals, record (goal_x, goal_y, success).
    Returns raw result array for binning into a heatmap grid.
    """
    env = gym.make(env_id, render_mode=None)
    policy.eval()
    results = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        goal = obs["desired_goal"].copy()
        done = truncated = False
        success = False

        while not (done or truncated):
            action = policy.act(obs, device=device)
            obs, _, done, truncated, info = env.step(action)
            if info.get("is_success", False):
                success = True

        results.append((goal[0], goal[1], float(success)))

    env.close()
    return np.array(results)   # shape (n_episodes, 3): [goal_x, goal_y, success]


def bin_to_grid(results: np.ndarray, grid_size: int = 7):
    """Bin (goal_x, goal_y, success) results into a success-rate grid."""
    x, y, s = results[:, 0], results[:, 1], results[:, 2]
    x_edges = np.linspace(x.min() - 1e-6, x.max() + 1e-6, grid_size + 1)
    y_edges = np.linspace(y.min() - 1e-6, y.max() + 1e-6, grid_size + 1)
    grid = np.full((grid_size, grid_size), np.nan)

    for i in range(grid_size):
        for j in range(grid_size):
            mask = ((x >= x_edges[i]) & (x < x_edges[i + 1]) &
                    (y >= y_edges[j]) & (y < y_edges[j + 1]))
            if mask.sum() >= 2:
                grid[j, i] = s[mask].mean()

    return grid, x_edges, y_edges


# ── Figure 1: Three-way comparison plot ──────────────────────────────────────

def comparison_plot(
    bc_metrics:      dict,
    dagger_metrics:  dict,
    self_her_metrics: dict,
    out_dir:         str = "results",
    ood_only:        bool = False,
):
    """
    Plot success rate (and OOD success rate) vs iteration for all three methods.

    This is the main argument figure: shows self-improve+HER nearly matching
    DAgger without requiring the online expert, both far exceeding BC alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Self-Improving BC vs. Baselines\n"
        "FetchPush-v4  ·  50 expert demos, restricted goal distribution",
        fontsize=13,
    )

    for ax, key, ylabel, title in [
        (axes[0], "success_rate",     "Success Rate (%)", "Overall Success Rate"),
        (axes[1], "ood_success_rate", "Success Rate (%)", "OOD Goal Success Rate\n(goals never demonstrated)"),
    ]:
        for method, metrics in [("bc", bc_metrics), ("dagger", dagger_metrics), ("self_her", self_her_metrics)]:
            color, ls = METHOD_COLORS[method]
            label     = METHOD_LABELS[method]
            iters     = metrics["iteration"]
            vals      = [v * 100 for v in metrics[key]]

            # BC is just a single point (no self-improvement)
            if method == "bc":
                ax.axhline(vals[0], color=color, linestyle=ls, linewidth=2, label=label, alpha=0.8)
            else:
                ax.plot(iters, vals, color=color, linestyle=ls,
                        linewidth=2.5, marker="o", markersize=7, label=label)

        ax.set_xlabel("Self-Improvement Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)

    # Shade the OOD region annotation on the right plot
    axes[1].axvspan(-0.3, 0.3, alpha=0.07, color="red")
    axes[1].text(0, 5, "BC\nbaseline", ha="center", va="bottom",
                 fontsize=8, color="#9E9E9E")

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Comparison plot saved → {path}")
    plt.close()


# ── Figure 2: Goal-space heatmap series ──────────────────────────────────────

def heatmap_series(
    policies:    List[BCPolicy],
    iterations:  List[int],
    env_id:      str,
    device:      str = "cpu",
    n_episodes:  int = 400,
    grid_size:   int = 7,
    goal_axis:   int = 0,
    goal_threshold: float = 1.35,
    out_dir:     str = "results",
):
    """
    For each policy checkpoint, evaluate goal-space success and plot as a heatmap.

    Shows a series of heatmaps (one per iteration) side by side. The demo region
    boundary is drawn as a dashed line. You can visually see the policy's
    competence expanding outward from the demo region.

    Args:
        policies:   list of BCPolicy objects (one per iteration to visualise)
        iterations: corresponding iteration numbers (for titles)
        env_id:     gymnasium env id
        device:     torch device
        n_episodes: episodes per heatmap (more = smoother)
        grid_size:  NxN grid resolution
        goal_axis:  axis used for demo/OOD split (for boundary line)
        goal_threshold: threshold for demo/OOD split
        out_dir:    output directory
    """
    n_maps = len(policies)
    fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4.5))
    if n_maps == 1:
        axes = [axes]

    fig.suptitle(
        "Goal-Space Success Heatmap — Self-Improve + HER\n"
        "Each cell = success rate at that goal position  ·  dashed line = demo region boundary",
        fontsize=11,
    )

    all_grids = []
    all_x_edges = []
    all_y_edges = []

    print(f"\n  Evaluating goal-space heatmaps ({n_maps} checkpoints × {n_episodes} episodes each)...")
    for i, (policy, iteration) in enumerate(zip(policies, iterations)):
        print(f"  Heatmap {i+1}/{n_maps}: iteration {iteration}...")
        results = evaluate_goal_space(policy, env_id, n_episodes=n_episodes,
                                      device=device, seed_offset=i * 10000)
        grid, x_edges, y_edges = bin_to_grid(results, grid_size=grid_size)
        all_grids.append(grid)
        all_x_edges.append(x_edges)
        all_y_edges.append(y_edges)

    # Use consistent colour scale across all heatmaps
    vmin, vmax = 0.0, 1.0

    for i, (ax, grid, x_edges, y_edges, iteration) in enumerate(
        zip(axes, all_grids, all_x_edges, all_y_edges, iterations)
    ):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            cmap=HEATMAP_CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        # Draw demo region boundary
        if goal_axis == 0:   # vertical line on x
            ax.axvline(goal_threshold, color="white", linestyle="--", linewidth=1.8,
                       label=f"Demo boundary (x={goal_threshold})")
            ax.text(goal_threshold + 0.01, y_edges[0] + 0.02, "demo\nregion",
                    color="white", fontsize=7, va="bottom")
            ax.text(goal_threshold - 0.01, y_edges[0] + 0.02, "OOD",
                    color="white", fontsize=7, va="bottom", ha="right")
        elif goal_axis == 1:  # horizontal line on y
            ax.axhline(goal_threshold, color="white", linestyle="--", linewidth=1.8)

        title = "BC Baseline" if iteration == 0 else f"Iteration {iteration}"
        ax.set_title(title, fontsize=10, fontweight="bold" if iteration == 0 else "normal")
        ax.set_xlabel("Goal x (table)")
        if i == 0:
            ax.set_ylabel("Goal y (table)")

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Success Rate", fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    path = os.path.join(out_dir, "heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Heatmap figure saved → {path}")
    plt.close()


# ── Figure 3: Dataset composition growth ─────────────────────────────────────

def dataset_growth_plot(metrics: dict, out_dir: str = "results"):
    """Stacked bar chart showing how the dataset composition changes per iteration."""
    iters = metrics["iteration"]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Training Dataset Composition Over Self-Improvement Iterations", fontsize=11)

    expert_counts = metrics.get("n_expert_transitions", [0] * len(iters))
    success_counts = metrics.get("n_success_transitions", [0] * len(iters))
    her_counts = metrics.get("n_her_transitions", [0] * len(iters))

    ax.bar(iters, expert_counts, label="Expert demos", color="#1976D2", alpha=0.85)
    ax.bar(iters, success_counts, bottom=expert_counts,
           label="Self-rollout successes", color="#43A047", alpha=0.85)
    bottoms = [e + s for e, s in zip(expert_counts, success_counts)]
    ax.bar(iters, her_counts, bottom=bottoms,
           label="HER relabeled failures", color="#FB8C00", alpha=0.85)

    ax.set_xlabel("Self-Improvement Iteration")
    ax.set_ylabel("Number of Transitions")
    ax.set_xticks(iters)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    path = os.path.join(out_dir, "dataset_growth.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Dataset growth plot saved → {path}")
    plt.close()