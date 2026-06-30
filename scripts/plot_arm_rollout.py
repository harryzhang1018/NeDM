"""Plot open-loop EE trajectories of the arm dynamics model vs ground truth.

Seeds each selected val episode with the first ``context`` recorded steps, rolls the model
open-loop (``next = state + predict_next_delta`` while feeding the recorded actions), and
overlays the predicted end-effector path on the Chrono ground-truth path. Produces two
figures: a grid of 3-D EE trajectories, and per-axis (x/y/z) time series showing where the
prediction drifts. Companion to ``scripts/eval_arm_rollout.py`` (which reports the numbers).

    PYTHONPATH=src python scripts/plot_arm_rollout.py \
        --checkpoint artifacts/training_runs/arm_transformer_full_v1 --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.dynamics import load_frozen_dynamics
from nedm.training.dataset import load_rollout_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot arm open-loop EE rollout vs ground truth.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Run dir or .pt checkpoint.")
    parser.add_argument("--processed-dataset-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-episodes", type=int, default=9, help="Episodes in the 3-D grid.")
    parser.add_argument("--num-axes", type=int, default=4, help="Episodes in the per-axis figure.")
    parser.add_argument("--min-steps", type=int, default=100, help="Min rollout steps to qualify.")
    parser.add_argument("--max-steps", type=int, default=250, help="Cap on rollout steps per episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write PNGs (default: run dir).")
    return parser.parse_args(argv)


@torch.no_grad()
def rollout_ee(model, states: torch.Tensor, actions: torch.Tensor, context: int,
               max_steps: int, ee_idx: torch.Tensor) -> np.ndarray:
    """Open-loop predicted ee_base over the rollout horizon, shape (steps, 3)."""
    steps = min(max_steps, states.shape[0] - context)
    hist_s = states[:context].clone()
    hist_a = actions[:context].clone()
    pred = []
    for step in range(steps):
        delta = model.predict_next_delta(
            hist_s[-context:].unsqueeze(0), hist_a[-context:].unsqueeze(0), terrain=None
        ).squeeze(0)
        next_state = hist_s[-1] + delta
        pred.append(next_state[ee_idx])
        if context + step < actions.shape[0]:
            hist_a = torch.cat([hist_a, actions[context + step].unsqueeze(0)], dim=0)
        hist_s = torch.cat([hist_s, next_state.unsqueeze(0)], dim=0)
    return torch.stack(pred, dim=0).cpu().numpy()


def collect(model, episode, context, max_steps, ee_idx, device):
    """Return dict with context/gt/pred EE arrays (all in ee_base coords) for one episode."""
    states = torch.from_numpy(episode["states"]).to(device)
    actions = torch.from_numpy(episode["actions"]).to(device)
    steps = min(max_steps, states.shape[0] - context)
    ee_idx_np = ee_idx.cpu().numpy()
    states_np = states.cpu().numpy()
    pred = rollout_ee(model, states, actions, context, max_steps, ee_idx)
    anchor = states_np[context - 1][ee_idx_np]            # last seeded EE, shared start
    gt_full = states_np[: context + steps][:, ee_idx_np]  # full gt path (seed + rollout)
    pred_path = np.concatenate([anchor[None, :], pred], axis=0)  # anchored predicted path
    return {
        "id": episode["episode_id"],
        "family": episode.get("scenario_family", ""),
        "context": context,
        "gt_full": gt_full,
        "pred_path": pred_path,
        "anchor": anchor,
        "final_err": float(np.linalg.norm(pred[-1] - gt_full[-1])),
        "steps": steps,
    }


def plot_traj3d(items, dt, out_path):
    n = len(items)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(5 * cols, 4.3 * rows))
    for i, it in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ctx = it["context"]
        gt = it["gt_full"]
        seed, gt_roll = gt[:ctx], gt[ctx - 1:]
        pred = it["pred_path"]
        ax.plot(seed[:, 0], seed[:, 1], seed[:, 2], color="0.6", lw=1.0, label="seed (context)")
        ax.plot(gt_roll[:, 0], gt_roll[:, 1], gt_roll[:, 2], color="tab:blue", lw=1.8, label="ground truth")
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color="tab:red", lw=1.6, ls="--", label="predicted (open-loop)")
        ax.scatter(*it["anchor"], color="green", s=30, label="rollout start")
        ax.scatter(*gt_roll[-1], color="tab:blue", marker="X", s=40)
        ax.scatter(*pred[-1], color="tab:red", marker="X", s=40)
        # honest box aspect from data extent
        allp = np.concatenate([gt, pred], axis=0)
        span = (allp.max(0) - allp.min(0))
        span[span < 1e-6] = 1e-6
        ax.set_box_aspect(span)
        ax.set_title(f"{it['id']}  ({it['family']})\n{it['steps']} steps / {it['steps']*dt:.1f}s  "
                     f"final EE err={it['final_err']:.3f}", fontsize=8)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left")
    fig.suptitle("Arm open-loop EE rollout vs ground truth (ee_base, scaled-arm meters)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_axes(items, dt, out_path):
    items = items[: ]  # already truncated by caller
    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.6 * n), squeeze=False)
    labels = ["ee_base x", "ee_base y", "ee_base z"]
    for r, it in enumerate(items):
        ctx = it["context"]
        gt = it["gt_full"]
        pred = it["pred_path"]
        t_gt = np.arange(gt.shape[0]) * dt
        t_pred = np.arange(ctx - 1, ctx - 1 + pred.shape[0]) * dt
        t_split = (ctx - 1) * dt
        for c in range(3):
            ax = axes[r][c]
            ax.plot(t_gt, gt[:, c], color="tab:blue", lw=1.6, label="ground truth")
            ax.plot(t_pred, pred[:, c], color="tab:red", lw=1.4, ls="--", label="predicted")
            ax.axvline(t_split, color="0.6", ls=":", lw=1.0)
            if r == 0:
                ax.set_title(labels[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{it['id']}\n[m]", fontsize=8)
            if r == n - 1:
                ax.set_xlabel("time [s]", fontsize=9)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 2:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle("Arm open-loop EE rollout per axis (dotted line = rollout start)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dyn = load_frozen_dynamics(args.checkpoint, device=args.device,
                               processed_dataset_dir=args.processed_dataset_dir)
    context = dyn.context_steps
    dt = dyn.dt_s
    state_fields = list(dyn.metadata["state_fields"])
    ee_idx = torch.tensor([state_fields.index(f"ee_base_{a}") for a in "xyz"],
                          dtype=torch.long, device=args.device)

    processed_dir = Path(args.processed_dataset_dir if args.processed_dataset_dir is not None
                         else dyn.config["processed_dataset_dir"]).resolve()
    episodes = load_rollout_split(processed_dir, args.split)["episodes"]

    qualifying = [e for e in episodes if e["states"].shape[0] - context >= args.min_steps]
    if not qualifying:
        raise SystemExit(f"no val episodes with >= {args.min_steps} rollout steps")
    rng = np.random.default_rng(args.seed)
    k = min(args.num_episodes, len(qualifying))
    picks = [qualifying[int(i)] for i in rng.choice(len(qualifying), size=k, replace=False)]

    items = [collect(dyn.model, e, context, args.max_steps, ee_idx, args.device) for e in picks]
    items_sorted = sorted(items, key=lambda x: x["steps"], reverse=True)

    out_dir = Path(args.output_dir) if args.output_dir is not None else dyn.checkpoint_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    p3d = out_dir / "rollout_traj3d.png"
    pax = out_dir / "rollout_axes.png"
    plot_traj3d(items, dt, p3d)
    plot_axes(items_sorted[: args.num_axes], dt, pax)

    errs = np.array([it["final_err"] for it in items])
    print(f"plotted {len(items)} episodes from {processed_dir.name} ({args.split} split)")
    print(f"  final EE error over plotted episodes: mean={errs.mean():.3f} "
          f"min={errs.min():.3f} max={errs.max():.3f} (ee_base scaled-arm meters)")
    print(f"  wrote {p3d}")
    print(f"  wrote {pax}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
