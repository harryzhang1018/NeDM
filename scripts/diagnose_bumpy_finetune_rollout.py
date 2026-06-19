from __future__ import annotations

"""Decisive diagnosis of why bumpy fine-tuning destroys open-loop rollout.

Hypothesis: one-step body predictions stay accurate after fine-tuning, but the
recurrent tire normal-force (Fz) channel drifts out of the flat normalization
range during autoregressive rollout and drags the body channels with it.

Test: roll out base vs fine-tuned models on flat and bumpy val episodes, with
and without teacher-forcing only the 4 Fz channels in the recurrent feedback.
If teacher-forcing Fz restores stability, the recurrent Fz loop is the cause.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.dataset import load_metadata, load_rollout_split
from nedm.training.model import HMMWVDynamicsModel

FZ_FIELDS = [
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fz_n",
]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HMMWVDynamicsModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    cfg = ckpt["config"]
    model = HMMWVDynamicsModel(
        state_dim=len(meta["state_fields"]),
        action_dim=len(meta["action_fields"]),
        target_dim=len(meta["state_fields"]),
        transformer_cfg=cfg["model"],
        normalization=meta["normalization"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, meta


@torch.no_grad()
def rollout(
    model: HMMWVDynamicsModel,
    episode: dict,
    seq_len: int,
    horizon: int,
    state_index: dict[str, int],
    dt: float,
    tf_channels: list[int] | None = None,
) -> dict:
    """Autoregressive rollout. tf_channels = channel indices to teacher-force
    (replace predicted next-state with ground truth before feeding back)."""
    device = model.state_mean.device
    states = torch.from_numpy(episode["states"]).to(device)
    actions = torch.from_numpy(episode["actions"]).to(device)
    rollout_pose = torch.from_numpy(episode["rollout"]).to(device)
    n = states.shape[0]
    steps = min(horizon, n - seq_len)
    if steps <= 1:
        return {}

    yaw_i = state_index["yaw_rate_radps"]
    vx_i = state_index["vel_body_x_mps"]
    vy_i = state_index["vel_body_y_mps"]
    fz_idx = [state_index[f] for f in FZ_FIELDS]

    hist_s = states[:seq_len].clone()
    hist_a = actions[:seq_len].clone()
    pose = rollout_pose[seq_len - 1].clone()

    pred_states = []
    poses = []
    for k in range(steps):
        sw = hist_s[-seq_len:].unsqueeze(0)
        aw = hist_a[-seq_len:].unsqueeze(0)
        delta = model.predict_delta(sw, aw)[:, -1, :].squeeze(0)
        next_state = hist_s[-1] + delta
        if tf_channels:
            gt_next = states[seq_len + k]
            next_state = next_state.clone()
            next_state[tf_channels] = gt_next[tf_channels]
        # pose integration
        yaw = pose[2] + dt * next_state[yaw_i]
        vxw = torch.cos(yaw) * next_state[vx_i] - torch.sin(yaw) * next_state[vy_i]
        vyw = torch.sin(yaw) * next_state[vx_i] + torch.cos(yaw) * next_state[vy_i]
        pose = torch.stack([pose[0] + dt * vxw, pose[1] + dt * vyw, yaw])
        pred_states.append(next_state)
        poses.append(pose.clone())
        if seq_len + k < actions.shape[0]:
            hist_a = torch.cat([hist_a, actions[seq_len + k].unsqueeze(0)], 0)
        hist_s = torch.cat([hist_s, next_state.unsqueeze(0)], 0)

    pred_states = torch.stack(pred_states, 0)
    poses = torch.stack(poses, 0)
    gt_states = states[seq_len : seq_len + steps]
    gt_pose = rollout_pose[seq_len : seq_len + steps]

    # per-horizon XY rmse
    def wrap(a):
        return torch.atan2(torch.sin(a), torch.cos(a))

    xy_err = (poses[:, :2] - gt_pose[:, :2]).pow(2).sum(-1).sqrt()  # per-step displacement err
    # Fz state magnitude and how far body channels drift
    fz_pred = pred_states[:, fz_idx]
    fz_gt = gt_states[:, fz_idx]
    xy_e = xy_err.cpu().numpy()
    traj_rmse = float(np.sqrt(np.mean(xy_e ** 2)))
    return {
        "steps": int(steps),
        "traj_xy_rmse": traj_rmse,
        "xy_err_per_step": xy_e,
        "vx_pred": pred_states[:, vx_i].cpu().numpy(),
        "vx_gt": gt_states[:, vx_i].cpu().numpy(),
        "fz_pred_mean": fz_pred.mean(-1).cpu().numpy(),
        "fz_gt_mean": fz_gt.mean(-1).cpu().numpy(),
        "fz_pred_absmax": fz_pred.abs().max(-1).values.cpu().numpy(),
    }


def horizon_xy(res: dict, hsteps: list[int]) -> dict[str, float]:
    out = {}
    e = res.get("xy_err_per_step")
    if e is None:
        return out
    for h in hsteps:
        if h <= len(e):
            out[f"xy@{h}"] = float(e[h - 1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path,
                    default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth")
    ap.add_argument("--ft-ckpt", type=Path,
                    default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy10g_flat90g_finetune_fz001_track10_lr3em6_last1/checkpoints/best_val.pt")
    ap.add_argument("--flat-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1")
    ap.add_argument("--bumpy-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "artifacts/analysis/hmmwv_bumpy_finetune_diagnosis/rollout_diag.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base, base_meta = load_model(args.base_ckpt, device)
    ft, ft_meta = load_model(args.ft_ckpt, device)
    seq_len = int(base_meta_blocksize(args.base_ckpt))
    state_index = {f: i for i, f in enumerate(base_meta["state_fields"])}
    fz_idx = [state_index[f] for f in FZ_FIELDS]
    dt = float(base_meta["dt_s"])
    hsteps = [100, 500, 1000, 2000, 3000]

    print(f"seq_len={seq_len} dt={dt} fz_idx={fz_idx}")
    print(f"base target_std Fz: {[round(base.target_std[i].item(),3) for i in fz_idx]}")
    print(f"ft   target_std Fz: {[round(ft.target_std[i].item(),3) for i in fz_idx]}")

    datasets = {"flat": args.flat_dir, "bumpy": args.bumpy_dir}
    summary = {}
    for dname, ddir in datasets.items():
        eps = load_rollout_split(ddir, "val")["episodes"][: args.n_episodes]
        configs = {
            "base_full": (base, None),
            "base_tfFz": (base, fz_idx),
            "ft_full": (ft, None),
            "ft_tfFz": (ft, fz_idx),
        }
        agg = {k: {f"xy@{h}": [] for h in hsteps} for k in configs}
        for k in configs:
            agg[k]["traj_rmse"] = []
        traj_sample = {}
        for ei, ep in enumerate(eps):
            for cname, (mdl, tf) in configs.items():
                res = rollout(mdl, ep, seq_len, args.horizon, state_index, dt, tf)
                if not res:
                    continue
                for k, v in horizon_xy(res, hsteps).items():
                    agg[cname][k].append(v)
                agg[cname]["traj_rmse"].append(res["traj_xy_rmse"])
                if ei == 0:
                    traj_sample[cname] = {
                        "xy_err_per_step": res["xy_err_per_step"].tolist(),
                        "vx_pred": res["vx_pred"].tolist(),
                        "vx_gt": res["vx_gt"].tolist(),
                        "fz_pred_mean": res["fz_pred_mean"].tolist(),
                        "fz_gt_mean": res["fz_gt_mean"].tolist(),
                        "fz_pred_absmax": res["fz_pred_absmax"].tolist(),
                    }
        summary[dname] = {
            "n_episodes": len(eps),
            "mean_xy": {c: {k: (float(np.mean(v)) if v else None) for k, v in agg[c].items()} for c in configs},
            "median_xy": {c: {k: (float(np.median(v)) if v else None) for k, v in agg[c].items()} for c in configs},
        }
        # print table
        cols = [f"xy@{h}" for h in hsteps] + ["traj_rmse"]
        print(f"\n===== {dname} (mean over {len(eps)} eps) =====")
        print(f"{'config':12s} " + " ".join(f"{c:>10s}" for c in cols))
        for c in configs:
            row = summary[dname]["mean_xy"][c]
            print(f"{c:12s} " + " ".join(f"{(row[col] if row.get(col) is not None else float('nan')):>10.2f}" for col in cols))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / f"traj_{dname}.json").write_text(json.dumps(traj_sample))

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def base_meta_blocksize(ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt["config"]["model"]["block_size"])


if __name__ == "__main__":
    raise SystemExit(main())
