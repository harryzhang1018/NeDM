from __future__ import annotations
"""Open-loop rollout of the flat-trained base on CRM episodes, via the standard
npz pipeline (load_rollout_split) — same data path the rigid rollout eval uses.

Reports per-episode XY-trajectory RMSE, distance traveled and error-per-distance
(the honest cross-terrain comparison, since CRM episodes are short and slow), and
per-channel open-loop state RMSE to localize where the base breaks on soft soil.

Build the CRM seq cache first:
  build_hmmwv_training_dataset.py --dataset-root artifacts/datasets/hmmwv_crm_100 \
    --output-dir artifacts/training_datasets/hmmwv_crm_100_normal_force_omega_seq_v1 \
    --state-field-preset tire_normal_force_omega
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nedm.training.dataset import load_rollout_split
from nedm.training.model import HMMWVDynamicsModel

BASE_CKPT = REPO / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth"
CRM_SEQ = REPO / "artifacts/training_datasets/hmmwv_crm_100_normal_force_omega_seq_v1"


def load_model(p, device):
    c = torch.load(p, map_location="cpu", weights_only=False)
    m, cfg = c["metadata"], c["config"]
    model = HMMWVDynamicsModel(len(m["state_fields"]), len(m["action_fields"]),
                               len(m["state_fields"]), cfg["model"], m["normalization"])
    model.load_state_dict(c["model_state_dict"])
    model.to(device).eval()
    return model, m, int(cfg["model"]["block_size"])


@torch.no_grad()
def rollout(model, ep, seq, dt, idx, horizon):
    dev = model.state_mean.device
    S = torch.from_numpy(ep["states"]).to(dev)
    A = torch.from_numpy(ep["actions"]).to(dev)
    P = torch.from_numpy(ep["rollout"]).to(dev)             # (T, 3) = x, y, yaw
    steps = min(horizon, S.shape[0] - seq)
    if steps <= 1:
        return None
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs, ha = S[:seq].clone(), A[:seq].clone()
    p = P[seq - 1].clone()
    xy_err, preds = [], []
    for k in range(steps):
        d = model.predict_delta(hs[-seq:].unsqueeze(0), ha[-seq:].unsqueeze(0))[:, -1, :].squeeze(0)
        ns = hs[-1] + d
        yaw = p[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        p = torch.stack([p[0] + dt * vxw, p[1] + dt * vyw, yaw])
        xy_err.append(((p[:2] - P[seq + k][:2]) ** 2).sum().sqrt())
        preds.append(ns)
        if seq + k < A.shape[0]:
            ha = torch.cat([ha, A[seq + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    xy_rmse = float(torch.stack(xy_err).pow(2).mean().sqrt().cpu())
    gt = S[seq:seq + steps]
    chan_rmse = (torch.stack(preds) - gt).pow(2).mean(0).sqrt().cpu().numpy()
    # ground-truth distance traveled over the rolled-out window
    gt_xy = P[seq - 1:seq + steps, :2].cpu().numpy()
    dist = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())
    vx_med = float(gt[:, xi].abs().median().cpu())
    return xy_rmse, chan_rmse, steps, dist, vx_med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=BASE_CKPT)
    ap.add_argument("--seq-dir", type=Path, default=CRM_SEQ)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=100000, help="caps at episode length (full episode by default)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, seq = load_model(args.ckpt, dev)
    sf = meta["state_fields"]
    idx = {f: i for i, f in enumerate(sf)}
    dt = float(meta["dt_s"])

    eps = load_rollout_split(args.seq_dir, args.split)["episodes"][: args.n]
    print(f"checkpoint : {args.ckpt.name}  (flat-trained base, 15D)")
    print(f"data       : {args.seq_dir.name} [{args.split}], {len(eps)} episodes, full-episode open-loop\n")

    print(f"{'#':>2s} {'steps':>6s} {'vx_med':>7s} {'dist_m':>8s} {'XY_RMSE_m':>10s} {'err/dist':>9s}")
    xy_all, chan_all, dist_all = [], [], []
    for i, ep in enumerate(eps):
        res = rollout(model, ep, seq, dt, idx, args.horizon)
        if res is None:
            continue
        xy, chan, steps, dist, vx_med = res
        xy_all.append(xy); chan_all.append(chan); dist_all.append(dist)
        frac = xy / dist if dist > 1e-6 else float("nan")
        print(f"{i:2d} {steps:6d} {vx_med:7.2f} {dist:8.1f} {xy:10.2f} {frac:9.1%}")

    xy_all = np.array(xy_all); chan_all = np.array(chan_all); dist_all = np.array(dist_all)
    print(f"\nXY RMSE : mean {xy_all.mean():.2f} m | median {np.median(xy_all):.2f} m")
    print(f"err/dist: mean {(xy_all/np.maximum(dist_all,1e-6)).mean():.1%}  "
          f"(flat base on flat ≈ 4%/dist, on bumpy ≈ 2%/dist)")

    print("\n== per-channel open-loop rollout RMSE (mean over episodes) ==")
    for i, f in enumerate(sf):
        print(f"  {f:32s} {chan_all[:, i].mean():12.4f}")


if __name__ == "__main__":
    main()
