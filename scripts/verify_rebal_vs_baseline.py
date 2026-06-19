from __future__ import annotations
"""Robust full-episode open-loop verification: compare checkpoints on flat and CRM
val episodes via the standard npz rollout pipeline (load_rollout_split).

Reports per domain:
  - aggregate err/dist = sqrt(sum pos_sq_err / sum steps) / mean_episode_dist  (robust;
    no per-episode division, so stuck/short episodes do not blow it up)
  - mean-of-ratios err/dist (matches the original CRM analysis numbers)
  - raw XY RMSE (m) and per-channel rollout RMSE for vx / mean omega / mean Fz

Usage:
  verify_rebal_vs_baseline.py --ckpts base_last=PATH new_best=PATH ... \
     --flat-n 16 --crm-n 22 --flat-horizon 100000 --crm-horizon 100000
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

FLAT_SEQ = REPO / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
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
    P = torch.from_numpy(ep["rollout"]).to(dev)
    steps = min(horizon, S.shape[0] - seq)
    if steps <= 1:
        return None
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs, ha = S[:seq].clone(), A[:seq].clone()
    p = P[seq - 1].clone()
    pos_sq = []
    preds = []
    for k in range(steps):
        d = model.predict_delta(hs[-seq:].unsqueeze(0), ha[-seq:].unsqueeze(0))[:, -1, :].squeeze(0)
        ns = hs[-1] + d
        yaw = p[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        p = torch.stack([p[0] + dt * vxw, p[1] + dt * vyw, yaw])
        pos_sq.append(((p[:2] - P[seq + k][:2]) ** 2).sum())
        preds.append(ns)
        if seq + k < A.shape[0]:
            ha = torch.cat([ha, A[seq + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    pos_sq = torch.stack(pos_sq)
    xy_rmse = float(pos_sq.mean().sqrt().cpu())
    gt_xy = P[seq - 1:seq + steps, :2].cpu().numpy()
    dist = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())
    chan_rmse = (torch.stack(preds) - S[seq:seq + steps]).pow(2).mean(0).sqrt().cpu().numpy()
    return float(pos_sq.sum().cpu()), steps, xy_rmse, dist, chan_rmse


def eval_domain(model, meta, seq, eps, horizon):
    idx = {f: i for i, f in enumerate(meta["state_fields"])}
    dt = float(meta["dt_s"])
    tot_pos_sq = 0.0; tot_steps = 0; dists = []; ratios = []; chans = []
    for ep in eps:
        r = rollout(model, ep, seq, dt, idx, horizon)
        if r is None:
            continue
        pos_sq_sum, steps, xy_rmse, dist, chan = r
        tot_pos_sq += pos_sq_sum; tot_steps += steps
        dists.append(dist); chans.append(chan)
        if dist > 1e-6:
            ratios.append(xy_rmse / dist)
    agg_xy_rmse = float(np.sqrt(tot_pos_sq / max(tot_steps, 1)))
    mean_dist = float(np.mean(dists)) if dists else float("nan")
    agg_errdist = agg_xy_rmse / mean_dist if mean_dist > 1e-6 else float("nan")
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    chan = np.mean(np.stack(chans), 0)
    return {"agg_errdist": agg_errdist, "mean_ratio_errdist": mean_ratio,
            "agg_xy_rmse": agg_xy_rmse, "mean_dist": mean_dist, "n": len(dists), "chan": chan, "fields": meta["state_fields"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--flat-n", type=int, default=16)
    ap.add_argument("--crm-n", type=int, default=22)
    ap.add_argument("--flat-horizon", type=int, default=100000)
    ap.add_argument("--crm-horizon", type=int, default=100000)
    ap.add_argument("--domains", nargs="+", default=["flat", "crm"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    domain_eps = {}
    if "flat" in args.domains:
        domain_eps["flat"] = (load_rollout_split(FLAT_SEQ, "val")["episodes"][: args.flat_n], args.flat_horizon)
    if "crm" in args.domains:
        domain_eps["crm"] = (load_rollout_split(CRM_SEQ, "val")["episodes"][: args.crm_n], args.crm_horizon)

    results = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        model, meta, seq = load_model(Path(path), dev)
        results[name] = {}
        for dom, (eps, hz) in domain_eps.items():
            results[name][dom] = eval_domain(model, meta, seq, eps, hz)
        print(f"[done] {name}")

    for dom in domain_eps:
        print(f"\n===== {dom.upper()} val (full-episode open-loop) =====")
        print(f"{'ckpt':18s} {'n':>3s} {'agg_err/dist':>12s} {'mean-ratio':>11s} {'xy_rmse_m':>10s} {'mean_dist_m':>11s}")
        for name in results:
            r = results[name][dom]
            print(f"{name:18s} {r['n']:>3d} {r['agg_errdist']:>12.1%} {r['mean_ratio_errdist']:>11.1%} {r['agg_xy_rmse']:>10.2f} {r['mean_dist']:>11.1f}")
        # per-channel summary for vx / mean omega / mean Fz
        fields = next(iter(results.values()))[dom]["fields"]
        vxi = fields.index("vel_body_x_mps")
        omi = [i for i, f in enumerate(fields) if "spindle_omega" in f]
        fzi = [i for i, f in enumerate(fields) if "force_wheel_fz" in f]
        print(f"  per-channel rollout RMSE:  {'ckpt':16s} {'vx(m/s)':>8s} {'omega(rad/s)':>12s} {'Fz(N)':>9s}")
        for name in results:
            c = results[name][dom]["chan"]
            print(f"  {'':27s}{name:16s} {c[vxi]:>8.3f} {np.mean([c[i] for i in omi]):>12.3f} {np.mean([c[i] for i in fzi]):>9.0f}")


if __name__ == "__main__":
    main()
