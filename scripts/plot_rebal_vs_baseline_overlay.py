from __future__ import annotations
"""Open-loop XY overlays: ground truth (black) vs baseline mix25 last.pt (red) vs
new rebalanced+rollout-selected best_val (blue), on flat and CRM val episodes.

Produces two contact sheets:
  rebal_overlay_crm.png  and  rebal_overlay_flat.png
under artifacts/analysis/hmmwv_crm_15d_distribution/.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from nedm.training.dataset import load_rollout_split
from nedm.training.model import HMMWVDynamicsModel

FLAT_SEQ = REPO / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
CRM_SEQ = REPO / "artifacts/training_datasets/hmmwv_crm_100_normal_force_omega_seq_v1"
BASE_LAST = REPO / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_scratch/checkpoints/last.pt"
NEW_BEST = REPO / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_rebal_rollout/checkpoints/last.pt"
OUT = REPO / "artifacts/analysis/hmmwv_crm_15d_distribution"


def load_model(p, device):
    c = torch.load(p, map_location="cpu", weights_only=False)
    m, cfg = c["metadata"], c["config"]
    model = HMMWVDynamicsModel(len(m["state_fields"]), len(m["action_fields"]),
                               len(m["state_fields"]), cfg["model"], m["normalization"])
    model.load_state_dict(c["model_state_dict"])
    model.to(device).eval()
    return model, m, int(cfg["model"]["block_size"])


@torch.no_grad()
def rollout_xy(model, ep, seq, dt, idx, horizon):
    dev = model.state_mean.device
    S = torch.from_numpy(ep["states"]).to(dev)
    A = torch.from_numpy(ep["actions"]).to(dev)
    P = torch.from_numpy(ep["rollout"]).to(dev)
    steps = min(horizon, S.shape[0] - seq)
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs, ha = S[:seq].clone(), A[:seq].clone()
    p = P[seq - 1].clone()
    path = [p[:2].cpu().numpy().copy()]
    for k in range(steps):
        d = model.predict_delta(hs[-seq:].unsqueeze(0), ha[-seq:].unsqueeze(0))[:, -1, :].squeeze(0)
        ns = hs[-1] + d
        yaw = p[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        p = torch.stack([p[0] + dt * vxw, p[1] + dt * vyw, yaw])
        path.append(p[:2].cpu().numpy().copy())
        if seq + k < A.shape[0]:
            ha = torch.cat([ha, A[seq + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    pp = np.array(path)
    gp = P[seq - 1:seq + steps, :2].cpu().numpy()
    rmse = float(np.sqrt(((pp[1:] - gp[1:]) ** 2).sum(1).mean()))
    dist = float(np.linalg.norm(np.diff(gp, axis=0), axis=1).sum())
    return pp, gp, rmse, dist


def make_sheet(domain, seq_dir, n, horizon, base, new, ncol=3):
    bm, meta, bs = base
    nm, _, ns_ = new
    idx = {f: i for i, f in enumerate(meta["state_fields"])}
    dt = float(meta["dt_s"])
    eps = load_rollout_split(seq_dir, "val")["episodes"][:n]
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, ep in enumerate(eps):
        ax = axes[i]
        bpp, gp, brm, dist = rollout_xy(bm, ep, bs, dt, idx, horizon)
        npp, _, nrm, _ = rollout_xy(nm, ep, ns_, dt, idx, horizon)
        ax.plot(gp[:, 0], gp[:, 1], "k-", lw=2.2, label="ground truth")
        ax.plot(bpp[:, 0], bpp[:, 1], "r--", lw=1.4, label="baseline last")
        ax.plot(npp[:, 0], npp[:, 1], "b-", lw=1.4, label="new ep80 (rebal+rollout)")
        ax.plot(gp[0, 0], gp[0, 1], "go", ms=6)
        bf = brm / dist if dist > 1e-6 else float("nan")
        nf = nrm / dist if dist > 1e-6 else float("nan")
        ax.set_title(f"{domain} ep {i}  ({dist:.0f} m)\nbase {bf:.0%}/dist  →  new {nf:.0%}/dist", fontsize=9)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{domain.upper()} open-loop: GT (black) vs baseline mix25 (red) vs rebalanced+rollout (blue)", fontsize=12)
    fig.tight_layout()
    f = OUT / f"rebal_overlay_{domain}.png"
    fig.savefig(f, dpi=120)
    print(f"wrote {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--flat-horizon", type=int, default=100000)
    ap.add_argument("--crm-horizon", type=int, default=100000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_model(BASE_LAST, dev)
    new = load_model(NEW_BEST, dev)
    make_sheet("crm", CRM_SEQ, args.n, args.crm_horizon, base, new)
    make_sheet("flat", FLAT_SEQ, args.n, args.flat_horizon, base, new)


if __name__ == "__main__":
    main()
