from __future__ import annotations
"""Plot flat-base open-loop rollout on CRM episodes (via the npz pipeline):
  fig1: GT vs predicted XY trajectory overlays (grid of episodes)
  fig2: GT vs predicted key channels (vx, mean omega, mean Fz, pitch) for one episode
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

BASE_CKPT = REPO / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth"
CRM_SEQ = REPO / "artifacts/training_datasets/hmmwv_crm_100_normal_force_omega_seq_v1"
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
def rollout(model, ep, seq, dt, idx, horizon):
    dev = model.state_mean.device
    S = torch.from_numpy(ep["states"]).to(dev)
    A = torch.from_numpy(ep["actions"]).to(dev)
    P = torch.from_numpy(ep["rollout"]).to(dev)
    steps = min(horizon, S.shape[0] - seq)
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs, ha = S[:seq].clone(), A[:seq].clone()
    p = P[seq - 1].clone()
    path, preds = [p[:2].cpu().numpy().copy()], []
    for k in range(steps):
        d = model.predict_delta(hs[-seq:].unsqueeze(0), ha[-seq:].unsqueeze(0))[:, -1, :].squeeze(0)
        ns = hs[-1] + d
        yaw = p[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        p = torch.stack([p[0] + dt * vxw, p[1] + dt * vyw, yaw])
        path.append(p[:2].cpu().numpy().copy())
        preds.append(ns.cpu().numpy())
        if seq + k < A.shape[0]:
            ha = torch.cat([ha, A[seq + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    pred_path = np.array(path)
    gt_path = P[seq - 1:seq + steps, :2].cpu().numpy()
    pred_s = np.array(preds)                       # (steps,15) open-loop
    gt_s = S[seq:seq + steps].cpu().numpy()
    return pred_path, gt_path, pred_s, gt_s, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=100000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, seq = load_model(BASE_CKPT, dev)
    sf = meta["state_fields"]
    idx = {f: i for i, f in enumerate(sf)}
    dt = float(meta["dt_s"])
    eps = load_rollout_split(CRM_SEQ, "val")["episodes"][: args.n]
    results = [rollout(model, ep, seq, dt, idx, args.horizon) for ep in eps]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fig1: XY trajectory overlays ----
    ncol = 3
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.array(axes).reshape(-1)
    for i, (pp, gp, _, _, steps) in enumerate(results):
        ax = axes[i]
        ax.plot(gp[:, 0], gp[:, 1], "k-", lw=2, label="ground truth")
        ax.plot(pp[:, 0], pp[:, 1], "r--", lw=1.5, label="base (open-loop)")
        ax.plot(gp[0, 0], gp[0, 1], "go", ms=6)
        rmse = np.sqrt(((pp[1:] - gp[1:]) ** 2).sum(1).mean())
        ax.set_title(f"val ep {i}  ({steps} steps)\nXY RMSE {rmse:.1f} m", fontsize=10)
        ax.set_aspect("equal", "datalim")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Flat-base open-loop rollout on CRM (deformable) — trajectory drift", fontsize=13)
    fig.tight_layout()
    f1 = OUT / "crm_base_rollout_xy.png"
    fig.savefig(f1, dpi=130)
    print(f"wrote {f1}")

    # ---- fig2: key channels for the most-mobile of the chosen episodes ----
    mob = int(np.argmax([np.median(np.abs(gs[:, idx["vel_body_x_mps"]])) for _, _, _, gs, _ in results]))
    pp, gp, ps, gs, steps = results[mob]
    t = np.arange(steps) * dt
    om = [idx[f] for f in sf if "spindle_omega" in f]
    fz = [idx[f] for f in sf if "force_wheel_fz" in f]
    panels = [
        ("vel_body_x_mps (forward speed)", gs[:, idx["vel_body_x_mps"]], ps[:, idx["vel_body_x_mps"]], "m/s"),
        ("mean wheel omega", gs[:, om].mean(1), ps[:, om].mean(1), "rad/s"),
        ("mean tire Fz", gs[:, fz].mean(1), ps[:, fz].mean(1), "N"),
        ("pitch", gs[:, idx["pitch_rad"]], ps[:, idx["pitch_rad"]], "rad"),
    ]
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (title, g, p, unit) in zip(axes2.reshape(-1), panels):
        ax.plot(t, g, "k-", lw=1.8, label="ground truth")
        ax.plot(t, p, "r--", lw=1.4, label="base (open-loop)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time (s)"); ax.set_ylabel(unit)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig2.suptitle(f"Flat-base open-loop channels on CRM (val ep {mob}) — vx/omega/Fz break, attitude ~ok",
                  fontsize=13)
    fig2.tight_layout()
    f2 = OUT / "crm_base_rollout_channels.png"
    fig2.savefig(f2, dpi=130)
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
