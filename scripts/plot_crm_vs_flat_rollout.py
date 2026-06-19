from __future__ import annotations
"""Side-by-side flat-vs-CRM contrast of the flat base open-loop rollout.
Left column = flat (red hugs black), right column = CRM (red diverges); one row per
episode, each cell an XY overlay of GT (black) and base open-loop (red). Via npz pipeline.
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
FLAT_SEQ = REPO / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
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
    return pp, gp, rmse, dist, steps


def panel(ax, pp, gp, rmse, dist, steps):
    ax.plot(gp[:, 0], gp[:, 1], "k-", lw=2, label="ground truth")
    ax.plot(pp[:, 0], pp[:, 1], "r--", lw=1.5, label="base (open-loop)")
    ax.plot(gp[0, 0], gp[0, 1], "go", ms=6)
    frac = rmse / dist if dist > 1e-6 else float("nan")
    ax.set_title(f"{steps} steps · XY RMSE {rmse:.1f} m · {frac:.0%}/dist", fontsize=9)
    ax.set_aspect("equal", "datalim")
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--flat-horizon", type=int, default=2000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta, seq = load_model(BASE_CKPT, dev)
    idx = {f: i for i, f in enumerate(meta["state_fields"])}
    dt = float(meta["dt_s"])

    flat = load_rollout_split(FLAT_SEQ, "val")["episodes"][: args.n]
    crm = load_rollout_split(CRM_SEQ, "val")["episodes"][: args.n]
    flat_r = [rollout(model, e, seq, dt, idx, args.flat_horizon) for e in flat]
    crm_r = [rollout(model, e, seq, dt, idx, 100000) for e in crm]

    n = args.n
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.0 * n))
    for i in range(n):
        panel(axes[i, 0], *flat_r[i])
        panel(axes[i, 1], *crm_r[i])
        axes[i, 0].set_ylabel(f"ep {i}", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")
    fm = np.mean([r[2] / max(r[3], 1e-6) for r in flat_r])
    cm = np.mean([r[2] / max(r[3], 1e-6) for r in crm_r])
    axes[0, 0].annotate(f"FLAT  (mean {fm:.0%} err/dist)", xy=(0.5, 1.25), xycoords="axes fraction",
                        ha="center", fontsize=14, fontweight="bold", color="darkgreen")
    axes[0, 1].annotate(f"CRM  (mean {cm:.0%} err/dist)", xy=(0.5, 1.25), xycoords="axes fraction",
                        ha="center", fontsize=14, fontweight="bold", color="darkred")
    fig.suptitle("Flat base open-loop rollout: FLAT (red tracks black) vs CRM (red diverges)",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    f = OUT / "crm_vs_flat_rollout_overlay.png"
    fig.savefig(f, dpi=120, bbox_inches="tight")
    print(f"wrote {f}")
    print(f"FLAT mean err/dist {fm:.1%} | CRM mean err/dist {cm:.1%}")


if __name__ == "__main__":
    main()
