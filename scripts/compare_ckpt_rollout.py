from __future__ import annotations
"""Lean full-horizon rollout comparison of a list of checkpoints on flat+bumpy."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.training.dataset import load_rollout_split
from nedm.training.model import HMMWVDynamicsModel


def load_model(p, device):
    c = torch.load(p, map_location="cpu", weights_only=False)
    m = c["metadata"]; cfg = c["config"]
    model = HMMWVDynamicsModel(len(m["state_fields"]), len(m["action_fields"]), len(m["state_fields"]), cfg["model"], m["normalization"])
    model.load_state_dict(c["model_state_dict"]); model.to(device).eval()
    return model, m, int(cfg["model"]["block_size"])


@torch.no_grad()
def traj_rmse(model, ep, seq, horizon, idx, dt):
    dev = model.state_mean.device
    s = torch.from_numpy(ep["states"]).to(dev); a = torch.from_numpy(ep["actions"]).to(dev)
    pose0 = torch.from_numpy(ep["rollout"]).to(dev)
    steps = min(horizon, s.shape[0] - seq)
    if steps <= 1: return None
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs = s[:seq].clone(); ha = a[:seq].clone(); pose = pose0[seq - 1].clone(); errs = []
    for k in range(steps):
        d = model.predict_delta(hs[-seq:].unsqueeze(0), ha[-seq:].unsqueeze(0))[:, -1, :].squeeze(0)
        ns = hs[-1] + d
        yaw = pose[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        pose = torch.stack([pose[0] + dt * vxw, pose[1] + dt * vyw, yaw])
        errs.append(((pose[:2] - pose0[seq + k][:2]) ** 2).sum().sqrt())
        if seq + k < a.shape[0]: ha = torch.cat([ha, a[seq + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    return float(torch.stack(errs).pow(2).mean().sqrt().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--flat-dir", default=str(REPO_ROOT / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"))
    ap.add_argument("--bumpy-dir", default=str(REPO_ROOT / "artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=6000)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = load_rollout_split(Path(args.flat_dir), "val")["episodes"][: args.n]
    bumpy = load_rollout_split(Path(args.bumpy_dir), "val")["episodes"][: args.n]
    print(f"{'ckpt':28s} {'flat':>8s} {'bumpy':>8s}")
    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        model, meta, seq = load_model(path, dev)
        idx = {f: i for i, f in enumerate(meta["state_fields"])}; dt = float(meta["dt_s"])
        fr = np.mean([traj_rmse(model, e, seq, args.horizon, idx, dt) for e in flat])
        br = np.mean([traj_rmse(model, e, seq, args.horizon, idx, dt) for e in bumpy])
        print(f"{name:28s} {fr:8.2f} {br:8.2f}")


if __name__ == "__main__":
    main()
