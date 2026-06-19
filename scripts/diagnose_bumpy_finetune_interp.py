from __future__ import annotations

"""Weight-space interpolation (WiSE-FT) sweep + per-channel rollout divergence.

theta(alpha) = (1-alpha)*theta_base + alpha*theta_ft, evaluated by full-episode
open-loop rollout on flat and bumpy val episodes. Also reports which state
channel's rollout error grows for base vs ft to localize the compounding bias.
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

from nedm.training.dataset import load_rollout_split
from nedm.training.model import HMMWVDynamicsModel


def load_ckpt(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)


def build_model(meta, cfg, device):
    m = HMMWVDynamicsModel(
        state_dim=len(meta["state_fields"]),
        action_dim=len(meta["action_fields"]),
        target_dim=len(meta["state_fields"]),
        transformer_cfg=cfg["model"],
        normalization=meta["normalization"],
    )
    return m.to(device).eval()


@torch.no_grad()
def rollout_channels(model, episode, seq_len, horizon, state_index, dt):
    device = model.state_mean.device
    states = torch.from_numpy(episode["states"]).to(device)
    actions = torch.from_numpy(episode["actions"]).to(device)
    pose0 = torch.from_numpy(episode["rollout"]).to(device)
    n = states.shape[0]
    steps = min(horizon, n - seq_len)
    if steps <= 1:
        return None
    yaw_i, vx_i, vy_i = state_index["yaw_rate_radps"], state_index["vel_body_x_mps"], state_index["vel_body_y_mps"]
    hist_s = states[:seq_len].clone()
    hist_a = actions[:seq_len].clone()
    pose = pose0[seq_len - 1].clone()
    pred_states, poses = [], []
    for k in range(steps):
        sw = hist_s[-seq_len:].unsqueeze(0)
        aw = hist_a[-seq_len:].unsqueeze(0)
        delta = model.predict_delta(sw, aw)[:, -1, :].squeeze(0)
        ns = hist_s[-1] + delta
        yaw = pose[2] + dt * ns[yaw_i]
        vxw = torch.cos(yaw) * ns[vx_i] - torch.sin(yaw) * ns[vy_i]
        vyw = torch.sin(yaw) * ns[vx_i] + torch.cos(yaw) * ns[vy_i]
        pose = torch.stack([pose[0] + dt * vxw, pose[1] + dt * vyw, yaw])
        pred_states.append(ns)
        poses.append(pose.clone())
        if seq_len + k < actions.shape[0]:
            hist_a = torch.cat([hist_a, actions[seq_len + k].unsqueeze(0)], 0)
        hist_s = torch.cat([hist_s, ns.unsqueeze(0)], 0)
    pred_states = torch.stack(pred_states, 0)
    poses = torch.stack(poses, 0)
    gt_states = states[seq_len:seq_len + steps]
    gt_pose = pose0[seq_len:seq_len + steps]
    xy = (poses[:, :2] - gt_pose[:, :2]).pow(2).sum(-1).sqrt()
    traj = float(np.sqrt(np.mean(xy.cpu().numpy() ** 2)))
    chan_se = (pred_states - gt_states).pow(2).mean(0)  # per-channel mean sq err over steps
    return traj, chan_se.cpu().numpy()


def interp_state(sd_base, sd_ft, alpha):
    out = {}
    for k in sd_base:
        a, b = sd_base[k], sd_ft[k]
        if a.dtype.is_floating_point:
            out[k] = (1 - alpha) * a + alpha * b
        else:
            out[k] = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=Path, default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth")
    ap.add_argument("--ft-ckpt", type=Path, default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy10g_flat90g_finetune_fz001_track10_lr3em6_last1/checkpoints/best_val.pt")
    ap.add_argument("--flat-dir", type=Path, default=REPO_ROOT / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1")
    ap.add_argument("--bumpy-dir", type=Path, default=REPO_ROOT / "artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1")
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6000)
    ap.add_argument("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts/analysis/hmmwv_bumpy_finetune_diagnosis/interp.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base_ck = load_ckpt(args.base_ckpt)
    ft_ck = load_ckpt(args.ft_ckpt)
    meta = base_ck["metadata"]
    cfg = base_ck["config"]
    seq_len = int(cfg["model"]["block_size"])
    dt = float(meta["dt_s"])
    state_index = {f: i for i, f in enumerate(meta["state_fields"])}
    sd_base = base_ck["model_state_dict"]
    sd_ft = ft_ck["model_state_dict"]
    alphas = [float(x) for x in args.alphas.split(",")]
    model = build_model(meta, cfg, device)

    flat_eps = load_rollout_split(args.flat_dir, "val")["episodes"][: args.n_episodes]
    bumpy_eps = load_rollout_split(args.bumpy_dir, "val")["episodes"][: args.n_episodes]

    results = {"alphas": alphas, "flat": {}, "bumpy": {}}
    chan_store = {}
    for alpha in alphas:
        model.load_state_dict(interp_state(sd_base, sd_ft, alpha))
        model.eval()
        for dname, eps in [("flat", flat_eps), ("bumpy", bumpy_eps)]:
            trajs = []
            chan_acc = np.zeros(len(meta["state_fields"]))
            cnt = 0
            for ep in eps:
                r = rollout_channels(model, ep, seq_len, args.horizon, state_index, dt)
                if r is None:
                    continue
                trajs.append(r[0])
                chan_acc += r[1]
                cnt += 1
            results[dname][f"{alpha:.2f}"] = float(np.mean(trajs)) if trajs else None
            chan_store[(dname, alpha)] = np.sqrt(chan_acc / max(cnt, 1))

    print(f"{'alpha':>6s} | {'flat traj':>10s} | {'bumpy traj':>10s}")
    for alpha in alphas:
        print(f"{alpha:6.2f} | {results['flat'][f'{alpha:.2f}']:10.2f} | {results['bumpy'][f'{alpha:.2f}']:10.2f}")

    # per-channel divergence base(alpha=0) vs ft(alpha=1) on flat
    print("\nper-channel rollout RMSE (flat) base(a=0) vs ft(a=1):")
    fields = meta["state_fields"]
    c0 = chan_store[("flat", alphas[0])]
    c1 = chan_store[("flat", alphas[-1])]
    for i, f in enumerate(fields):
        print(f"  {f:32s} base={c0[i]:.4f}  ft={c1[i]:.4f}  x{(c1[i]/max(c0[i],1e-9)):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
