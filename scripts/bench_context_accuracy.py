from __future__ import annotations
"""Does truncating the dynamics-model context window hurt rollout accuracy?

The RL env feeds the dynamics model the full block_size=128 history every
substep but only reads the last token. This script runs the SAME autoregressive
rollout the env does (predict_next_delta + pose integration) at several context
truncations K and reports multi-step pose RMSE vs ground truth, so we can pick
the smallest K that preserves accuracy before cutting compute.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.training.dataset import load_metadata, load_rollout_split  # noqa: E402
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402

DEFAULT_CKPT = (
    "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/"
    "checkpoints/best_val.pth"
)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    md, cfg = ck["metadata"], ck["config"]
    model = HMMWVDynamicsModel(
        len(md["state_fields"]), len(md["action_fields"]),
        len(md["state_fields"]), cfg["model"], md["normalization"])
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model_state_dict"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, md, cfg, ck["config"]["processed_dataset_dir"]


@torch.no_grad()
def rollout_pose_rmse(model, ep, ctx, full_block, horizon, idx, dt, device):
    """Autoregressive rollout using exactly `ctx` context tokens at each step.

    Mirrors HMMWVNeuralTrackingEnv._nn_substep / _integrate_pose. `full_block`
    primes from the full block_size warmup (matching the env's reset), then each
    step only the last `ctx` tokens are fed to the model.
    """
    s = torch.from_numpy(ep["states"]).to(device)
    a = torch.from_numpy(ep["actions"]).to(device)
    pose0 = torch.from_numpy(ep["rollout"]).to(device)
    steps = min(horizon, s.shape[0] - full_block)
    if steps <= 1:
        return None
    yi, xi, yyi = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hs = s[:full_block].clone()
    ha = a[:full_block].clone()
    pose = pose0[full_block - 1].clone()
    errs = []
    for k in range(steps):
        win_s = hs[-ctx:].unsqueeze(0)
        win_a = ha[-ctx:].unsqueeze(0)
        d = model.predict_next_delta(win_s, win_a).squeeze(0)
        ns = hs[-1] + d
        yaw = pose[2] + dt * ns[yi]
        vxw = torch.cos(yaw) * ns[xi] - torch.sin(yaw) * ns[yyi]
        vyw = torch.sin(yaw) * ns[xi] + torch.cos(yaw) * ns[yyi]
        pose = torch.stack([pose[0] + dt * vxw, pose[1] + dt * vyw, yaw])
        errs.append(((pose[:2] - pose0[full_block + k][:2]) ** 2).sum().sqrt())
        if full_block + k < a.shape[0]:
            ha = torch.cat([ha, a[full_block + k].unsqueeze(0)], 0)
        hs = torch.cat([hs, ns.unsqueeze(0)], 0)
    return float(torch.stack(errs).pow(2).mean().sqrt().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[128, 64, 32, 16, 11, 8, 4, 2, 1])
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, md, cfg, ds_dir = load_model(args.ckpt, device)
    block = int(cfg["model"]["block_size"])
    dt = float(md["dt_s"])
    idx = {f: i for i, f in enumerate(md["state_fields"])}

    data = load_rollout_split(Path(ds_dir), args.split)
    eps = data["episodes"][: args.n_episodes]
    print(f"block_size={block} dt={dt}  split={args.split}  episodes={len(eps)}  "
          f"horizon={args.horizon}\n")
    print(f"{'context K':>9} | {'pose RMSE (m)':>13} | {'vs full-128':>11}")
    print("-" * 42)

    base = None
    for ctx in args.contexts:
        if ctx > block:
            continue
        vals = []
        for ep in eps:
            r = rollout_pose_rmse(model, ep, ctx, block, args.horizon, idx, dt, device)
            if r is not None:
                vals.append(r)
        mean = float(np.mean(vals))
        if base is None:
            base = mean
        ratio = mean / base
        print(f"{ctx:>9} | {mean:>13.4f} | {ratio:>10.2f}x")


if __name__ == "__main__":
    main()
