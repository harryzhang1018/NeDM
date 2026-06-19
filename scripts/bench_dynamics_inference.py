from __future__ import annotations
"""Standalone throughput benchmark for the HMMWV neural dynamics model.

Isolates the dynamics-model forward pass from the RL loop so we can see how
inference throughput scales with the env-batch dimension (the "(15,1)->(15,n)"
parallelism the user expects to be ~free) and how much of the cost comes from
running the full block_size context every substep.

Run:
    /home/harry/anaconda3/envs/nedm/bin/python scripts/bench_dynamics_inference.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402

DEFAULT_CKPT = (
    "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/"
    "checkpoints/best_val.pth"
)


def load_model(path: str, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    md, cfg = ck["metadata"], ck["config"]
    model = HMMWVDynamicsModel(
        state_dim=len(md["state_fields"]),
        action_dim=len(md["action_fields"]),
        target_dim=len(md["state_fields"]),
        transformer_cfg=cfg["model"],
        normalization=md["normalization"],
    )
    sd = ck["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, md, cfg, len(md["state_fields"]), len(md["action_fields"])


@torch.no_grad()
def bench(model, batch, seq, state_dim, action_dim, device, iters, dtype):
    states = torch.randn(batch, seq, state_dim, device=device)
    actions = torch.randn(batch, seq, action_dim, device=device)

    def run():
        return model.predict_next_delta(states, actions)

    # warmup
    for _ in range(10):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            run()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt  # seconds per forward pass


def fmt(x):
    return f"{x/1e3:7.1f}K" if x >= 1e3 else f"{x:8.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--action-repeat", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 64, 256, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 11, 1])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tf32", action="store_true", help="enable TF32 matmul (matmul_precision=high)")
    ap.add_argument("--bf16", action="store_true", help="autocast to bfloat16")
    args = ap.parse_args()

    device = torch.device("cuda")
    if args.tf32:
        torch.set_float32_matmul_precision("high")
    dtype = torch.bfloat16 if args.bf16 else None

    model, md, cfg, sd, ad = load_model(args.ckpt, device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"model    : n_layer={cfg['model']['n_layer']} n_embd={cfg['model']['n_embd']} "
          f"block_size={cfg['model']['block_size']} params={nparams/1e6:.2f}M")
    print(f"state_dim={sd} action_dim={ad}  tf32={args.tf32} bf16={args.bf16} "
          f"compile={args.compile} action_repeat={args.action_repeat}\n")

    if args.compile:
        model = torch.compile(model)

    block_size = int(cfg["model"]["block_size"])
    args.seqs = [s for s in args.seqs if s <= block_size]

    for seq in args.seqs:
        print(f"=== seq_len (context) = {seq} ===")
        print(f"{'batch':>8} | {'ms/fwd':>9} | {'fwd/s':>9} | "
              f"{'envstep/s':>10} | {'policy fps*':>11} | {'us/env':>8}")
        print("-" * 72)
        for batch in args.batches:
            try:
                dt = bench(model, batch, seq, sd, ad, device, args.iters, dtype)
            except RuntimeError as e:
                print(f"{batch:>8} | OOM/err: {str(e)[:40]}")
                continue
            fwd_per_s = 1.0 / dt
            envstep_per_s = fwd_per_s * batch
            policy_fps = envstep_per_s / args.action_repeat
            us_per_env = dt / batch * 1e6
            print(f"{batch:>8} | {dt*1e3:9.3f} | {fmt(fwd_per_s)} | "
                  f"{fmt(envstep_per_s)} | {fmt(policy_fps)} | {us_per_env:8.3f}")
        print()
    print("* policy fps = envstep/s / action_repeat  (env.step() calls per second, "
          "the unit rsl_rl reports as FPS, ignoring PPO update + python overhead)")


if __name__ == "__main__":
    main()
