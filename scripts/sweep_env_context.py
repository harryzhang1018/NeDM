from __future__ import annotations
"""2D sweep: dynamics context length x number of envs -> RL collection throughput.

Measures the actual env.step() rollout loop (policy forward + action_repeat NN
substeps + reward/obs) the PPO collection phase runs, for each (n_env, context)
combo, plus peak GPU memory. Collection is ~97% of RL iteration time, so its
steps/s is the right proxy for the training-FPS sweet spot.

Run:
    /home/harry/anaconda3/envs/nedm/bin/python scripts/sweep_env_context.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv, default_env_cfg  # noqa: E402


def build_actor(num_obs: int, num_actions: int, device) -> nn.Module:
    """ELU MLP 512-256-128 matching the PPO actor in train_cfg.json."""
    net = nn.Sequential(
        nn.Linear(num_obs, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, num_actions),
    ).to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def measure(num_envs, context, warmup, steps, device):
    cfg = default_env_cfg()
    cfg["num_envs"] = num_envs
    cfg["device"] = str(device)
    cfg["dynamics_context_steps"] = context
    torch.cuda.reset_peak_memory_stats()
    env = HMMWVNeuralTrackingEnv(cfg, device=device)
    actor = build_actor(env.num_obs, env.num_actions, device)
    obs, _ = env.get_observations()

    for _ in range(warmup):
        obs, _, _, extras = env.step(actor(obs))
        obs = extras["observations"]["critic"]
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        obs, _, _, extras = env.step(actor(obs))
        obs = extras["observations"]["critic"]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    steps_per_s = steps * num_envs / elapsed
    iter_s_128 = 128 * num_envs / steps_per_s  # collection time for a 128-step PPO iter
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del env, actor, obs, extras
    torch.cuda.empty_cache()
    return steps_per_s, iter_s_128, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--num-envs", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=60)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")  # matches train_cfg matmul_precision=high

    print(f"device={torch.cuda.get_device_name(0)}  matmul=high  warmup={args.warmup} "
          f"timed_steps={args.steps}")
    print("Metric = collection steps/s (policy_steps x n_env / s); higher is better.\n")

    results = {}
    for ctx in args.contexts:
        for n in args.num_envs:
            try:
                sps, it128, gb = measure(n, ctx, args.warmup, args.steps, device)
                results[(ctx, n)] = (sps, it128, gb)
                print(f"  ctx={ctx:>2} n_env={n:>5}: {sps/1e3:7.1f}K steps/s | "
                      f"{it128:6.2f}s/iter(128) | peak {gb:5.2f} GB")
            except RuntimeError as e:
                results[(ctx, n)] = None
                print(f"  ctx={ctx:>2} n_env={n:>5}: ERR {str(e)[:50]}")

    def table(title, pick, fmt):
        print(f"\n=== {title} ===")
        print("ctx \\ n_env | " + " | ".join(f"{n:>8}" for n in args.num_envs))
        print("-" * (12 + 11 * len(args.num_envs)))
        for ctx in args.contexts:
            cells = []
            for n in args.num_envs:
                r = results.get((ctx, n))
                cells.append(fmt(pick(r)) if r else "    ERR ")
            print(f"{ctx:>10} | " + " | ".join(cells))

    table("Collection throughput (K steps/s)", lambda r: r[0] / 1e3, lambda v: f"{v:8.1f}")
    table("Peak GPU memory (GB)", lambda r: r[2], lambda v: f"{v:8.2f}")
    print("\nReading: for a fixed ctx, if throughput is ~flat as n_env grows, the GPU is "
          "already saturated at the smaller n_env -- extra envs cost memory + per-iter time "
          "without raising steps/s. The sweet-spot n_env is the smallest one still at peak "
          "throughput (best samples-per-update without paying for idle headroom).")


if __name__ == "__main__":
    main()
