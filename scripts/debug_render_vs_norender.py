"""Locate where a Chrono rollout diverges with vs. without the Irrlicht renderer active.

The GL driver flips the CPU SSE flags (flush-to-zero / denormals-are-zero) at context
creation, which perturbs Chrono's solver. Because that flag change is process-wide, the
rendered and non-rendered rollouts must be produced in *separate processes*. This tool runs
one single-reference rollout per invocation (recording per-step obs/state/pose/action), then a
``--compare`` mode lines up two such recordings and reports the divergence onset and growth.

Usage:
    python scripts/debug_render_vs_norender.py --run-dir <run> --reference-index 9 \
        --out /tmp/dbg/norender.npz
    DISPLAY=:1 python scripts/debug_render_vs_norender.py --run-dir <run> --reference-index 9 \
        --render --out /tmp/dbg/render.npz
    python scripts/debug_render_vs_norender.py --compare /tmp/dbg/norender.npz /tmp/dbg/render.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def latest_policy_checkpoint(run_dir: Path) -> Path:
    import re

    candidates = list(run_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt in {run_dir}")
    return max(candidates, key=lambda p: int(re.match(r"model_(\d+)\.pt$", p.name).group(1)))


def run_rollout(args: argparse.Namespace) -> None:
    from rsl_rl.runners import OnPolicyRunner

    from nedm.rl.hmmwv_chrono_tracking_env import HMMWVChronoTrackingEnv

    run_dir = args.run_dir.resolve()
    env_cfg = load_json(run_dir / "env_cfg.json")
    train_cfg = load_json(run_dir / "train_cfg.json")
    env_cfg.update(
        {
            "device": "cpu",
            "num_envs": 1,
            "auto_reset": False,
            "chrono_config": str(args.chrono_config),
            "initial_reference_ids": [int(args.reference_index)],
        }
    )
    if args.render:
        env_cfg.update({"render": True, "render_fps": 50.0})

    checkpoint = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    env = HMMWVChronoTrackingEnv(env_cfg, device="cpu")
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device="cpu")
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    policy = runner.get_inference_policy(device="cpu")

    env_id = torch.tensor([0], dtype=torch.long)
    ref_id = torch.tensor([int(args.reference_index)], dtype=torch.long)
    env.reset_idx(env_id, reference_ids=ref_id)
    if args.render:
        frames = env.start_render(int(args.reference_index), output_dir=str(Path(args.out).with_suffix("")) + "_frames")
        print(f"rendering active -> {frames}")
    obs, _ = env.get_observations()

    rec: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "action", "state", "pose", "ref_pose")}
    max_steps = int(args.max_steps) if args.max_steps else int(env.max_episode_length)
    with torch.no_grad():
        for _ in range(max_steps):
            rec["obs"].append(obs[0].cpu().numpy().copy())
            action = policy(obs)
            rec["action"].append(action[0].cpu().numpy().copy())
            obs, _, dones, _ = env.step(action)
            ref_state, ref_pose = env.current_reference_state_pose()
            rec["state"].append(env.current_state()[0].cpu().numpy().copy())
            rec["pose"].append(env.current_pose()[0].cpu().numpy().copy())
            rec["ref_pose"].append(ref_pose[0].cpu().numpy().copy())
            if bool(dones[0].item()):
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    packed = {k: np.stack(v, axis=0) for k, v in rec.items()}
    packed["state_fields"] = np.array(env.state_fields)
    packed["rendered"] = np.array(bool(args.render))
    np.savez_compressed(out, **packed)
    final_xy = float(np.linalg.norm(packed["pose"][-1, :2] - packed["ref_pose"][-1, :2]))
    print(f"saved {packed['obs'].shape[0]} steps -> {out}  (rendered={bool(args.render)}, final_xy_err={final_xy:.3f} m)")


def compare(path_a: Path, path_b: Path) -> None:
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)
    fields = list(a["state_fields"])
    n = min(a["obs"].shape[0], b["obs"].shape[0])
    obs_d = b["obs"][:n] - a["obs"][:n]
    act_d = b["action"][:n] - a["action"][:n]
    pos_d = np.linalg.norm(b["pose"][:n, :2] - a["pose"][:n, :2], axis=1)
    state_d = b["state"][:n] - a["state"][:n]

    obs_l2 = np.linalg.norm(obs_d, axis=1)
    act_l2 = np.linalg.norm(act_d, axis=1)
    print(f"comparing A={path_a.name} (rendered={bool(a['rendered'])})  B={path_b.name} (rendered={bool(b['rendered'])})")
    print(f"{'step':>4} {'obs_l2':>10} {'action_l2':>10} {'pos_diff_m':>11} {'vx_diff':>9} {'yawrate_diff':>13}")
    vx, yr = fields.index("vel_body_x_mps"), fields.index("yaw_rate_radps")
    for i in range(n):
        if i < 12 or i % 10 == 0 or i == n - 1:
            print(f"{i:>4} {obs_l2[i]:>10.2e} {act_l2[i]:>10.2e} {pos_d[i]:>11.4f} "
                  f"{state_d[i, vx]:>9.4f} {state_d[i, yr]:>13.4f}")

    def first_exceed(arr: np.ndarray, thr: float) -> int:
        idx = np.flatnonzero(arr > thr)
        return int(idx[0]) if idx.size else -1

    print("\ndivergence onset:")
    print(f"  first obs_l2  > 1e-4 : step {first_exceed(obs_l2, 1e-4)}")
    print(f"  first pos_diff> 1e-3 m: step {first_exceed(pos_d, 1e-3)}")
    print(f"  first pos_diff> 0.1 m : step {first_exceed(pos_d, 0.1)}")
    print(f"  first pos_diff> 1.0 m : step {first_exceed(pos_d, 1.0)}")
    print(f"  final pos_diff       : {pos_d[-1]:.3f} m at step {n - 1}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--policy-checkpoint", type=Path, default=None)
    p.add_argument("--reference-index", type=int, default=9)
    p.add_argument("--chrono-config", type=Path, default=Path("configs/hmmwv_overfit_v1.json"))
    p.add_argument("--render", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--compare", type=Path, nargs=2, default=None, metavar=("A", "B"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare is not None:
        compare(args.compare[0].resolve(), args.compare[1].resolve())
        return 0
    if args.run_dir is None or args.out is None:
        raise SystemExit("Provide --run-dir and --out for a rollout, or --compare A B.")
    run_rollout(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
