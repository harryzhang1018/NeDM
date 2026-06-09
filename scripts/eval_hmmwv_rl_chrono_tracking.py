from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.hmmwv_chrono_tracking_env import HMMWVChronoTrackingEnv
from nedm.rl.references import load_reference_set


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HMMWV RL policy on Chrono HMMWV.")
    parser.add_argument("--run-dir", type=Path, required=True, help="RL run directory with env_cfg/train_cfg JSON.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None, help="Policy model_*.pt. Defaults latest.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Policy/observation tensor device. Chrono itself is CPU-bound.",
    )
    parser.add_argument(
        "--chrono-config",
        type=Path,
        default=Path("configs/hmmwv_overfit_v1.json"),
        help="Collector config that defines HMMWV and terrain setup.",
    )
    parser.add_argument("--num-references", type=int, default=None, help="Number of references to evaluate.")
    parser.add_argument("--reference-index", type=int, default=None, help="Evaluate only one reference index.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional policy-step limit.")
    parser.add_argument(
        "--ignore-dones",
        action="store_true",
        help="Continue plotting until max-steps even after tracking termination.",
    )
    parser.add_argument(
        "--chrono-step-size-s",
        type=float,
        default=None,
        help="Optional Chrono solver step size. With default action_repeat=5 and dt=0.01, 0.01 gives 5 solver steps per policy update.",
    )
    parser.add_argument(
        "--steering-rate-limit",
        type=float,
        default=None,
        help="Clamp the steering command to within this offset of the previous policy step's steering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and summary. Defaults under run-dir/chrono_eval_tracking.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the rollout with Chrono's Irrlicht renderer and save one PNG per frame "
        "(needs an X/GL context: a desktop session or `xvfb-run`). Requires --reference-index.",
    )
    parser.add_argument("--render-fps", type=float, default=50.0, help="Saved-frame rate for --render.")
    parser.add_argument("--render-width", type=int, default=1280, help="Render frame width.")
    parser.add_argument("--render-height", type=int, default=720, help="Render frame height.")
    return parser.parse_args(argv)


def latest_policy_checkpoint(run_dir: Path) -> Path:
    candidates = list(run_dir.glob("model_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")

    def checkpoint_iter(path: Path) -> int:
        match = re.match(r"model_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=checkpoint_iter)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def maybe_encode_video(frames_dir: Path, fps: float) -> None:
    """Encode saved PNG frames into an mp4 next to the frames dir, if ffmpeg is available."""
    import shutil
    import subprocess

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"no frames written in {frames_dir} (did the renderer get a display?)")
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"saved {len(frames)} frames in {frames_dir}; ffmpeg not found, skipping mp4 encode.")
        return
    output_path = frames_dir.with_suffix(".mp4")
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.png"),
         "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output_path)],
        check=True,
    )
    print(f"encoded {len(frames)} frames -> {output_path}")


def load_runner_checkpoint(runner: OnPolicyRunner, checkpoint_path: Path, device: str) -> None:
    loaded_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    if runner.alg.rnd:
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = int(loaded_dict["iter"])


def rollout_one_reference(
    env: HMMWVChronoTrackingEnv,
    policy,
    reference_id: int,
    max_steps: int,
    reset: bool = True,
    ignore_dones: bool = False,
    render: bool = False,
    render_output_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    if reset:
        env_id = torch.tensor([0], dtype=torch.long, device=env.device)
        ref_id = torch.tensor([reference_id], dtype=torch.long, device=env.device)
        env.reset_idx(env_id, reference_ids=ref_id)
    if render:
        frames_dir = env.start_render(reference_id, output_dir=render_output_dir)
        print(f"rendering frames -> {frames_dir}")
    obs, _ = env.get_observations()

    record: dict[str, list[np.ndarray]] = {
        "pose": [],
        "state": [],
        "ref_pose": [],
        "ref_state": [],
        "action": [],
        "reward": [],
    }

    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs.to(env.device))
            obs, rewards, dones, _ = env.step(actions)
            ref_state, ref_pose = env.current_reference_state_pose()
            record["pose"].append(tensor_to_numpy(env.current_pose()[0]))
            record["state"].append(tensor_to_numpy(env.current_state()[0]))
            record["ref_pose"].append(tensor_to_numpy(ref_pose[0]))
            record["ref_state"].append(tensor_to_numpy(ref_state[0]))
            record["action"].append(tensor_to_numpy(env.actions[0]))
            record["reward"].append(np.array(float(rewards[0].item()), dtype=np.float32))
            if bool(dones[0].item()) and not ignore_dones:
                break

    return {
        key: np.stack(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for key, values in record.items()
    }


def compute_metrics(record: dict[str, np.ndarray]) -> dict[str, float]:
    pose = record["pose"]
    ref_pose = record["ref_pose"]
    if pose.size == 0:
        return {"steps": 0.0}
    xy_error = np.linalg.norm(pose[:, :2] - ref_pose[:, :2], axis=1)
    yaw_error = np.arctan2(np.sin(pose[:, 2] - ref_pose[:, 2]), np.cos(pose[:, 2] - ref_pose[:, 2]))
    return {
        "steps": float(pose.shape[0]),
        "xy_rmse_m": float(np.sqrt(np.mean(np.square(xy_error)))),
        "xy_mean_m": float(np.mean(xy_error)),
        "xy_final_m": float(xy_error[-1]),
        "yaw_rmse_rad": float(np.sqrt(np.mean(np.square(yaw_error)))),
        "reward_sum": float(np.sum(record["reward"])),
    }


def plot_record(
    record: dict[str, np.ndarray],
    env: HMMWVChronoTrackingEnv,
    reference_id: int,
    output_path: Path,
) -> None:
    pose = record["pose"]
    ref_pose = record["ref_pose"]
    state = record["state"]
    ref_state = record["ref_state"]
    actions = record["action"]
    if pose.size == 0:
        return

    time_s = np.arange(pose.shape[0], dtype=np.float32) * env.step_dt
    xy_error = np.linalg.norm(pose[:, :2] - ref_pose[:, :2], axis=1)
    yaw_error = np.arctan2(np.sin(pose[:, 2] - ref_pose[:, 2]), np.cos(pose[:, 2] - ref_pose[:, 2]))
    vx_idx = env.state_index["vel_body_x_mps"]
    yaw_rate_idx = env.state_index["yaw_rate_radps"]
    reference_name = env.reference_names()[reference_id]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ref_pose[:, 0], ref_pose[:, 1], label="reference", linewidth=2.0, color="#0f766e")
    ax.plot(pose[:, 0], pose[:, 1], label="chrono policy", linewidth=1.8, color="#dc2626")
    ax.scatter(ref_pose[0, 0], ref_pose[0, 1], label="reference start", marker="o", s=34, color="#0f766e")
    ax.scatter(ref_pose[-1, 0], ref_pose[-1, 1], label="reference end", marker="s", s=34, color="#0f766e")
    ax.scatter(pose[0, 0], pose[0, 1], label="chrono start", marker="o", s=44, color="#dc2626")
    ax.scatter(pose[-1, 0], pose[-1, 1], label="chrono end", marker="s", s=44, color="#dc2626")
    ax.set_title("XY Tracking")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(time_s, xy_error, label="xy error [m]", color="#7c3aed")
    ax.plot(time_s, np.abs(yaw_error), label="abs yaw error [rad]", color="#d97706")
    ax.set_title("Tracking Error")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(time_s, ref_state[:, vx_idx], label="reference vx", color="#0f766e")
    ax.plot(time_s, state[:, vx_idx], label="chrono vx", color="#dc2626")
    ax.plot(time_s, ref_state[:, yaw_rate_idx], label="reference yaw rate", color="#1d4ed8", alpha=0.8)
    ax.plot(time_s, state[:, yaw_rate_idx], label="chrono yaw rate", color="#b45309", alpha=0.8)
    ax.set_title("State Tracking")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(time_s, actions[:, 0], label="steer", color="#1d4ed8")
    ax.plot(time_s, actions[:, 1], label="throttle", color="#15803d")
    ax.plot(time_s, actions[:, 2], label="brake", color="#b45309")
    ax.set_title("Policy Actions")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Chrono HMMWV RL Tracking: {reference_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    env_cfg = load_json(run_dir / "env_cfg.json")
    train_cfg = load_json(run_dir / "train_cfg.json")
    env_cfg["device"] = args.device
    env_cfg["auto_reset"] = False
    env_cfg["num_envs"] = 1
    env_cfg["chrono_config"] = str(args.chrono_config)
    if args.chrono_step_size_s is not None:
        env_cfg["chrono_step_size_s"] = float(args.chrono_step_size_s)
    if args.steering_rate_limit is not None:
        env_cfg["steering_rate_limit"] = float(args.steering_rate_limit)
    if args.render:
        if args.reference_index is None:
            raise ValueError("--render renders a single rollout; pass --reference-index too.")
        env_cfg["render"] = True
        env_cfg["render_fps"] = float(args.render_fps)
        env_cfg["render_width"] = int(args.render_width)
        env_cfg["render_height"] = int(args.render_height)

    checkpoint_path = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    reference_count = load_reference_set(env_cfg["reference_path"]).num_references
    if args.reference_index is not None:
        if args.reference_index < 0 or args.reference_index >= reference_count:
            raise ValueError(f"reference-index must be in [0, {reference_count - 1}]")
        reference_indices = [int(args.reference_index)]
    else:
        num_references = reference_count if args.num_references is None else min(int(args.num_references), reference_count)
        reference_indices = list(range(num_references))
    env_cfg["initial_reference_ids"] = [reference_indices[0]]

    env = HMMWVChronoTrackingEnv(env_cfg, device=args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=args.device)
    load_runner_checkpoint(runner, checkpoint_path, device=args.device)
    policy = runner.get_inference_policy(device=args.device)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_length)

    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "chrono_eval_tracking").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for index, reference_id in enumerate(reference_indices):
        record = rollout_one_reference(
            env,
            policy,
            reference_id=reference_id,
            max_steps=max_steps,
            reset=index > 0,
            ignore_dones=bool(args.ignore_dones),
            render=bool(args.render),
            render_output_dir=(output_dir / f"frames_ref{reference_id:02d}") if args.render else None,
        )
        if args.render:
            maybe_encode_video(output_dir / f"frames_ref{reference_id:02d}", args.render_fps)
        metrics = compute_metrics(record)
        metrics["reference"] = env.reference_names()[reference_id]
        summary.append(metrics)
        np.savez_compressed(output_dir / f"chrono_tracking_{reference_id:02d}.npz", **record)
        plot_record(record, env, reference_id, output_dir / f"chrono_tracking_{reference_id:02d}.png")
        print(json.dumps(metrics))

    valid_xy = [row["xy_rmse_m"] for row in summary if "xy_rmse_m" in row]
    aggregate = {
        "backend": "chrono_hmmwv",
        "policy_checkpoint": str(checkpoint_path),
        "chrono_config": str(args.chrono_config),
        "num_rollouts": len(summary),
        "mean_xy_rmse_m": float(np.mean(valid_xy)) if valid_xy else None,
        "median_xy_rmse_m": float(np.median(valid_xy)) if valid_xy else None,
        "rollouts": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
