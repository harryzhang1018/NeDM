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

from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv
from nedm.rl.references import load_reference_set


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HMMWV RL tracking policy.")
    parser.add_argument("--run-dir", type=Path, required=True, help="RL run directory with env_cfg/train_cfg JSON.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None, help="Policy model_*.pt. Defaults latest.")
    parser.add_argument("--device", type=str, default=None, help="Override device from env_cfg.")
    parser.add_argument("--num-references", type=int, default=None, help="Number of references to evaluate.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional policy-step limit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plots and summary. Defaults under run-dir/eval_tracking.",
    )
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


def load_runner_checkpoint(runner: OnPolicyRunner, checkpoint_path: Path, device: str) -> None:
    loaded_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    if runner.alg.rnd:
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = int(loaded_dict["iter"])


def rollout_policy(
    env: HMMWVNeuralTrackingEnv,
    policy,
    max_steps: int,
) -> list[dict[str, np.ndarray]]:
    num_envs = env.num_envs
    env_ids = torch.arange(num_envs, device=env.device)
    reference_ids = torch.arange(num_envs, device=env.device) % env.num_references
    env.reset_idx(env_ids, reference_ids=reference_ids)
    obs, _ = env.get_observations()
    done_mask = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    records: list[dict[str, list[np.ndarray]]] = [
        {"pose": [], "state": [], "ref_pose": [], "ref_state": [], "action": [], "reward": []}
        for _ in range(num_envs)
    ]

    with torch.no_grad():
        for _ in range(max_steps):
            actions = policy(obs.to(env.device))
            obs, rewards, dones, _ = env.step(actions)
            ref_state, ref_pose = env.current_reference_state_pose()
            current_pose = env.current_pose()
            current_state = env.current_state()
            active_ids = (~done_mask).nonzero(as_tuple=False).flatten()
            for env_index_tensor in active_ids:
                env_index = int(env_index_tensor.item())
                records[env_index]["pose"].append(tensor_to_numpy(current_pose[env_index]))
                records[env_index]["state"].append(tensor_to_numpy(current_state[env_index]))
                records[env_index]["ref_pose"].append(tensor_to_numpy(ref_pose[env_index]))
                records[env_index]["ref_state"].append(tensor_to_numpy(ref_state[env_index]))
                records[env_index]["action"].append(tensor_to_numpy(env.actions[env_index]))
                records[env_index]["reward"].append(np.array(float(rewards[env_index].item()), dtype=np.float32))
            done_mask |= dones.bool()
            if bool(torch.all(done_mask).item()):
                break

    packed_records: list[dict[str, np.ndarray]] = []
    for record in records:
        packed_records.append(
            {
                key: np.stack(values, axis=0) if values else np.empty((0,), dtype=np.float32)
                for key, values in record.items()
            }
        )
    return packed_records


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
    env: HMMWVNeuralTrackingEnv,
    env_index: int,
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
    reference_name = env.reference_names()[env_index % env.num_references]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ref_pose[:, 0], ref_pose[:, 1], label="reference", linewidth=2.0, color="#0f766e")
    ax.plot(pose[:, 0], pose[:, 1], label="policy", linewidth=1.6, color="#dc2626")
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
    ax.plot(time_s, state[:, vx_idx], label="policy vx", color="#dc2626")
    ax.plot(time_s, ref_state[:, yaw_rate_idx], label="reference yaw rate", color="#1d4ed8", alpha=0.8)
    ax.plot(time_s, state[:, yaw_rate_idx], label="policy yaw rate", color="#b45309", alpha=0.8)
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

    fig.suptitle(f"HMMWV RL Tracking: {reference_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    env_cfg = load_json(run_dir / "env_cfg.json")
    train_cfg = load_json(run_dir / "train_cfg.json")
    if args.device is not None:
        env_cfg["device"] = args.device
    env_cfg["auto_reset"] = False

    checkpoint_path = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    reference_count = load_reference_set(env_cfg["reference_path"]).num_references
    env_cfg["num_envs"] = int(args.num_references) if args.num_references is not None else reference_count

    device = env_cfg["device"]
    env = HMMWVNeuralTrackingEnv(env_cfg, device=device)

    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    load_runner_checkpoint(runner, checkpoint_path, device=device)
    policy = runner.get_inference_policy(device=device)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_length)
    records = rollout_policy(env, policy, max_steps=max_steps)

    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "eval_tracking").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for env_index, record in enumerate(records):
        metrics = compute_metrics(record)
        metrics["reference"] = env.reference_names()[env_index % env.num_references]
        summary.append(metrics)
        plot_record(record, env, env_index, output_dir / f"tracking_{env_index:02d}.png")

    aggregate = {
        "policy_checkpoint": str(checkpoint_path),
        "num_rollouts": len(summary),
        "mean_xy_rmse_m": float(np.mean([row["xy_rmse_m"] for row in summary if "xy_rmse_m" in row])),
        "median_xy_rmse_m": float(np.median([row["xy_rmse_m"] for row in summary if "xy_rmse_m" in row])),
        "rollouts": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
