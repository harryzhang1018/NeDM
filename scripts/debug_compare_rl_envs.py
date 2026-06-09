from __future__ import annotations

import argparse
import copy
import csv
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
from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv
from nedm.rl.references import load_reference_set


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one closed-loop PPO rollout in NN dynamics and Chrono envs."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="RL run directory with env_cfg/train_cfg JSON.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None, help="Policy model_*.pt. Defaults latest.")
    parser.add_argument("--device", type=str, default="cpu", help="Policy/env tensor device.")
    parser.add_argument("--reference-index", type=int, default=9, help="Reference index to debug.")
    parser.add_argument("--max-steps", type=int, default=40, help="Policy steps to roll out. 40 = 2s at 0.05s.")
    parser.add_argument(
        "--chrono-config",
        type=Path,
        default=Path("configs/hmmwv_overfit_v1.json"),
        help="Collector config that defines HMMWV and terrain setup.",
    )
    parser.add_argument(
        "--chrono-step-size-s",
        type=float,
        default=0.002,
        help="Chrono solver step. The data collection default is 0.002; 0.01 gives 5 solver steps per policy update.",
    )
    parser.add_argument(
        "--no-warm-start-context",
        action="store_true",
        help="Disable Chrono context replay and use the older direct pose/speed reset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults under run-dir/debug_refXX_nn_vs_chrono_2s.",
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


def make_env_cfg(
    run_dir: Path,
    device: str,
    reference_index: int,
    chrono_config: Path,
    chrono_step_size_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_cfg = load_json(run_dir / "env_cfg.json")
    common_cfg = dict(base_cfg)
    common_cfg.update(
        {
            "device": device,
            "num_envs": 1,
            "auto_reset": False,
        }
    )

    nn_cfg = dict(common_cfg)

    chrono_cfg = dict(common_cfg)
    chrono_cfg.update(
        {
            "chrono_config": str(chrono_config),
            "chrono_step_size_s": float(chrono_step_size_s),
            "initial_reference_ids": [int(reference_index)],
        }
    )
    return nn_cfg, chrono_cfg


def build_policy(
    env: HMMWVNeuralTrackingEnv,
    train_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: str,
):
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir=None, device=device)
    load_runner_checkpoint(runner, checkpoint_path, device=device)
    return runner.get_inference_policy(device=device)


def current_snapshot(env: HMMWVNeuralTrackingEnv | HMMWVChronoTrackingEnv) -> dict[str, np.ndarray]:
    ref_state, ref_pose = env.current_reference_state_pose()
    return {
        "state": tensor_to_numpy(env.current_state()[0]),
        "pose": tensor_to_numpy(env.current_pose()[0]),
        "ref_state": tensor_to_numpy(ref_state[0]),
        "ref_pose": tensor_to_numpy(ref_pose[0]),
        "driver_action": tensor_to_numpy(env.actions[0]),
        "ref_step": np.array(int(env.ref_step_buf[0].item()), dtype=np.int64),
    }


def rollout_debug(
    name: str,
    env: HMMWVNeuralTrackingEnv | HMMWVChronoTrackingEnv,
    policy,
    reference_index: int,
    max_steps: int,
) -> dict[str, np.ndarray]:
    env_id = torch.tensor([0], dtype=torch.long, device=env.device)
    ref_id = torch.tensor([reference_index], dtype=torch.long, device=env.device)
    env.reset_idx(env_id, reference_ids=ref_id)
    obs, _ = env.get_observations()

    records: dict[str, list[np.ndarray]] = {
        "step": [],
        "time_s": [],
        "obs_before": [],
        "raw_action": [],
        "driver_action_cmd": [],
        "state_before": [],
        "pose_before": [],
        "ref_state_before": [],
        "ref_pose_before": [],
        "ref_step_before": [],
        "state_after": [],
        "pose_after": [],
        "ref_state_after": [],
        "ref_pose_after": [],
        "ref_step_after": [],
        "reward": [],
        "done": [],
    }

    with torch.no_grad():
        for step in range(max_steps):
            before = current_snapshot(env)
            raw_action = policy(obs.to(env.device))
            driver_action_cmd = env._scale_policy_actions(raw_action)  # noqa: SLF001 - debug parity check.

            obs_next, rewards, dones, _ = env.step(raw_action)
            after = current_snapshot(env)

            records["step"].append(np.array(step, dtype=np.int64))
            records["time_s"].append(np.array(step * env.step_dt, dtype=np.float32))
            records["obs_before"].append(tensor_to_numpy(obs[0]))
            records["raw_action"].append(tensor_to_numpy(raw_action[0]))
            records["driver_action_cmd"].append(tensor_to_numpy(driver_action_cmd[0]))
            records["state_before"].append(before["state"])
            records["pose_before"].append(before["pose"])
            records["ref_state_before"].append(before["ref_state"])
            records["ref_pose_before"].append(before["ref_pose"])
            records["ref_step_before"].append(before["ref_step"])
            records["state_after"].append(after["state"])
            records["pose_after"].append(after["pose"])
            records["ref_state_after"].append(after["ref_state"])
            records["ref_pose_after"].append(after["ref_pose"])
            records["ref_step_after"].append(after["ref_step"])
            records["reward"].append(np.array(float(rewards[0].item()), dtype=np.float32))
            records["done"].append(np.array(bool(dones[0].item()), dtype=np.bool_))
            obs = obs_next

    packed = {
        key: np.stack(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for key, values in records.items()
    }
    packed["backend"] = np.array(name)
    return packed


def obs_slices(env: HMMWVNeuralTrackingEnv) -> dict[str, slice]:
    state_dim = len(env.state_fields)
    action_dim = len(env.action_fields)
    cursor = 0
    slices: dict[str, slice] = {}
    slices["history_states"] = slice(cursor, cursor + env.obs_history_steps * state_dim)
    cursor = slices["history_states"].stop
    slices["history_actions"] = slice(cursor, cursor + env.obs_history_steps * action_dim)
    cursor = slices["history_actions"].stop
    slices["state_error"] = slice(cursor, cursor + state_dim)
    cursor = slices["state_error"].stop
    slices["pose_error"] = slice(cursor, cursor + 3)
    cursor = slices["pose_error"].stop
    slices["reference_preview"] = slice(cursor, cursor + env.reference_preview_steps * 3)
    cursor = slices["reference_preview"].stop
    slices["last_action"] = slice(cursor, cursor + action_dim)
    return slices


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def compute_tracking_metrics(record: dict[str, np.ndarray]) -> dict[str, float]:
    pose = record["pose_after"]
    ref_pose = record["ref_pose_after"]
    xy_error = np.linalg.norm(pose[:, :2] - ref_pose[:, :2], axis=1)
    yaw_error = angle_error(pose[:, 2], ref_pose[:, 2])
    return {
        "steps": float(len(xy_error)),
        "xy_initial_m": float(xy_error[0]) if len(xy_error) else 0.0,
        "xy_final_m": float(xy_error[-1]) if len(xy_error) else 0.0,
        "xy_rmse_m": rmse(xy_error),
        "yaw_initial_rad": float(abs(yaw_error[0])) if len(yaw_error) else 0.0,
        "yaw_final_rad": float(abs(yaw_error[-1])) if len(yaw_error) else 0.0,
        "yaw_rmse_rad": rmse(yaw_error),
        "reward_sum": float(np.sum(record["reward"])),
        "done_step": float(np.flatnonzero(record["done"])[0]) if np.any(record["done"]) else -1.0,
    }


def compute_comparison(
    nn_record: dict[str, np.ndarray],
    chrono_record: dict[str, np.ndarray],
    env: HMMWVNeuralTrackingEnv,
) -> dict[str, Any]:
    obs_diff = chrono_record["obs_before"] - nn_record["obs_before"]
    raw_action_diff = chrono_record["raw_action"] - nn_record["raw_action"]
    driver_action_diff = chrono_record["driver_action_cmd"] - nn_record["driver_action_cmd"]
    pose_diff = chrono_record["pose_after"] - nn_record["pose_after"]
    pose_diff[:, 2] = angle_error(chrono_record["pose_after"][:, 2], nn_record["pose_after"][:, 2])
    state_diff = chrono_record["state_after"] - nn_record["state_after"]

    slice_metrics = {}
    for name, obs_slice in obs_slices(env).items():
        diff = obs_diff[:, obs_slice]
        slice_metrics[name] = {
            "rmse": rmse(diff),
            "max_abs": max_abs(diff),
            "initial_max_abs": max_abs(diff[0]),
        }

    state_initial = chrono_record["state_before"][0] - nn_record["state_before"][0]
    state_initial_by_field = {
        field: float(state_initial[index])
        for index, field in enumerate(env.state_fields)
    }
    state_rmse_by_field = {
        field: rmse(state_diff[:, index])
        for index, field in enumerate(env.state_fields)
    }

    return {
        "nn_metrics": compute_tracking_metrics(nn_record),
        "chrono_metrics": compute_tracking_metrics(chrono_record),
        "compare": {
            "obs_rmse": rmse(obs_diff),
            "obs_max_abs": max_abs(obs_diff),
            "obs_initial_max_abs": max_abs(obs_diff[0]),
            "raw_action_rmse": rmse(raw_action_diff),
            "raw_action_max_abs": max_abs(raw_action_diff),
            "driver_action_rmse": rmse(driver_action_diff),
            "driver_action_max_abs": max_abs(driver_action_diff),
            "pose_after_xy_final_diff_m": float(np.linalg.norm(pose_diff[-1, :2])),
            "pose_after_xy_rmse_diff_m": rmse(np.linalg.norm(pose_diff[:, :2], axis=1)),
            "pose_after_yaw_final_diff_rad": float(abs(pose_diff[-1, 2])),
            "state_after_rmse": rmse(state_diff),
            "slice_metrics": slice_metrics,
            "state_initial_diff_by_field": state_initial_by_field,
            "state_after_rmse_by_field": state_rmse_by_field,
        },
    }


def write_step_csv(
    path: Path,
    nn_record: dict[str, np.ndarray],
    chrono_record: dict[str, np.ndarray],
    env: HMMWVNeuralTrackingEnv,
) -> None:
    vx_idx = env.state_index["vel_body_x_mps"]
    vy_idx = env.state_index["vel_body_y_mps"]
    yaw_rate_idx = env.state_index["yaw_rate_radps"]
    rows = []
    for i in range(nn_record["step"].shape[0]):
        nn_pose = nn_record["pose_after"][i]
        chrono_pose = chrono_record["pose_after"][i]
        ref_pose = nn_record["ref_pose_after"][i]
        nn_state = nn_record["state_after"][i]
        chrono_state = chrono_record["state_after"][i]
        nn_xy_err = float(np.linalg.norm(nn_pose[:2] - ref_pose[:2]))
        chrono_xy_err = float(np.linalg.norm(chrono_pose[:2] - ref_pose[:2]))
        rows.append(
            {
                "step": int(nn_record["step"][i]),
                "time_s": float(nn_record["time_s"][i]),
                "ref_step": int(nn_record["ref_step_after"][i]),
                "nn_x": float(nn_pose[0]),
                "nn_y": float(nn_pose[1]),
                "nn_yaw": float(nn_pose[2]),
                "chrono_x": float(chrono_pose[0]),
                "chrono_y": float(chrono_pose[1]),
                "chrono_yaw": float(chrono_pose[2]),
                "ref_x": float(ref_pose[0]),
                "ref_y": float(ref_pose[1]),
                "ref_yaw": float(ref_pose[2]),
                "nn_xy_error_m": nn_xy_err,
                "chrono_xy_error_m": chrono_xy_err,
                "nn_vx": float(nn_state[vx_idx]),
                "chrono_vx": float(chrono_state[vx_idx]),
                "nn_vy": float(nn_state[vy_idx]),
                "chrono_vy": float(chrono_state[vy_idx]),
                "nn_yaw_rate": float(nn_state[yaw_rate_idx]),
                "chrono_yaw_rate": float(chrono_state[yaw_rate_idx]),
                "nn_steer": float(nn_record["driver_action_cmd"][i, 0]),
                "nn_throttle": float(nn_record["driver_action_cmd"][i, 1]),
                "nn_brake": float(nn_record["driver_action_cmd"][i, 2]),
                "chrono_steer": float(chrono_record["driver_action_cmd"][i, 0]),
                "chrono_throttle": float(chrono_record["driver_action_cmd"][i, 1]),
                "chrono_brake": float(chrono_record["driver_action_cmd"][i, 2]),
                "obs_l2_diff": float(np.linalg.norm(chrono_record["obs_before"][i] - nn_record["obs_before"][i])),
                "action_l2_diff": float(
                    np.linalg.norm(chrono_record["driver_action_cmd"][i] - nn_record["driver_action_cmd"][i])
                ),
            }
        )

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    path: Path,
    nn_record: dict[str, np.ndarray],
    chrono_record: dict[str, np.ndarray],
    env: HMMWVNeuralTrackingEnv,
    reference_name: str,
) -> None:
    time_s = nn_record["time_s"]
    ref_pose = nn_record["ref_pose_after"]
    nn_pose = nn_record["pose_after"]
    chrono_pose = chrono_record["pose_after"]
    nn_xy_error = np.linalg.norm(nn_pose[:, :2] - ref_pose[:, :2], axis=1)
    chrono_xy_error = np.linalg.norm(chrono_pose[:, :2] - ref_pose[:, :2], axis=1)
    vx_idx = env.state_index["vel_body_x_mps"]
    yaw_rate_idx = env.state_index["yaw_rate_radps"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ref_pose[:, 0], ref_pose[:, 1], label="reference", linewidth=2.0, color="#0f766e")
    ax.plot(nn_pose[:, 0], nn_pose[:, 1], label="NN env", linewidth=1.8, color="#1d4ed8")
    ax.plot(chrono_pose[:, 0], chrono_pose[:, 1], label="Chrono env", linewidth=1.8, color="#dc2626")
    ax.scatter(ref_pose[0, 0], ref_pose[0, 1], marker="o", s=36, color="#0f766e")
    ax.scatter(nn_pose[0, 0], nn_pose[0, 1], marker="o", s=34, color="#1d4ed8")
    ax.scatter(chrono_pose[0, 0], chrono_pose[0, 1], marker="o", s=34, color="#dc2626")
    ax.set_title("XY Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(time_s, nn_xy_error, label="NN xy error", color="#1d4ed8")
    ax.plot(time_s, chrono_xy_error, label="Chrono xy error", color="#dc2626")
    ax.set_title("Tracking Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(time_s, nn_record["ref_state_after"][:, vx_idx], label="reference vx", color="#0f766e")
    ax.plot(time_s, nn_record["state_after"][:, vx_idx], label="NN vx", color="#1d4ed8")
    ax.plot(time_s, chrono_record["state_after"][:, vx_idx], label="Chrono vx", color="#dc2626")
    ax.set_title("Forward Speed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m/s")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(time_s, nn_record["ref_state_after"][:, yaw_rate_idx], label="reference yaw rate", color="#0f766e")
    ax.plot(time_s, nn_record["state_after"][:, yaw_rate_idx], label="NN yaw rate", color="#1d4ed8")
    ax.plot(time_s, chrono_record["state_after"][:, yaw_rate_idx], label="Chrono yaw rate", color="#dc2626")
    ax.set_title("Yaw Rate")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(time_s, nn_record["driver_action_cmd"][:, 0], label="NN steer", color="#1d4ed8")
    ax.plot(time_s, chrono_record["driver_action_cmd"][:, 0], label="Chrono steer", color="#dc2626")
    ax.plot(time_s, nn_record["driver_action_cmd"][:, 1], label="NN throttle", color="#15803d")
    ax.plot(time_s, chrono_record["driver_action_cmd"][:, 1], label="Chrono throttle", color="#b45309")
    ax.set_title("Closed-Loop Policy Commands")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[2, 1]
    obs_l2 = np.linalg.norm(chrono_record["obs_before"] - nn_record["obs_before"], axis=1)
    action_l2 = np.linalg.norm(chrono_record["driver_action_cmd"] - nn_record["driver_action_cmd"], axis=1)
    ax.plot(time_s, obs_l2, label="obs L2 diff", color="#7c3aed")
    ax.plot(time_s, action_l2, label="action L2 diff", color="#d97706")
    ax.set_title("Feedback Difference")
    ax.set_xlabel("time [s]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"NN vs Chrono Debug Rollout: {reference_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    train_cfg = load_json(run_dir / "train_cfg.json")
    nn_cfg, chrono_cfg = make_env_cfg(
        run_dir=run_dir,
        device=args.device,
        reference_index=args.reference_index,
        chrono_config=args.chrono_config,
        chrono_step_size_s=args.chrono_step_size_s,
    )
    chrono_cfg["warm_start_context"] = not bool(args.no_warm_start_context)

    refs = load_reference_set(nn_cfg["reference_path"])
    if args.reference_index < 0 or args.reference_index >= refs.num_references:
        raise ValueError(f"reference-index must be in [0, {refs.num_references - 1}]")
    reference_name = f"{refs.scenario_families[args.reference_index]}/{refs.episode_ids[args.reference_index]}"

    nn_env = HMMWVNeuralTrackingEnv(nn_cfg, device=args.device)
    policy = build_policy(nn_env, train_cfg, checkpoint_path=checkpoint_path, device=args.device)
    chrono_env = HMMWVChronoTrackingEnv(chrono_cfg, device=args.device)

    nn_record = rollout_debug("nn", nn_env, policy, args.reference_index, args.max_steps)
    chrono_record = rollout_debug("chrono", chrono_env, policy, args.reference_index, args.max_steps)
    comparison = compute_comparison(nn_record, chrono_record, nn_env)
    comparison.update(
        {
            "reference_index": int(args.reference_index),
            "reference_name": reference_name,
            "policy_checkpoint": str(checkpoint_path),
            "max_steps": int(args.max_steps),
            "nn_step_dt_s": float(nn_env.step_dt),
            "chrono_step_dt_s": float(chrono_env.step_dt),
            "chrono_solver_step_size_s": float(chrono_env.chrono_step_size_s),
            "chrono_steps_per_nn_step": int(chrono_env.chrono_steps_per_nn_step),
            "chrono_steps_per_policy_step": int(chrono_env.chrono_steps_per_policy_step),
            "chrono_warm_start_context": bool(chrono_env.warm_start_context),
        }
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (run_dir / f"debug_ref{args.reference_index:02d}_nn_vs_chrono_2s").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "nn_rollout.npz", **nn_record)
    np.savez_compressed(output_dir / "chrono_rollout.npz", **chrono_record)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    write_step_csv(output_dir / "steps.csv", nn_record, chrono_record, nn_env)
    plot_comparison(output_dir / "nn_vs_chrono_debug.png", nn_record, chrono_record, nn_env, reference_name)

    print(json.dumps(comparison, indent=2))
    print(f"Saved debug artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
