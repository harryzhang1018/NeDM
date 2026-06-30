#!/usr/bin/env python3
"""Evaluate an arm reaching RL policy in the Chrono M113+arm scene."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pychrono as chrono  # noqa: F401  # load Chrono before torch/libstdc++ users
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.arm_reaching_chrono_env import ArmReachingChronoEnv
from nedm.rl.defaults import (
    DEFAULT_ARM_DYNAMICS_CHECKPOINT,
    DEFAULT_ARM_GEOMETRY_PATH,
    DEFAULT_ARM_PROCESSED_DATASET_DIR,
)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def latest_policy_checkpoint(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt policy checkpoints found under {run_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_runner_checkpoint(runner: OnPolicyRunner, checkpoint_path: Path, device: str) -> None:
    loaded_dict = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    if runner.alg.rnd:
        runner.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = int(loaded_dict["iter"])


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def rollout_one_goal(
    env: ArmReachingChronoEnv,
    policy,
    max_steps: int,
    ignore_dones: bool = False,
    reset_scene: bool = True,
) -> dict[str, Any]:
    env_id = torch.tensor([0], dtype=torch.long, device=env.device)
    if reset_scene:
        env.reset_idx(env_id)
    else:
        env.reset_goal_idx(env_id)
    obs, _ = env.get_observations()

    ee_values = [tensor_to_numpy(env.current_ee_base()[0])]
    q_values = [tensor_to_numpy(env.current_q()[0])]
    qcmd_values = [tensor_to_numpy(env.current_qcmd()[0])]
    goal = tensor_to_numpy(env.goal_base[0])
    errors = [float(np.linalg.norm(goal - ee_values[-1]))]
    actions = []
    rewards = []
    dones = []
    clearances = []
    unsafe_actions = []

    with torch.no_grad():
        for _ in range(max_steps):
            action = policy(obs.to(env.device))
            obs, reward, done, _ = env.step(action)
            ee_values.append(tensor_to_numpy(env.current_ee_base()[0]))
            q_values.append(tensor_to_numpy(env.current_q()[0]))
            qcmd_values.append(tensor_to_numpy(env.current_qcmd()[0]))
            errors.append(float(np.linalg.norm(goal - ee_values[-1])))
            actions.append(tensor_to_numpy(env.actions[0]))
            rewards.append(float(reward[0].item()))
            dones.append(bool(done[0].item()))
            clearances.append(float(env.clearance_buf[0].item()))
            unsafe_actions.append(bool(env.unsafe_action_buf[0].item()))
            if bool(done[0].item()) and not ignore_dones:
                break

    return {
        "ee_base": np.stack(ee_values, axis=0).astype(np.float32),
        "q": np.stack(q_values, axis=0).astype(np.float32),
        "qcmd": np.stack(qcmd_values, axis=0).astype(np.float32),
        "goal_base": goal.astype(np.float32),
        "error_m": np.asarray(errors, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.bool_),
        "clearance_m": np.asarray(clearances, dtype=np.float32),
        "unsafe_action": np.asarray(unsafe_actions, dtype=np.bool_),
        "done_reason": done_reason(env),
        "contact_kind": env.contact_kinds[0],
        "contact_links": list(env.contact_links[0]),
        "joint_limit_labels": list(env.joint_limit_labels[0]),
    }


def done_reason(env: ArmReachingChronoEnv) -> str | None:
    success_steps = int(env.cfg["reward"]["success_steps"])
    if int(env.success_count_buf[0].item()) >= success_steps:
        return "success"
    if bool(env.contact_buf[0].item()):
        kind = env.contact_kinds[0] or "unknown"
        return f"collision:{kind}"
    if bool(env.joint_limit_buf[0].item()):
        return "joint_limit"
    if bool(env.nonfinite_buf[0].item()):
        return "nonfinite"
    if bool(env.time_out_buf[0].item()):
        return "timeout"
    return None


def can_continue_consecutive(done_reason_value: str | None) -> bool:
    return done_reason_value in {None, "success", "timeout"}


def compute_metrics(record: dict[str, Any], success_tolerance_m: float) -> dict[str, Any]:
    errors = record["error_m"]
    rewards = record["reward"]
    steps = int(rewards.shape[0])
    return {
        "steps": steps,
        "done_reason": record["done_reason"],
        "success": bool(record["done_reason"] == "success"),
        "reached_tolerance": bool(np.min(errors) < success_tolerance_m),
        "final_ee_error_m": float(errors[-1]),
        "min_ee_error_m": float(np.min(errors)),
        "mean_ee_error_m": float(np.mean(errors)),
        "reward_sum": float(np.sum(rewards)) if rewards.size else 0.0,
        "unsafe_action_rate": float(np.mean(record["unsafe_action"])) if record["unsafe_action"].size else 0.0,
        "min_clearance_m": float(np.min(record["clearance_m"])) if record["clearance_m"].size else None,
        "goal_base": [float(v) for v in record["goal_base"]],
        "contact_kind": record["contact_kind"],
        "contact_links": record["contact_links"],
        "joint_limit_labels": record["joint_limit_labels"],
    }


def plot_record(record: dict[str, Any], output_path: Path, success_tolerance_m: float, dt_s: float) -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = Path("/tmp/nedm_mplconfig")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ee = record["ee_base"]
    goal = record["goal_base"]
    error = record["error_m"]
    t = np.arange(error.shape[0], dtype=np.float32) * dt_s

    fig = plt.figure(figsize=(11, 4.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="tab:blue", linewidth=1.8)
    ax3d.scatter(ee[0, 0], ee[0, 1], ee[0, 2], color="tab:green", s=35, label="start")
    ax3d.scatter(ee[-1, 0], ee[-1, 1], ee[-1, 2], color="tab:orange", s=35, label="final")
    ax3d.scatter(goal[0], goal[1], goal[2], color="tab:red", s=45, label="goal")
    ax3d.set_xlabel("x base (m)")
    ax3d.set_ylabel("y base (m)")
    ax3d.set_zlabel("z base (m)")
    ax3d.legend(loc="best")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(t, error, color="tab:blue", linewidth=1.8)
    ax.axhline(success_tolerance_m, color="tab:red", linestyle="--", linewidth=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("EE error (m)")
    ax.grid(True, alpha=0.3)
    ax.set_title(record["done_reason"] or "not done")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an arm reach PPO policy in Chrono.")
    parser.add_argument("--run-dir", type=Path, required=True, help="RSL-RL run directory with env/train cfg.")
    parser.add_argument("--policy-checkpoint", type=Path, default=None, help="Defaults to latest model_*.pt in run-dir.")
    parser.add_argument("--device", type=str, default="auto", help="Policy/obs tensor device; Chrono remains CPU-bound.")
    parser.add_argument("--num-goals", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--pre-roll-time-s",
        type=float,
        default=None,
        help="Seconds to settle the Chrono vehicle/arm scene before policy eval starts. Defaults to 6s.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=1, help="Chrono sims are serial; keep this small.")
    parser.add_argument("--dynamics-checkpoint", type=Path, default=None)
    parser.add_argument("--processed-dataset-dir", type=Path, default=None)
    parser.add_argument("--geometry-path", type=Path, default=None)
    parser.add_argument("--success-tolerance-m", type=float, default=None)
    parser.add_argument("--ignore-dones", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--render", action="store_true", help="Open the Chrono Irrlicht viewer during rollout.")
    parser.add_argument(
        "--consecutive-goals",
        action="store_true",
        help="Use one Chrono scene and reset only the goal after each completed attempt.",
    )
    parser.add_argument("--no-warm-start-context", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.num_envs) != 1:
        raise ValueError(
            "scripts/eval_arm_rl_chrono_reaching.py rolls out one Chrono env at a time; "
            "use --num-envs 1 or instantiate ArmReachingChronoEnv directly for serial vector batches."
        )
    run_dir = args.run_dir.resolve()
    device = resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    env_cfg_path = run_dir / "env_cfg.json"
    train_cfg_path = run_dir / "train_cfg.json"
    if env_cfg_path.is_file():
        env_cfg = json.loads(env_cfg_path.read_text())
    else:
        env_cfg = {
            "dynamics_checkpoint": str(DEFAULT_ARM_DYNAMICS_CHECKPOINT),
            "processed_dataset_dir": str(DEFAULT_ARM_PROCESSED_DATASET_DIR),
            "geometry_path": str(DEFAULT_ARM_GEOMETRY_PATH),
        }
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Missing train_cfg.json in {run_dir}")
    train_cfg = json.loads(train_cfg_path.read_text())

    env_cfg.update(
        {
            "num_envs": int(args.num_envs),
            "device": device,
            "auto_reset": False,
            "warm_start_context": not bool(args.no_warm_start_context),
            "defer_reset": True,
            "render": bool(args.render),
        }
    )
    if args.pre_roll_time_s is not None:
        env_cfg["pre_roll_time_s"] = float(args.pre_roll_time_s)
    if args.max_steps is not None:
        env_cfg["max_episode_steps"] = int(args.max_steps)
    if args.dynamics_checkpoint is not None:
        env_cfg["dynamics_checkpoint"] = str(args.dynamics_checkpoint)
    if args.processed_dataset_dir is not None:
        env_cfg["processed_dataset_dir"] = str(args.processed_dataset_dir)
    if args.geometry_path is not None:
        env_cfg["geometry_path"] = str(args.geometry_path)
    if args.success_tolerance_m is not None:
        env_cfg.setdefault("reward", {})["success_tolerance_m"] = float(args.success_tolerance_m)

    checkpoint_path = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "chrono_eval_reaching").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = ArmReachingChronoEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    load_runner_checkpoint(runner, checkpoint_path, device=device)
    policy = runner.get_inference_policy(device=device)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_length)
    success_tolerance_m = float(env.cfg["reward"]["success_tolerance_m"])

    summary = []
    for rollout_index in range(int(args.num_goals)):
        record = rollout_one_goal(
            env,
            policy,
            max_steps=max_steps,
            ignore_dones=bool(args.ignore_dones),
            reset_scene=(rollout_index == 0 or not bool(args.consecutive_goals)),
        )
        metrics = compute_metrics(record, success_tolerance_m=success_tolerance_m)
        metrics["rollout_index"] = rollout_index
        summary.append(metrics)
        np.savez_compressed(
            output_dir / f"chrono_arm_reach_{rollout_index:02d}.npz",
            ee_base=record["ee_base"],
            q=record["q"],
            qcmd=record["qcmd"],
            goal_base=record["goal_base"],
            error_m=record["error_m"],
            action=record["action"],
            reward=record["reward"],
            done=record["done"],
            clearance_m=record["clearance_m"],
            unsafe_action=record["unsafe_action"],
        )
        if not args.no_plots:
            plot_record(
                record,
                output_dir / f"chrono_arm_reach_{rollout_index:02d}.png",
                success_tolerance_m=success_tolerance_m,
                dt_s=env.dt_s * env.action_repeat,
        )
        print(json.dumps(metrics))
        if bool(args.consecutive_goals) and not can_continue_consecutive(metrics["done_reason"]):
            print(
                json.dumps(
                    {
                        "stopped_consecutive_rollout": True,
                        "rollout_index": rollout_index,
                        "done_reason": metrics["done_reason"],
                    }
                )
            )
            break

    aggregate = {
        "backend": "chrono_arm_reaching",
        "policy_checkpoint": str(checkpoint_path),
        "pre_roll_time_s": float(env.cfg.get("pre_roll_time_s", 0.0)),
        "consecutive_goals": bool(args.consecutive_goals),
        "requested_goals": int(args.num_goals),
        "num_rollouts": len(summary),
        "success_rate": float(np.mean([row["success"] for row in summary])) if summary else None,
        "reached_tolerance_rate": float(np.mean([row["reached_tolerance"] for row in summary])) if summary else None,
        "mean_final_ee_error_m": float(np.mean([row["final_ee_error_m"] for row in summary])) if summary else None,
        "mean_min_ee_error_m": float(np.mean([row["min_ee_error_m"] for row in summary])) if summary else None,
        "rollouts": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    (output_dir / "env_cfg.json").write_text(json.dumps(env.cfg, indent=2))
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
