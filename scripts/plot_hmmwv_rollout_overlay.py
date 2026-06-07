from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.dataset import load_rollout_split
from nedm.training.trainer import HMMWVTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one HMMWV rollout against ground truth.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Validation episode id to plot. If omitted, the first matching episode is used.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Optional family filter when episode-id is not provided.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to draw the episode from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults next to the checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for plotting/eval. Use 'cpu' for portability or 'cuda' on the training machine.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=1,
        help="Number of random episodes to sample and plot. Ignored when --episode-id is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260405,
        help="Random seed used when sampling multiple episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for multi-episode output. Defaults under the checkpoint run directory.",
    )
    return parser.parse_args(argv)


def select_episode(episodes: list[dict], episode_id: str | None, family: str | None) -> dict:
    if episode_id is not None:
        for episode in episodes:
            if episode["episode_id"] == episode_id:
                return episode
        raise ValueError(f"Episode id {episode_id!r} not found in selected split")

    if family is not None:
        for episode in episodes:
            if episode["scenario_family"] == family:
                return episode
        raise ValueError(f"No episode found for family {family!r} in selected split")

    return episodes[0]


def select_random_episodes(
    episodes: list[dict],
    num_random: int,
    seed: int,
    family: str | None,
) -> list[dict]:
    candidate_episodes = episodes
    if family is not None:
        candidate_episodes = [episode for episode in episodes if episode["scenario_family"] == family]
        if not candidate_episodes:
            raise ValueError(f"No episode found for family {family!r} in selected split")

    num_random = max(1, min(int(num_random), len(candidate_episodes)))
    rng = random.Random(seed)
    selected = rng.sample(candidate_episodes, k=num_random)
    selected.sort(key=lambda episode: (episode["scenario_family"], episode["episode_id"]))
    return selected


def run_episode_rollout(trainer: HMMWVTrainer, episode: dict) -> dict:
    horizon_steps = int(episode["states"].shape[0] - trainer.sequence_length)
    result = trainer._rollout_episode(episode, horizon_steps=horizon_steps)
    if result is None:
        raise ValueError(f"Episode {episode['episode_id']} is too short for sequence_length={trainer.sequence_length}")
    predicted_states, predicted_pose, gt_states, gt_pose = result

    steps = predicted_states.shape[0]
    action_slice = episode["actions"][trainer.sequence_length : trainer.sequence_length + steps]
    rollout_time_s = np.arange(steps, dtype=np.float32) * trainer.dt_s

    gt_states_np = gt_states.detach().cpu().numpy()
    pred_states_np = predicted_states.detach().cpu().numpy()
    gt_pose_np = gt_pose.detach().cpu().numpy()
    pred_pose_np = predicted_pose.detach().cpu().numpy()
    action_np = np.asarray(action_slice[:steps], dtype=np.float32)

    metrics = {}
    for field_index, field_name in enumerate(trainer.metadata["state_fields"]):
        rmse = math.sqrt(np.mean((pred_states_np[:, field_index] - gt_states_np[:, field_index]) ** 2))
        metrics[field_name] = rmse
    metrics["xy_rmse_m"] = math.sqrt(np.mean(np.sum((pred_pose_np[:, :2] - gt_pose_np[:, :2]) ** 2, axis=1)))
    yaw_diff = np.arctan2(np.sin(pred_pose_np[:, 2] - gt_pose_np[:, 2]), np.cos(pred_pose_np[:, 2] - gt_pose_np[:, 2]))
    metrics["yaw_rmse_rad"] = math.sqrt(np.mean(yaw_diff**2))

    return {
        "time_s": rollout_time_s,
        "actions": action_np,
        "gt_states": gt_states_np,
        "pred_states": pred_states_np,
        "gt_pose": gt_pose_np,
        "pred_pose": pred_pose_np,
        "metrics": metrics,
        "steps": steps,
    }


def plot_rollout(
    trainer: HMMWVTrainer,
    episode: dict,
    rollout_data: dict,
    output_path: Path,
) -> None:
    state_index = trainer.state_index
    time_s = rollout_data["time_s"]
    gt_states = rollout_data["gt_states"]
    pred_states = rollout_data["pred_states"]
    gt_pose = rollout_data["gt_pose"]
    pred_pose = rollout_data["pred_pose"]
    actions = rollout_data["actions"]

    fig, axes = plt.subplots(4, 2, figsize=(15, 14), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(gt_pose[:, 0], gt_pose[:, 1], label="ground truth", linewidth=2.0, color="#0f766e")
    ax.plot(pred_pose[:, 0], pred_pose[:, 1], label="model rollout", linewidth=1.6, color="#dc2626", alpha=0.9)
    ax.set_title("XY Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    series_specs = [
        ("vel_body_x_mps", "Body vx [m/s]", axes[0, 1]),
        ("vel_body_y_mps", "Body vy [m/s]", axes[1, 0]),
        ("yaw_rate_radps", "Yaw Rate [rad/s]", axes[1, 1]),
        ("roll_rad", "Roll [rad]", axes[2, 0]),
        ("pitch_rad", "Pitch [rad]", axes[2, 1]),
    ]
    for field_name, title, ax in series_specs:
        idx = state_index[field_name]
        ax.plot(time_s, gt_states[:, idx], label="ground truth", linewidth=2.0, color="#0f766e")
        ax.plot(time_s, pred_states[:, idx], label="model rollout", linewidth=1.4, color="#dc2626", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("rollout time [s]")
        ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(time_s, actions[:, 0], label="steering", linewidth=1.8, color="#1d4ed8")
    ax.plot(time_s, actions[:, 1], label="throttle", linewidth=1.8, color="#15803d")
    ax.plot(time_s, actions[:, 2], label="braking", linewidth=1.8, color="#b45309")
    ax.set_title("Command Sequence")
    ax.set_xlabel("rollout time [s]")
    ax.set_ylabel("command")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[3, 1]
    metric_lines = [
        f"episode: {episode['episode_id']}",
        f"family: {episode['scenario_family']}",
        f"context: {trainer.sequence_length} steps ({trainer.sequence_length * trainer.dt_s:.2f} s)",
        f"rollout: {rollout_data['steps']} steps ({rollout_data['steps'] * trainer.dt_s:.2f} s)",
        "",
        "RMSE",
    ]
    for key in ["vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps", "roll_rad", "pitch_rad"]:
        metric_lines.append(f"{key}: {rollout_data['metrics'][key]:.6f}")
    metric_lines.append(f"xy_rmse_m: {rollout_data['metrics']['xy_rmse_m']:.6f}")
    metric_lines.append(f"yaw_rmse_rad: {rollout_data['metrics']['yaw_rmse_rad']:.6f}")
    ax.axis("off")
    ax.text(0.0, 1.0, "\n".join(metric_lines), va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("HMMWV Single-Episode Rollout Overlay", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    checkpoint["config"]["training"]["device"] = args.device
    trainer = HMMWVTrainer(checkpoint["config"])
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(trainer.device)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.model.eval()

    split_data = load_rollout_split(trainer.processed_root, args.split)
    if args.episode_id is not None:
        selected_episodes = [select_episode(split_data["episodes"], args.episode_id, args.family)]
    else:
        selected_episodes = select_random_episodes(
            split_data["episodes"],
            num_random=args.num_random,
            seed=args.seed,
            family=args.family,
        )

    if args.output is not None and len(selected_episodes) > 1:
        raise ValueError("--output can only be used when plotting a single episode")

    default_root = args.checkpoint.resolve().parents[1] / "plots"
    if len(selected_episodes) == 1:
        output_dir = args.output_dir.resolve() if args.output_dir is not None else default_root.resolve()
    else:
        if args.output_dir is not None:
            output_dir = args.output_dir.resolve()
        else:
            family_suffix = args.family if args.family is not None else "mixed"
            output_dir = (default_root / f"random_{family_suffix}_{args.split}_seed_{args.seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result_records = []
    for episode in selected_episodes:
        with torch.inference_mode():
            rollout_data = run_episode_rollout(trainer, episode)

        if args.output is not None:
            output_path = args.output.resolve()
        else:
            output_path = output_dir / f"{episode['episode_id']}_overlay.png"

        plot_rollout(trainer, episode, rollout_data, output_path)

        metrics_payload = {
            "episode_id": episode["episode_id"],
            "scenario_family": episode["scenario_family"],
            "split": args.split,
            "context_steps": trainer.sequence_length,
            "dt_s": trainer.dt_s,
            "rollout_steps": rollout_data["steps"],
            "metrics": rollout_data["metrics"],
            "plot_path": str(output_path),
        }
        output_path.with_suffix(".json").write_text(json.dumps(metrics_payload, indent=2))
        result_records.append(metrics_payload)

    if len(result_records) > 1:
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "split": args.split,
            "seed": args.seed,
            "family": args.family,
            "num_episodes": len(result_records),
            "episodes": result_records,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result_records[0], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
