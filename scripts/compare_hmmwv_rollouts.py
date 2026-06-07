from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.dataset import load_rollout_split
from nedm.training.trainer import HMMWVTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two trained HMMWV models on the same rollout episodes.")
    parser.add_argument("--checkpoint-a", type=Path, required=True, help="Baseline checkpoint.")
    parser.add_argument("--checkpoint-b", type=Path, required=True, help="Comparison checkpoint.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args(argv)


def load_trainer(checkpoint_path: Path, device: str) -> tuple[HMMWVTrainer, dict]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu")
    checkpoint["config"]["training"]["device"] = device
    trainer = HMMWVTrainer(checkpoint["config"])
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(trainer.device)
    trainer.model.eval()
    return trainer, checkpoint


def wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def eval_episode(trainer: HMMWVTrainer, episode: dict) -> dict[str, float]:
    horizon_steps = int(episode["states"].shape[0] - trainer.sequence_length)
    with torch.inference_mode():
        result = trainer._rollout_episode(episode, horizon_steps=horizon_steps)
    if result is None:
        raise ValueError(f"Episode {episode['episode_id']} is too short for sequence_length={trainer.sequence_length}")
    predicted_states, predicted_pose, gt_states, gt_pose = result

    predicted_states_np = predicted_states.detach().cpu().numpy()
    predicted_pose_np = predicted_pose.detach().cpu().numpy()
    gt_states_np = gt_states.detach().cpu().numpy()
    gt_pose_np = gt_pose.detach().cpu().numpy()

    metrics: dict[str, float] = {}
    for field_index, field_name in enumerate(trainer.metadata["state_fields"]):
        metrics[field_name] = float(
            np.sqrt(np.mean((predicted_states_np[:, field_index] - gt_states_np[:, field_index]) ** 2))
        )

    metrics["xy_rmse_m"] = float(
        np.sqrt(np.mean(np.sum((predicted_pose_np[:, :2] - gt_pose_np[:, :2]) ** 2, axis=1)))
    )
    yaw_diff = wrap_angle_np(predicted_pose_np[:, 2] - gt_pose_np[:, 2])
    metrics["yaw_rmse_rad"] = float(np.sqrt(np.mean(yaw_diff**2)))
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trainer_a, checkpoint_a = load_trainer(args.checkpoint_a, args.device)
    trainer_b, checkpoint_b = load_trainer(args.checkpoint_b, args.device)

    episodes = load_rollout_split(trainer_a.processed_root, args.split)["episodes"]
    rng = random.Random(args.seed)
    selected = rng.sample(episodes, k=min(args.num_episodes, len(episodes)))
    selected.sort(key=lambda episode: (episode["scenario_family"], episode["episode_id"]))

    records = []
    metric_keys = ["xy_rmse_m", "yaw_rmse_rad", "vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps"]

    for episode in selected:
        metrics_a = eval_episode(trainer_a, episode)
        metrics_b = eval_episode(trainer_b, episode)
        records.append(
            {
                "episode_id": episode["episode_id"],
                "scenario_family": episode["scenario_family"],
                "model_a": metrics_a,
                "model_b": metrics_b,
                "delta_b_minus_a": {key: metrics_b[key] - metrics_a[key] for key in metrics_a},
                "model_b_improved": {key: metrics_b[key] < metrics_a[key] for key in metrics_a},
            }
        )

    aggregate = {}
    for key in metric_keys:
        avg_a = sum(record["model_a"][key] for record in records) / len(records)
        avg_b = sum(record["model_b"][key] for record in records) / len(records)
        aggregate[key] = {
            "model_a_avg": avg_a,
            "model_b_avg": avg_b,
            "delta_b_minus_a": avg_b - avg_a,
            "relative_change_pct": 100.0 * (avg_b - avg_a) / avg_a if avg_a != 0.0 else None,
            "model_b_improved_episodes": sum(1 for record in records if record["model_b"][key] < record["model_a"][key]),
            "total_episodes": len(records),
        }

    summary = {
        "split": args.split,
        "seed": args.seed,
        "num_episodes": len(records),
        "episode_ids": [record["episode_id"] for record in records],
        "checkpoint_a": str(args.checkpoint_a.resolve()),
        "checkpoint_b": str(args.checkpoint_b.resolve()),
        "checkpoint_a_epoch": int(checkpoint_a["epoch"]),
        "checkpoint_b_epoch": int(checkpoint_b["epoch"]),
        "aggregate": aggregate,
        "episodes": records,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
