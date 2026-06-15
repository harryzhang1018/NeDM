#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.references import ReferenceSet, build_reference_set, summarize_reference_set
from nedm.training.dataset import load_metadata, load_split_metadata


DEFAULT_PROCESSED_ROOT = Path(
    "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
)
DEFAULT_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz"
)
DEFAULT_TRAIN_REFERENCE = Path(
    "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz"
)
PINNED_REPLACEMENTS = {
    # This was selected manually after inspecting the original near-static tracking_04 plot.
    "t300_s075_chirp_steer_00008": "t300_s052_chirp_steer_00025",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered rest-start HMMWV NN tracking eval references."
    )
    parser.add_argument("--processed-dataset-dir", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--dynamics-checkpoint", type=Path, default=DEFAULT_DYNAMICS_CHECKPOINT)
    parser.add_argument("--training-reference", type=Path, default=DEFAULT_TRAIN_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-references", type=int, default=20)
    parser.add_argument("--segment-nn-steps", type=int, default=1100)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--max-eval-steps", type=int, default=180)
    parser.add_argument("--min-eval-path-m", type=float, default=20.0)
    parser.add_argument("--target-replacement-path-m", type=float, default=40.0)
    return parser.parse_args()


def load_context_steps(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(checkpoint["config"]["model"]["block_size"])


def relative_poses(poses: np.ndarray) -> np.ndarray:
    origin_x = float(poses[0, 0])
    origin_y = float(poses[0, 1])
    origin_yaw = float(poses[0, 2])
    dx = poses[:, 0] - origin_x
    dy = poses[:, 1] - origin_y
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    relative = np.empty_like(poses, dtype=np.float32)
    relative[:, 0] = cos_yaw * dx + sin_yaw * dy
    relative[:, 1] = -sin_yaw * dx + cos_yaw * dy
    yaw_delta = poses[:, 2] - origin_yaw
    relative[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    return relative


def evaluated_path_m(poses: np.ndarray, sample_indices: np.ndarray) -> float:
    segment = poses[sample_indices, :2]
    return float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())


def zero_motion_fields(state: np.ndarray, state_index: dict[str, int], state_fields: list[str]) -> np.ndarray:
    zeroed = state.copy()
    zero_fields = [
        "vel_body_x_mps",
        "vel_body_y_mps",
        "roll_rate_radps",
        "ang_vel_body_y_radps",
        "yaw_rate_radps",
    ]
    zero_fields.extend(name for name in state_fields if name.endswith("_spindle_omega_radps"))
    for field in zero_fields:
        index = state_index.get(field)
        if index is not None:
            zeroed[index] = 0.0
    return zeroed


def reference_episode_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
    return set(metadata["episode_ids"])


def select_replacements(
    *,
    selected_indices: list[int],
    selected_paths: list[float],
    selected_families: list[str],
    selected_episode_ids: list[str],
    processed_root: Path,
    split: str,
    segment_nn_steps: int,
    min_eval_path_m: float,
    target_replacement_path_m: float,
    source_eval_indices: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]]]:
    split_metadata = load_split_metadata(processed_root, split)
    episode_lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
    rollout_offsets = split_metadata["rollout_episode_offsets"]
    rollout_mmap = np.load(processed_root / f"{split}_rollout.npy", mmap_mode="r")
    used_indices = set(selected_indices)
    used_ids = set(selected_episode_ids)
    replacements: list[dict[str, Any]] = []

    def path_for_episode(episode_index: int) -> float:
        rollout_start = int(rollout_offsets[episode_index])
        count = int(source_eval_indices.max()) + 1
        poses = np.array(rollout_mmap[rollout_start : rollout_start + count], dtype=np.float32, copy=True)
        return evaluated_path_m(relative_poses(poses), source_eval_indices)

    for position, path in enumerate(selected_paths):
        if path >= min_eval_path_m:
            continue
        family = selected_families[position]
        pinned_episode_id = PINNED_REPLACEMENTS.get(selected_episode_ids[position])
        if pinned_episode_id is not None and pinned_episode_id not in used_ids:
            try:
                new_episode_index = split_metadata["episode_ids"].index(pinned_episode_id)
            except ValueError:
                new_episode_index = -1
            if new_episode_index >= 0 and split_metadata["scenario_families"][new_episode_index] == family:
                new_path = path_for_episode(new_episode_index)
                if new_path >= min_eval_path_m:
                    old_episode_id = selected_episode_ids[position]
                    selected_indices[position] = int(new_episode_index)
                    used_indices.add(int(new_episode_index))
                    used_ids.add(pinned_episode_id)
                    replacements.append(
                        {
                            "position": int(position),
                            "family": family,
                            "old_episode_id": old_episode_id,
                            "old_eval_path_m": float(path),
                            "new_episode_id": pinned_episode_id,
                            "new_episode_index": int(new_episode_index),
                            "new_eval_path_m": float(new_path),
                            "reason": f"evaluated reference path below {min_eval_path_m:g} m",
                            "selection": "pinned_inspected_replacement",
                        }
                    )
                    continue
        candidates: list[tuple[float, float, int, str]] = []
        for episode_index, (episode_id, episode_family) in enumerate(
            zip(split_metadata["episode_ids"], split_metadata["scenario_families"], strict=True)
        ):
            if episode_family != family:
                continue
            if int(episode_lengths[episode_index]) < segment_nn_steps:
                continue
            if episode_index in used_indices or episode_id in used_ids:
                continue
            candidate_path = path_for_episode(episode_index)
            if candidate_path < min_eval_path_m:
                continue
            candidates.append(
                (abs(candidate_path - target_replacement_path_m), candidate_path, episode_index, episode_id)
            )
        if not candidates:
            raise RuntimeError(f"No replacement found for position {position} family {family}")
        _, new_path, new_episode_index, new_episode_id = sorted(candidates)[0]
        old_episode_id = selected_episode_ids[position]
        selected_indices[position] = int(new_episode_index)
        used_indices.add(int(new_episode_index))
        used_ids.add(new_episode_id)
        replacements.append(
            {
                "position": int(position),
                "family": family,
                "old_episode_id": old_episode_id,
                "old_eval_path_m": float(path),
                "new_episode_id": new_episode_id,
                "new_episode_index": int(new_episode_index),
                "new_eval_path_m": float(new_path),
                "reason": f"evaluated reference path below {min_eval_path_m:g} m",
            }
        )
    return selected_indices, replacements


def build_from_episode_indices(
    *,
    processed_root: Path,
    split: str,
    selected_indices: list[int],
    total_steps: int,
    source_len: int,
    pad_steps: int,
) -> ReferenceSet:
    metadata = load_metadata(processed_root)
    split_metadata = load_split_metadata(processed_root, split)
    states_mmap = np.load(processed_root / f"{split}_states.npy", mmap_mode="r")
    actions_mmap = np.load(processed_root / f"{split}_actions.npy", mmap_mode="r")
    targets_mmap = np.load(processed_root / f"{split}_targets.npy", mmap_mode="r")
    rollout_mmap = np.load(processed_root / f"{split}_rollout.npy", mmap_mode="r")
    episode_starts = np.load(processed_root / f"{split}_episode_starts.npy")
    rollout_offsets = split_metadata["rollout_episode_offsets"]
    state_fields = list(metadata["state_fields"])
    action_fields = list(metadata["action_fields"])
    rollout_fields = list(metadata["rollout_fields"])
    state_index = {name: index for index, name in enumerate(state_fields)}

    state_segments: list[np.ndarray] = []
    action_segments: list[np.ndarray] = []
    pose_segments: list[np.ndarray] = []
    episode_ids: list[str] = []
    families: list[str] = []
    segment_records: list[dict[str, Any]] = []

    for episode_index in selected_indices:
        transition_start = int(episode_starts[episode_index])
        transition_stop = transition_start + source_len - 1
        rollout_start = int(rollout_offsets[episode_index])
        rollout_stop = rollout_start + source_len

        source_states = np.empty((source_len, states_mmap.shape[1]), dtype=np.float32)
        source_states[:-1] = np.array(states_mmap[transition_start:transition_stop], dtype=np.float32, copy=True)
        source_states[-1] = source_states[-2] + np.array(
            targets_mmap[transition_stop - 1], dtype=np.float32, copy=True
        )

        source_actions = np.empty((source_len, actions_mmap.shape[1]), dtype=np.float32)
        source_actions[:-1] = np.array(actions_mmap[transition_start:transition_stop], dtype=np.float32, copy=True)
        source_actions[-1] = source_actions[-2]

        source_poses = relative_poses(np.array(rollout_mmap[rollout_start:rollout_stop], dtype=np.float32, copy=True))
        initial_state = zero_motion_fields(source_states[0], state_index, state_fields)
        source_states[0] = initial_state
        source_actions[0] = 0.0

        states = np.empty((total_steps, source_states.shape[1]), dtype=np.float32)
        actions = np.empty((total_steps, source_actions.shape[1]), dtype=np.float32)
        poses = np.empty((total_steps, source_poses.shape[1]), dtype=np.float32)
        states[:pad_steps] = initial_state
        states[pad_steps:] = source_states
        actions[:pad_steps] = 0.0
        actions[pad_steps:] = source_actions
        poses[:pad_steps] = 0.0
        poses[pad_steps:] = source_poses

        episode_id = split_metadata["episode_ids"][episode_index]
        family = split_metadata["scenario_families"][episode_index]
        net_xy = source_poses[-1, :2] - source_poses[0, :2]
        net_yaw = float(
            np.arctan2(
                np.sin(source_poses[-1, 2] - source_poses[0, 2]),
                np.cos(source_poses[-1, 2] - source_poses[0, 2]),
            )
        )
        segment_records.append(
            {
                "episode_index": int(episode_index),
                "episode_id": episode_id,
                "scenario_family": family,
                "local_start": 0,
                "segment_nn_steps": int(total_steps - 1),
                "duration_s": float((total_steps - 1) * float(metadata["dt_s"])),
                "net_dx_m": float(net_xy[0]),
                "net_dy_m": float(net_xy[1]),
                "net_yaw_rad": net_yaw,
                "mean_speed_mps": float(np.mean(source_states[:, state_index["vel_body_x_mps"]])),
            }
        )
        state_segments.append(states)
        action_segments.append(actions)
        pose_segments.append(poses)
        episode_ids.append(episode_id)
        families.append(family)

    return ReferenceSet(
        states=np.stack(state_segments, axis=0),
        actions=np.stack(action_segments, axis=0),
        poses=np.stack(pose_segments, axis=0),
        episode_ids=episode_ids,
        scenario_families=families,
        dt_s=float(metadata["dt_s"]),
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
        metadata={
            "source_processed_root": str(processed_root),
            "source_split": split,
            "segments": segment_records,
        },
    )


def main() -> int:
    args = parse_args()
    processed_root = args.processed_dataset_dir.expanduser().resolve()
    checkpoint_path = args.dynamics_checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    context_steps = load_context_steps(checkpoint_path)
    pad_steps = context_steps - 1
    total_steps = args.segment_nn_steps + 1
    source_len = total_steps - pad_steps
    source_eval_indices = args.action_repeat * np.arange(1, args.max_eval_steps + 1)
    handoff_eval_indices = pad_steps + source_eval_indices

    base = build_reference_set(
        processed_root=processed_root,
        split=args.split,
        num_references=args.num_references,
        segment_nn_steps=args.segment_nn_steps,
        seed=args.seed,
        random_segment_start=False,
    )
    selected_indices = [int(segment["episode_index"]) for segment in base.metadata["segments"]]
    selected_paths = [
        evaluated_path_m(relative_poses(base.poses[index, :source_len]), source_eval_indices)
        for index in range(base.num_references)
    ]
    selected_indices, replacements = select_replacements(
        selected_indices=selected_indices,
        selected_paths=selected_paths,
        selected_families=base.scenario_families,
        selected_episode_ids=base.episode_ids,
        processed_root=processed_root,
        split=args.split,
        segment_nn_steps=args.segment_nn_steps,
        min_eval_path_m=args.min_eval_path_m,
        target_replacement_path_m=args.target_replacement_path_m,
        source_eval_indices=source_eval_indices,
    )

    reference_set = build_from_episode_indices(
        processed_root=processed_root,
        split=args.split,
        selected_indices=selected_indices,
        total_steps=total_steps,
        source_len=source_len,
        pad_steps=pad_steps,
    )

    train_episode_ids = reference_episode_ids(args.training_reference)
    overlap = sorted(set(reference_set.episode_ids) & train_episode_ids)
    if overlap:
        raise RuntimeError(f"Validation references overlap training reference set: {overlap}")

    eval_paths = [
        evaluated_path_m(reference_set.poses[index], handoff_eval_indices)
        for index in range(reference_set.num_references)
    ]
    metadata = dict(reference_set.metadata)
    metadata.update(
        {
            "episode_ids": reference_set.episode_ids,
            "scenario_families": reference_set.scenario_families,
            "dt_s": reference_set.dt_s,
            "state_fields": reference_set.state_fields,
            "action_fields": reference_set.action_fields,
            "rollout_fields": reference_set.rollout_fields,
            "num_references": reference_set.num_references,
            "num_steps": reference_set.num_steps,
            "seed": int(args.seed),
            "random_segment_start": False,
            "rest_start": True,
            "rest_context_pad_steps": int(pad_steps),
            "rest_handoff_reference_index": int(pad_steps),
            "source_zero_index_aligned_to_reference_index": int(pad_steps),
            "pose_frame": "first_pose_relative",
            "min_eval_path_filter_m": float(args.min_eval_path_m),
            "target_replacement_path_m": float(args.target_replacement_path_m),
            "filtered_replacements": replacements,
            "eval_path_m": eval_paths,
            "training_reference_overlap": overlap,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=reference_set.states.astype(np.float32, copy=False),
        actions=reference_set.actions.astype(np.float32, copy=False),
        poses=reference_set.poses.astype(np.float32, copy=False),
        metadata_json=np.array(json.dumps(metadata, indent=2), dtype=np.str_),
    )

    summary = summarize_reference_set(reference_set)
    summary.update(
        {
            "output": str(output_path),
            "rest_handoff_reference_index": int(pad_steps),
            "training_reference_overlap_count": len(overlap),
            "replacements": replacements,
            "min_eval_path_m": float(min(eval_paths)),
            "median_eval_path_m": float(np.median(eval_paths)),
            "max_eval_path_m": float(max(eval_paths)),
        }
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
