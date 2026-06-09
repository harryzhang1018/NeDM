from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nedm.training.dataset import load_metadata, load_split_metadata


DEFAULT_REFERENCE_FAMILIES = [
    "launch_brake",
    "step_steer",
    "sustained_turn",
    "sine_steer",
    "doublet_steer",
    "multi_steer",
    "chirp_steer",
    "steer_brake",
    "aggressive_step_steer",
    "aggressive_sine_steer",
    "aggressive_doublet_steer",
    "aggressive_chirp_steer",
    "aggressive_steer_brake",
]


@dataclass(frozen=True)
class ReferenceSet:
    states: np.ndarray
    actions: np.ndarray
    poses: np.ndarray
    episode_ids: list[str]
    scenario_families: list[str]
    dt_s: float
    state_fields: list[str]
    action_fields: list[str]
    rollout_fields: list[str]
    metadata: dict[str, Any]

    @property
    def num_references(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.states.shape[1])


def wrap_angle_np(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def load_reference_set(path: str | Path) -> ReferenceSet:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return ReferenceSet(
            states=np.array(data["states"], dtype=np.float32, copy=True),
            actions=np.array(data["actions"], dtype=np.float32, copy=True),
            poses=np.array(data["poses"], dtype=np.float32, copy=True),
            episode_ids=list(metadata["episode_ids"]),
            scenario_families=list(metadata["scenario_families"]),
            dt_s=float(metadata["dt_s"]),
            state_fields=list(metadata["state_fields"]),
            action_fields=list(metadata["action_fields"]),
            rollout_fields=list(metadata["rollout_fields"]),
            metadata=metadata,
        )


def save_reference_set(reference_set: ReferenceSet, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
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
        }
    )
    np.savez_compressed(
        path,
        states=reference_set.states.astype(np.float32, copy=False),
        actions=reference_set.actions.astype(np.float32, copy=False),
        poses=reference_set.poses.astype(np.float32, copy=False),
        metadata_json=np.array(json.dumps(metadata, indent=2), dtype=np.str_),
    )
    return path


def _family_order(available_families: Iterable[str], requested_families: list[str] | None) -> list[str]:
    available = set(available_families)
    if requested_families:
        ordered = [family for family in requested_families if family in available]
    else:
        ordered = [family for family in DEFAULT_REFERENCE_FAMILIES if family in available]
        ordered.extend(sorted(available - set(ordered)))
    return ordered


def select_reference_episode_indices(
    processed_root: str | Path,
    split: str,
    num_references: int,
    segment_nn_steps: int,
    seed: int,
    requested_families: list[str] | None = None,
) -> list[int]:
    processed_root = Path(processed_root).expanduser().resolve()
    split_metadata = load_split_metadata(processed_root, split)
    episode_lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
    scenario_families = list(split_metadata["scenario_families"])
    families = _family_order(scenario_families, requested_families)
    rng = np.random.default_rng(seed)

    by_family: dict[str, list[int]] = {}
    for episode_index, family in enumerate(scenario_families):
        if int(episode_lengths[episode_index]) >= segment_nn_steps:
            by_family.setdefault(family, []).append(episode_index)

    for family_indices in by_family.values():
        rng.shuffle(family_indices)

    selected: list[int] = []
    family_cursor = 0
    active_families = [family for family in families if by_family.get(family)]
    while len(selected) < num_references and active_families:
        family = active_families[family_cursor % len(active_families)]
        selected.append(by_family[family].pop())
        if not by_family[family]:
            active_families.remove(family)
            family_cursor -= 1
        family_cursor += 1

    if len(selected) < num_references:
        raise ValueError(
            f"Only selected {len(selected)} references with segment_nn_steps={segment_nn_steps}; "
            f"requested {num_references}."
        )
    return selected


def build_reference_set(
    processed_root: str | Path,
    split: str = "train",
    num_references: int = 20,
    segment_nn_steps: int = 1100,
    seed: int = 20260607,
    requested_families: list[str] | None = None,
    random_segment_start: bool = True,
) -> ReferenceSet:
    """Build a compact reference set from the processed HMMWV cache.

    The large arrays are opened with mmap, and only the selected fixed-length
    segments are copied into the returned reference set.
    """
    processed_root = Path(processed_root).expanduser().resolve()
    metadata = load_metadata(processed_root)
    split_metadata = load_split_metadata(processed_root, split)
    selected_indices = select_reference_episode_indices(
        processed_root=processed_root,
        split=split,
        num_references=num_references,
        segment_nn_steps=segment_nn_steps,
        seed=seed,
        requested_families=requested_families,
    )

    states_mmap = np.load(processed_root / f"{split}_states.npy", mmap_mode="r")
    actions_mmap = np.load(processed_root / f"{split}_actions.npy", mmap_mode="r")
    targets_mmap = np.load(processed_root / f"{split}_targets.npy", mmap_mode="r")
    rollout_mmap = np.load(processed_root / f"{split}_rollout.npy", mmap_mode="r")
    episode_starts = np.load(processed_root / f"{split}_episode_starts.npy")
    episode_lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
    rollout_offsets = split_metadata["rollout_episode_offsets"]
    rng = np.random.default_rng(seed + 17)

    state_segments: list[np.ndarray] = []
    action_segments: list[np.ndarray] = []
    pose_segments: list[np.ndarray] = []
    selected_episode_ids: list[str] = []
    selected_families: list[str] = []
    segment_records: list[dict[str, Any]] = []

    for episode_index in selected_indices:
        episode_length = int(episode_lengths[episode_index])
        max_start = max(0, episode_length - segment_nn_steps)
        local_start = int(rng.integers(0, max_start + 1)) if random_segment_start and max_start > 0 else 0
        transition_start = int(episode_starts[episode_index]) + local_start
        transition_stop = transition_start + segment_nn_steps
        rollout_start = int(rollout_offsets[episode_index]) + local_start
        rollout_stop = rollout_start + segment_nn_steps + 1

        states = np.empty((segment_nn_steps + 1, states_mmap.shape[1]), dtype=np.float32)
        states[:-1] = np.array(states_mmap[transition_start:transition_stop], dtype=np.float32, copy=True)
        states[-1] = states[-2] + np.array(targets_mmap[transition_stop - 1], dtype=np.float32, copy=True)

        actions = np.empty((segment_nn_steps + 1, actions_mmap.shape[1]), dtype=np.float32)
        actions[:-1] = np.array(actions_mmap[transition_start:transition_stop], dtype=np.float32, copy=True)
        actions[-1] = actions[-2]

        poses = np.array(rollout_mmap[rollout_start:rollout_stop], dtype=np.float32, copy=True)
        if poses.shape[0] != segment_nn_steps + 1:
            raise ValueError(
                f"Reference segment for episode index {episode_index} has pose length {poses.shape[0]}, "
                f"expected {segment_nn_steps + 1}."
            )

        episode_id = split_metadata["episode_ids"][episode_index]
        family = split_metadata["scenario_families"][episode_index]
        net_yaw = float(wrap_angle_np(float(poses[-1, 2] - poses[0, 2])))
        net_xy = poses[-1, :2] - poses[0, :2]
        segment_records.append(
            {
                "episode_index": int(episode_index),
                "episode_id": episode_id,
                "scenario_family": family,
                "local_start": local_start,
                "segment_nn_steps": segment_nn_steps,
                "duration_s": float(segment_nn_steps * float(metadata["dt_s"])),
                "net_dx_m": float(net_xy[0]),
                "net_dy_m": float(net_xy[1]),
                "net_yaw_rad": net_yaw,
                "mean_speed_mps": float(np.mean(states[:, 0])),
            }
        )

        state_segments.append(states)
        action_segments.append(actions)
        pose_segments.append(poses)
        selected_episode_ids.append(episode_id)
        selected_families.append(family)

    return ReferenceSet(
        states=np.stack(state_segments, axis=0),
        actions=np.stack(action_segments, axis=0),
        poses=np.stack(pose_segments, axis=0),
        episode_ids=selected_episode_ids,
        scenario_families=selected_families,
        dt_s=float(metadata["dt_s"]),
        state_fields=list(metadata["state_fields"]),
        action_fields=list(metadata["action_fields"]),
        rollout_fields=list(metadata["rollout_fields"]),
        metadata={
            "source_processed_root": str(processed_root),
            "source_split": split,
            "seed": int(seed),
            "segment_nn_steps": int(segment_nn_steps),
            "random_segment_start": bool(random_segment_start),
            "segments": segment_records,
        },
    )


def summarize_reference_set(reference_set: ReferenceSet) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    for family in reference_set.scenario_families:
        family_counts[family] = family_counts.get(family, 0) + 1

    yaw_changes = [
        float(wrap_angle_np(reference_set.poses[index, -1, 2] - reference_set.poses[index, 0, 2]))
        for index in range(reference_set.num_references)
    ]
    return {
        "num_references": reference_set.num_references,
        "num_steps": reference_set.num_steps,
        "duration_s": (reference_set.num_steps - 1) * reference_set.dt_s,
        "families": family_counts,
        "left_turn_like": int(sum(value > math.radians(20.0) for value in yaw_changes)),
        "right_turn_like": int(sum(value < -math.radians(20.0) for value in yaw_changes)),
    }
