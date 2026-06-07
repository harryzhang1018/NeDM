from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nedm.training.constants import (
    DEFAULT_ACTION_FIELDS,
    DEFAULT_ROLLOUT_FIELDS,
    DEFAULT_STATE_FIELDS,
)


@dataclass(frozen=True)
class SplitBuffers:
    states: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    episode_starts: np.ndarray
    episode_lengths: np.ndarray
    rollout: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compute_dt_s(dataset_root: Path) -> float:
    resolved_cfg_path = dataset_root / "collector_config.resolved.json"
    if not resolved_cfg_path.exists():
        return 0.01
    resolved_cfg = load_json(resolved_cfg_path)
    return float(resolved_cfg["simulation"]["record_step_s"])


def compute_common_dt_s(dataset_roots: list[Path]) -> float:
    dts = [compute_dt_s(dataset_root) for dataset_root in dataset_roots]
    reference = dts[0]
    for dataset_root, dt_s in zip(dataset_roots, dts, strict=True):
        if abs(dt_s - reference) > 1e-12:
            raise ValueError(f"Dataset {dataset_root} has dt={dt_s}, expected {reference}")
    return reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact HMMWV training dataset cache.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=[Path("artifacts/datasets/hmmwv_overfit_6k")],
        help="Root(s) of raw episode datasets to merge into one processed cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_overfit_6k_seq_v1"),
        help="Directory where the processed arrays will be written.",
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="Optional cap per split for quick smoke tests.",
    )
    return parser.parse_args(argv)


def read_episode_csv(
    csv_path: Path,
    state_fields: list[str],
    action_fields: list[str],
    rollout_fields: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    rollout_rows: list[list[float]] = []

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [
            field
            for field in state_fields + action_fields + rollout_fields
            if field not in reader.fieldnames
        ]
        if missing:
            raise KeyError(f"{csv_path} is missing required fields: {missing}")
        for row in reader:
            state_rows.append([float(row[field]) for field in state_fields])
            action_rows.append([float(row[field]) for field in action_fields])
            rollout_rows.append([float(row[field]) for field in rollout_fields])

    states = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    rollout = np.asarray(rollout_rows, dtype=np.float32)
    if len(states) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 rows after warmup trimming")
    return states, actions, rollout


def build_split_buffers(
    episodes: list[dict[str, Any]],
    state_fields: list[str],
    action_fields: list[str],
    rollout_fields: list[str],
) -> SplitBuffers:
    total_transitions = sum(int(ep["rows"]) - 1 for ep in episodes)
    state_dim = len(state_fields)
    action_dim = len(action_fields)
    rollout_dim = len(rollout_fields)

    states = np.empty((total_transitions, state_dim), dtype=np.float32)
    actions = np.empty((total_transitions, action_dim), dtype=np.float32)
    targets = np.empty((total_transitions, state_dim), dtype=np.float32)
    rollout = np.empty((total_transitions + len(episodes), rollout_dim), dtype=np.float32)
    episode_starts = np.empty((len(episodes),), dtype=np.int64)
    episode_lengths = np.empty((len(episodes),), dtype=np.int32)

    cursor = 0
    rollout_cursor = 0
    for episode_index, episode in enumerate(episodes):
        csv_path = Path(episode["_dataset_root"]) / episode["csv_path"]
        episode_states, episode_actions, episode_rollout = read_episode_csv(
            csv_path,
            state_fields=state_fields,
            action_fields=action_fields,
            rollout_fields=rollout_fields,
        )
        length = episode_states.shape[0] - 1
        episode_starts[episode_index] = cursor
        episode_lengths[episode_index] = length

        states[cursor : cursor + length] = episode_states[:-1]
        actions[cursor : cursor + length] = episode_actions[:-1]
        targets[cursor : cursor + length] = episode_states[1:] - episode_states[:-1]
        rollout[rollout_cursor : rollout_cursor + length + 1] = episode_rollout

        cursor += length
        rollout_cursor += length + 1

    return SplitBuffers(
        states=states,
        actions=actions,
        targets=targets,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
        rollout=rollout,
    )


def save_split(output_dir: Path, split: str, buffers: SplitBuffers, episodes: list[dict[str, Any]]) -> None:
    np.save(output_dir / f"{split}_states.npy", buffers.states)
    np.save(output_dir / f"{split}_actions.npy", buffers.actions)
    np.save(output_dir / f"{split}_targets.npy", buffers.targets)
    np.save(output_dir / f"{split}_episode_starts.npy", buffers.episode_starts)
    np.save(output_dir / f"{split}_episode_lengths.npy", buffers.episode_lengths)
    np.save(output_dir / f"{split}_rollout.npy", buffers.rollout)

    split_metadata = {
        "split": split,
        "episode_count": len(episodes),
        "transition_count": int(buffers.targets.shape[0]),
        "episode_ids": [episode["episode_id"] for episode in episodes],
        "scenario_families": [episode["scenario_family"] for episode in episodes],
        "source_datasets": [episode["_dataset_name"] for episode in episodes],
        "source_csv_paths": [
            str(Path(episode["_dataset_root"]) / episode["csv_path"])
            for episode in episodes
        ],
        "rollout_episode_offsets": np.cumsum(
            np.concatenate(([0], buffers.episode_lengths.astype(np.int64) + 1))
        ).tolist(),
    }
    (output_dir / f"{split}_episodes.json").write_text(json.dumps(split_metadata, indent=2))


def summarize_stats(array: np.ndarray) -> tuple[list[float], list[float]]:
    mean = array.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = array.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean.tolist(), std.tolist()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_roots = [dataset_root.resolve() for dataset_root in args.dataset_root]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_indices: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for dataset_root in dataset_roots:
        dataset_index_path = dataset_root / "dataset_index.json"
        if not dataset_index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")
        dataset_index = load_json(dataset_index_path)
        dataset_indices.append(dataset_index)
        for episode in dataset_index["episodes"]:
            episode_record = dict(episode)
            episode_record["_dataset_root"] = str(dataset_root)
            episode_record["_dataset_name"] = str(dataset_index["dataset_name"])
            all_episodes.append(episode_record)

    episode_id_counts = Counter(episode["episode_id"] for episode in all_episodes)
    duplicate_ids = sorted(episode_id for episode_id, count in episode_id_counts.items() if count > 1)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"Duplicate episode IDs across dataset roots: {preview}")

    split_episodes: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for episode in all_episodes:
        split_episodes[episode["split"]].append(episode)

    if args.max_episodes_per_split is not None:
        for split in split_episodes:
            split_episodes[split] = split_episodes[split][: args.max_episodes_per_split]

    state_fields = list(DEFAULT_STATE_FIELDS)
    action_fields = list(DEFAULT_ACTION_FIELDS)
    rollout_fields = list(DEFAULT_ROLLOUT_FIELDS)
    dt_s = compute_common_dt_s(dataset_roots)

    train_buffers = build_split_buffers(
        episodes=split_episodes["train"],
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
    )
    val_buffers = build_split_buffers(
        episodes=split_episodes["val"],
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
    )

    save_split(output_dir, "train", train_buffers, split_episodes["train"])
    save_split(output_dir, "val", val_buffers, split_episodes["val"])

    state_mean, state_std = summarize_stats(train_buffers.states)
    action_mean, action_std = summarize_stats(train_buffers.actions)
    target_mean, target_std = summarize_stats(train_buffers.targets)

    metadata = {
        "dataset_name": "+".join(dataset_index["dataset_name"] for dataset_index in dataset_indices),
        "raw_dataset_root": str(dataset_roots[0]),
        "raw_dataset_roots": [str(dataset_root) for dataset_root in dataset_roots],
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": dt_s,
        "state_fields": state_fields,
        "action_fields": action_fields,
        "target_fields": [f"delta_{field}" for field in state_fields],
        "rollout_fields": rollout_fields,
        "splits": {
            split: {
                "episode_count": len(episodes),
                "transition_count": int(sum(int(ep["rows"]) - 1 for ep in episodes)),
            }
            for split, episodes in split_episodes.items()
        },
        "normalization": {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        f"wrote processed dataset to {output_dir} "
        f"with {metadata['splits']['train']['transition_count']} train transitions and "
        f"{metadata['splits']['val']['transition_count']} val transitions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
