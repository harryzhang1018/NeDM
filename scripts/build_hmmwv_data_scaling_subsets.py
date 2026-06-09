"""Build uniformly-subsampled training caches for the v07 data-scaling study.

Each subset keeps the action/maneuver coverage of the full
``hmmwv_turn_300g_plus_base_seq_v1`` pool by stratifying on
(source_dataset, scenario_family) and picking evenly spaced episodes within
every stratum (deterministic, no random sampling). The val split is shared
with the full cache via symlinks so all models are validated on identical
data, including the fixed rollout-eval episodes.

Subset sizes are labeled by nominal raw gigabytes, where the full train split
of the pool is treated as the 300 GB baseline that trained v07.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATASET = "artifacts/training_datasets/hmmwv_turn_300g_plus_base_seq_v1"
FULL_TRAIN_GB = 300.0
TARGET_GB = [200, 100, 50, 25, 10, 5]

VAL_FILES = [
    "val_states.npy",
    "val_actions.npy",
    "val_targets.npy",
    "val_rollout.npy",
    "val_episode_starts.npy",
    "val_episode_lengths.npy",
    "val_episodes.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_DATASET))
    parser.add_argument("--target-gb", type=int, nargs="+", default=TARGET_GB)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/training_datasets"),
        help="Subset dirs are created here as hmmwv_v07_scaling_<gb>g_seq_v1.",
    )
    return parser.parse_args()


def select_episode_indices(
    source_datasets: list[str],
    scenario_families: list[str],
    fraction: float,
) -> np.ndarray:
    strata: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(source_datasets, scenario_families, strict=True)):
        strata.setdefault(key, []).append(index)

    selected: list[int] = []
    for indices in strata.values():
        count = max(1, int(round(len(indices) * fraction)))
        positions = np.unique(np.round(np.linspace(0, len(indices) - 1, count)).astype(int))
        selected.extend(indices[p] for p in positions)
    return np.array(sorted(selected), dtype=np.int64)


def build_subset(source: Path, output_dir: Path, target_gb: int) -> None:
    fraction = target_gb / FULL_TRAIN_GB
    split_meta = json.loads((source / "train_episodes.json").read_text())
    metadata = json.loads((source / "metadata.json").read_text())

    states = np.load(source / "train_states.npy", mmap_mode="r")
    actions = np.load(source / "train_actions.npy", mmap_mode="r")
    targets = np.load(source / "train_targets.npy", mmap_mode="r")
    rollout = np.load(source / "train_rollout.npy", mmap_mode="r")
    episode_starts = np.load(source / "train_episode_starts.npy")
    episode_lengths = np.load(source / "train_episode_lengths.npy")
    rollout_offsets = np.array(split_meta["rollout_episode_offsets"], dtype=np.int64)

    selected = select_episode_indices(
        split_meta["source_datasets"], split_meta["scenario_families"], fraction
    )
    new_lengths = episode_lengths[selected].astype(np.int64)
    new_rollout_lengths = rollout_offsets[selected + 1] - rollout_offsets[selected]
    total_transitions = int(new_lengths.sum())
    total_rollout = int(new_rollout_lengths.sum())

    output_dir.mkdir(parents=True, exist_ok=True)
    out_arrays = {
        "train_states.npy": (states, total_transitions),
        "train_actions.npy": (actions, total_transitions),
        "train_targets.npy": (targets, total_transitions),
        "train_rollout.npy": (rollout, total_rollout),
    }
    writers = {
        name: open_memmap(
            output_dir / name, mode="w+", dtype=src.dtype, shape=(rows, src.shape[1])
        )
        for name, (src, rows) in out_arrays.items()
    }

    new_starts = np.zeros(len(selected), dtype=np.int64)
    new_rollout_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
    transition_cursor = 0
    rollout_cursor = 0
    for new_index, old_index in enumerate(selected):
        start, length = int(episode_starts[old_index]), int(episode_lengths[old_index])
        writers["train_states.npy"][transition_cursor : transition_cursor + length] = states[start : start + length]
        writers["train_actions.npy"][transition_cursor : transition_cursor + length] = actions[start : start + length]
        writers["train_targets.npy"][transition_cursor : transition_cursor + length] = targets[start : start + length]
        new_starts[new_index] = transition_cursor
        transition_cursor += length

        r_start, r_stop = int(rollout_offsets[old_index]), int(rollout_offsets[old_index + 1])
        writers["train_rollout.npy"][rollout_cursor : rollout_cursor + (r_stop - r_start)] = rollout[r_start:r_stop]
        rollout_cursor += r_stop - r_start
        new_rollout_offsets[new_index + 1] = rollout_cursor
    for writer in writers.values():
        writer.flush()

    np.save(output_dir / "train_episode_starts.npy", new_starts)
    np.save(output_dir / "train_episode_lengths.npy", new_lengths)

    subset_split_meta = {
        "split": "train",
        "episode_count": int(len(selected)),
        "transition_count": total_transitions,
        "episode_ids": [split_meta["episode_ids"][i] for i in selected],
        "scenario_families": [split_meta["scenario_families"][i] for i in selected],
        "source_datasets": [split_meta["source_datasets"][i] for i in selected],
        "source_csv_paths": [split_meta["source_csv_paths"][i] for i in selected],
        "rollout_episode_offsets": new_rollout_offsets.tolist(),
    }
    (output_dir / "train_episodes.json").write_text(json.dumps(subset_split_meta))

    metadata = dict(metadata)
    metadata["dataset_name"] = f"{metadata['dataset_name']}_sub{target_gb}g"
    metadata["splits"] = dict(metadata["splits"])
    metadata["splits"]["train"] = {
        "episode_count": int(len(selected)),
        "transition_count": total_transitions,
    }
    metadata["subset"] = {
        "source_dataset_dir": str(source),
        "nominal_gb": target_gb,
        "fraction_of_full_train": fraction,
        "selection": "evenly spaced per (source_dataset, scenario_family) stratum",
        "normalization": "inherited from full pool for cross-run comparability",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for name in VAL_FILES:
        link = output_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((source / name).resolve())

    print(
        f"{output_dir.name}: {len(selected):,} episodes, {total_transitions:,} transitions "
        f"({total_transitions / split_meta['transition_count'] * FULL_TRAIN_GB:.1f} GB-equivalent)",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    source = (REPO_ROOT / args.source).resolve() if not args.source.is_absolute() else args.source
    for target_gb in args.target_gb:
        output_dir = (REPO_ROOT / args.output_root / f"hmmwv_v07_scaling_{target_gb:03d}g_seq_v1").resolve()
        build_subset(source, output_dir, target_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
