from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.preprocess import summarize_stats


DEFAULT_SOURCE = Path("artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1")
DEFAULT_METRICS = Path("artifacts/analysis/hmmwv_bumpy_trajectory_cleaning/trajectory_metrics.csv")
DEFAULT_OUTPUT = Path(
    "artifacts/training_datasets/hmmwv_bumpy_10g_hard_force_clean_normal_force_omega_seq_v1"
)


FILTER_DEFINITIONS = {
    "severe_force_clean": {
        "description": "Remove trajectories with negative Fz state, Fz state > 100k N, or |delta Fz| > 100k N.",
        "keep_column": "severe_force_pathology",
        "keep_value": False,
    },
    "hard_force_clean": {
        "description": "Remove trajectories with negative Fz state, Fz state > 50k N, or |delta Fz| > 20k N.",
        "keep_column": "hard_force_outlier",
        "keep_value": False,
    },
    "body_state_target_flat_like": {
        "description": (
            "Keep non-severe trajectories with <=1% of all state values outside |flat z|>5 "
            "and <=1% of body-7 target values outside |flat z|>5."
        ),
        "query": "(severe_force_pathology == False) and "
        "(state_abs_z_gt_5_frac_all <= 0.01) and "
        "(target_abs_z_gt_5_frac_body7 <= 0.01)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned HMMWV bumpy processed dataset from per-trajectory cleaning metrics."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--filter",
        choices=sorted(FILTER_DEFINITIONS),
        default="hard_force_clean",
    )
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--stats-chunk-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output directory if it already exists.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def copy_rows(
    source: np.ndarray,
    target: np.ndarray,
    source_start: int,
    target_start: int,
    rows: int,
    chunk_rows: int,
) -> None:
    for offset in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - offset)
        target[target_start + offset : target_start + offset + count] = source[
            source_start + offset : source_start + offset + count
        ]


def select_metrics(metrics: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    definition = FILTER_DEFINITIONS[filter_name]
    if "query" in definition:
        selected = metrics.query(str(definition["query"]))
    else:
        column = str(definition["keep_column"])
        keep_value = bool(definition["keep_value"])
        selected = metrics[metrics[column].astype(bool) == keep_value]
    return selected.sort_values(["split", "episode_index"]).copy()


def build_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    selected_metrics: pd.DataFrame,
    chunk_rows: int,
) -> dict[str, Any]:
    split_meta = load_json(source_dir / f"{split}_episodes.json")
    selected = selected_metrics[selected_metrics["split"].eq(split)]
    selected_indices = selected["episode_index"].astype(np.int64).to_numpy()

    states_src = np.load(source_dir / f"{split}_states.npy", mmap_mode="r")
    actions_src = np.load(source_dir / f"{split}_actions.npy", mmap_mode="r")
    targets_src = np.load(source_dir / f"{split}_targets.npy", mmap_mode="r")
    rollout_src = np.load(source_dir / f"{split}_rollout.npy", mmap_mode="r")
    starts_src = np.load(source_dir / f"{split}_episode_starts.npy").astype(np.int64)
    lengths_src = np.load(source_dir / f"{split}_episode_lengths.npy").astype(np.int64)
    rollout_offsets_src = np.asarray(split_meta["rollout_episode_offsets"], dtype=np.int64)

    lengths = lengths_src[selected_indices]
    rollout_lengths = rollout_offsets_src[selected_indices + 1] - rollout_offsets_src[selected_indices]
    transition_count = int(lengths.sum())
    rollout_count = int(rollout_lengths.sum())
    episode_count = int(len(selected_indices))

    states = open_memmap(
        output_dir / f"{split}_states.npy",
        mode="w+",
        dtype=states_src.dtype,
        shape=(transition_count, states_src.shape[1]),
    )
    actions = open_memmap(
        output_dir / f"{split}_actions.npy",
        mode="w+",
        dtype=actions_src.dtype,
        shape=(transition_count, actions_src.shape[1]),
    )
    targets = open_memmap(
        output_dir / f"{split}_targets.npy",
        mode="w+",
        dtype=targets_src.dtype,
        shape=(transition_count, targets_src.shape[1]),
    )
    rollout = open_memmap(
        output_dir / f"{split}_rollout.npy",
        mode="w+",
        dtype=rollout_src.dtype,
        shape=(rollout_count, rollout_src.shape[1]),
    )
    starts = np.empty((episode_count,), dtype=np.int64)
    episode_lengths = lengths.astype(np.int32)
    rollout_offsets = np.empty((episode_count + 1,), dtype=np.int64)
    rollout_offsets[0] = 0

    transition_cursor = 0
    rollout_cursor = 0
    for new_index, old_index in enumerate(selected_indices.tolist()):
        length = int(lengths_src[old_index])
        source_start = int(starts_src[old_index])
        copy_rows(states_src, states, source_start, transition_cursor, length, chunk_rows)
        copy_rows(actions_src, actions, source_start, transition_cursor, length, chunk_rows)
        copy_rows(targets_src, targets, source_start, transition_cursor, length, chunk_rows)
        starts[new_index] = transition_cursor
        transition_cursor += length

        rollout_start = int(rollout_offsets_src[old_index])
        rollout_stop = int(rollout_offsets_src[old_index + 1])
        rollout_len = rollout_stop - rollout_start
        copy_rows(rollout_src, rollout, rollout_start, rollout_cursor, rollout_len, chunk_rows)
        rollout_cursor += rollout_len
        rollout_offsets[new_index + 1] = rollout_cursor

    for array in (states, actions, targets, rollout):
        array.flush()

    np.save(output_dir / f"{split}_episode_starts.npy", starts)
    np.save(output_dir / f"{split}_episode_lengths.npy", episode_lengths)

    selected_list = selected_indices.tolist()
    split_output = {
        "split": split,
        "episode_count": episode_count,
        "transition_count": transition_count,
        "episode_ids": [split_meta["episode_ids"][index] for index in selected_list],
        "scenario_families": [split_meta["scenario_families"][index] for index in selected_list],
        "source_datasets": [split_meta["source_datasets"][index] for index in selected_list],
        "source_csv_paths": [split_meta["source_csv_paths"][index] for index in selected_list],
        "rollout_episode_offsets": rollout_offsets.tolist(),
    }
    write_json(output_dir / f"{split}_episodes.json", split_output)

    return {
        "split": split,
        "source_episode_count": int(split_meta["episode_count"]),
        "source_transition_count": int(split_meta["transition_count"]),
        "retained_episode_count": episode_count,
        "retained_transition_count": transition_count,
        "removed_episode_count": int(split_meta["episode_count"]) - episode_count,
        "removed_transition_count": int(split_meta["transition_count"]) - transition_count,
    }


def recompute_normalization(output_dir: Path, chunk_rows: int) -> dict[str, list[float]]:
    train_states = np.load(output_dir / "train_states.npy", mmap_mode="r")
    train_actions = np.load(output_dir / "train_actions.npy", mmap_mode="r")
    train_targets = np.load(output_dir / "train_targets.npy", mmap_mode="r")
    state_mean, state_std = summarize_stats(train_states, chunk_rows=chunk_rows)
    action_mean, action_std = summarize_stats(train_actions, chunk_rows=chunk_rows)
    target_mean, target_std = summarize_stats(train_targets, chunk_rows=chunk_rows)
    return {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def main() -> int:
    args = parse_args()
    source_dir = resolve(args.source_dir)
    metrics_csv = resolve(args.metrics_csv)
    output_dir = resolve(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to recreate it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    metadata = load_json(source_dir / "metadata.json")
    metrics = pd.read_csv(metrics_csv)
    selected = select_metrics(metrics, args.filter)
    selected_ids = set(zip(selected["split"], selected["episode_id"], strict=True))

    split_summaries = [
        build_split(source_dir, output_dir, split, selected, int(args.chunk_rows))
        for split in ("train", "val")
    ]

    normalization = recompute_normalization(output_dir, int(args.stats_chunk_rows))
    train_summary = next(item for item in split_summaries if item["split"] == "train")
    val_summary = next(item for item in split_summaries if item["split"] == "val")

    output_metadata = dict(metadata)
    output_metadata["dataset_name"] = f"{metadata.get('dataset_name', 'hmmwv_bumpy')}_{args.filter}"
    output_metadata["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_metadata["normalization"] = normalization
    output_metadata["splits"] = {
        "train": {
            "episode_count": train_summary["retained_episode_count"],
            "transition_count": train_summary["retained_transition_count"],
        },
        "val": {
            "episode_count": val_summary["retained_episode_count"],
            "transition_count": val_summary["retained_transition_count"],
        },
    }
    output_metadata["trajectory_filter"] = {
        "source_processed_dataset_dir": str(source_dir),
        "source_metrics_csv": str(metrics_csv),
        "filter_name": args.filter,
        "filter_description": FILTER_DEFINITIONS[args.filter]["description"],
        "normalization": "recomputed from retained train split",
        "source_trajectory_count": int(len(metrics)),
        "retained_trajectory_count": int(len(selected)),
        "removed_trajectory_count": int(len(metrics) - len(selected)),
        "split_summaries": split_summaries,
    }
    write_json(output_dir / "metadata.json", output_metadata)

    selected[["split", "episode_index", "episode_id", "scenario_family", "source_dataset", "source_csv_path"]].to_csv(
        output_dir / "selected_trajectories.csv", index=False
    )
    removed = metrics[
        ~metrics.apply(lambda row: (row["split"], row["episode_id"]) in selected_ids, axis=1)
    ].copy()
    removed[
        ["split", "episode_index", "episode_id", "scenario_family", "source_dataset", "source_csv_path"]
    ].to_csv(output_dir / "removed_trajectories.csv", index=False)

    print(f"wrote {output_dir}")
    for summary in split_summaries:
        print(
            f"{summary['split']}: retained {summary['retained_episode_count']}/"
            f"{summary['source_episode_count']} episodes, "
            f"{summary['retained_transition_count']:,}/"
            f"{summary['source_transition_count']:,} transitions"
        )
    print(f"total retained episodes: {len(selected):,}; removed: {len(metrics) - len(selected):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
