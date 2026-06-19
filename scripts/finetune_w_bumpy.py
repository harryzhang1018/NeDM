from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.constants import DEFAULT_ACTION_FIELDS, DEFAULT_ROLLOUT_FIELDS, STATE_FIELD_PRESETS
from nedm.training.preprocess import summarize_stats
from nedm.training.trainer import HMMWVTrainer


STATE_FIELD_PRESET = "tire_normal_force_omega"
WHEEL_NAMES = ("tire_fl", "tire_fr", "tire_rl", "tire_rr")
FZ_TARGET_FIELDS = [f"{wheel_name}_force_wheel_fz_n" for wheel_name in WHEEL_NAMES]
TRACKING_TARGET_FIELDS = ["vel_body_x_mps", "vel_body_y_mps", "yaw_rate_radps"]
DEFAULT_WEIGHT_N = 25242.0
CONTACT_FZ_N = 50.0
NORMALIZATION_KEYS = {
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
    "target_mean",
    "target_std",
}


def tire_field_names() -> list[str]:
    fields: list[str] = []
    for wheel_name in WHEEL_NAMES:
        fields.extend(
            [
                f"{wheel_name}_longitudinal_slip",
                f"{wheel_name}_slip_angle_rad",
                f"{wheel_name}_camber_angle_rad",
                f"{wheel_name}_force_world_x_n",
                f"{wheel_name}_force_world_y_n",
                f"{wheel_name}_force_world_z_n",
                f"{wheel_name}_moment_world_x_nm",
                f"{wheel_name}_moment_world_y_nm",
                f"{wheel_name}_moment_world_z_nm",
                f"{wheel_name}_force_wheel_fx_n",
                f"{wheel_name}_force_wheel_fy_n",
                f"{wheel_name}_force_wheel_fz_n",
                f"{wheel_name}_spindle_omega_radps",
                f"{wheel_name}_wheel_vx_mps",
                f"{wheel_name}_slip_ratio",
                f"{wheel_name}_deflection_m",
            ]
        )
    return fields


@dataclass(frozen=True)
class FlatSelection:
    split: str
    target_transitions: int
    selected_episode_indices: list[int]
    selected_transitions: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the HMMWV tire-normal-force/omega dynamics model on the "
            "10G bumpy dataset plus a uniform 5G-equivalent subset of the 300G flat dataset."
        )
    )
    parser.add_argument(
        "--bumpy-raw-root",
        type=Path,
        default=Path("artifacts/datasets/hmmwv_bumpy_10g_shards"),
        help="Raw bumpy shard root. shard_* children are validated; smoke is ignored.",
    )
    parser.add_argument(
        "--bumpy-processed-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--flat-processed-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--combined-output-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_bumpy_10g_plus_flat5g_normal_force_omega_seq_v1"),
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("bumpy_flat", "flat_only"),
        default="bumpy_flat",
        help=(
            "bumpy_flat builds the original 10G bumpy + flat subset mix. "
            "flat_only builds a uniform flat subset using --flat-target-gb."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/hmmwv_transformer_v07_tire_normal_force_omega_300g.json"),
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path("configs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy_finetune_15g.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/"
            "checkpoints/best_val.pth"
        ),
        help="300G checkpoint used as the warm-start source.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy_finetune_15g"),
    )
    parser.add_argument(
        "--flat-target-gb",
        type=float,
        default=5.0,
        help="Raw-size equivalent to sample from the flat 300G pool.",
    )
    parser.add_argument(
        "--flat-source-gb",
        type=float,
        default=300.0,
        help="Raw-size equivalent of the flat source pool.",
    )
    parser.add_argument(
        "--fz-loss-weight",
        type=float,
        default=None,
        help="Optional loss weight for the four wheel-frame Fz target dimensions.",
    )
    parser.add_argument(
        "--tracking-loss-weight",
        type=float,
        default=None,
        help="Optional loss weight for vx, vy, and yaw-rate target dimensions.",
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=0.1,
        help="Fine-tune LR multiplier relative to the base config when --finetune-lr is not set.",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Explicit fine-tune max LR. Overrides --lr-scale for optimizer.lr.",
    )
    parser.add_argument(
        "--finetune-min-lr",
        type=float,
        default=None,
        help="Explicit fine-tune min LR. Defaults to base min_lr times --lr-scale.",
    )
    parser.add_argument(
        "--train-last-n-transformer-blocks",
        type=int,
        default=None,
        help="Freeze the backbone except the last N transformer blocks, final norm, and output head.",
    )
    parser.add_argument("--num-epochs", type=int, default=None, help="Override training.num_epochs in the output config.")
    parser.add_argument("--seed", type=int, default=2026061601)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--stats-chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--raw-validation-episodes", type=int, default=48)
    parser.add_argument("--skip-raw-validation", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/build/configure, then exit before training.")
    parser.add_argument("--log-file", type=Path, default=None, help="Append stdout/stderr to this log file.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-combined-normalization",
        action="store_true",
        help=(
            "Use the combined dataset statistics for training. By default the script keeps "
            "the warm-start checkpoint normalization, which preserves the pretrained function."
        ),
    )
    return parser.parse_args(argv)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def write_status(run_dir: Path, state: str, stage: str, message: str) -> None:
    write_json(
        run_dir / "status.json",
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "stage": stage,
            "message": message,
        },
    )


def expected_state_fields() -> list[str]:
    return list(STATE_FIELD_PRESETS[STATE_FIELD_PRESET])


def validate_raw_bumpy_shards(raw_root: Path, max_episodes: int, seed: int) -> None:
    shard_dirs = sorted(path for path in raw_root.glob("shard_*") if (path / "dataset_index.json").exists())
    if not shard_dirs:
        raise FileNotFoundError(f"no completed bumpy shard_* directories found under {raw_root}")

    required_fields = set(expected_state_fields())
    required_fields.update(DEFAULT_ACTION_FIELDS)
    required_fields.update(DEFAULT_ROLLOUT_FIELDS)
    required_fields.update(tire_field_names())
    fz_cols = [f"{name}_force_wheel_fz_n" for name in WHEEL_NAMES]
    slip_cols = [f"{name}_slip_ratio" for name in WHEEL_NAMES]

    for shard_index, shard_dir in enumerate(shard_dirs):
        index = load_json(shard_dir / "dataset_index.json")
        episodes = list(index.get("episodes", []))
        csv_paths = sorted((shard_dir / "episodes").glob("*.csv"))
        if len(csv_paths) != len(episodes):
            raise ValueError(
                f"{shard_dir}: index lists {len(episodes)} episodes but disk has {len(csv_paths)} CSV files"
            )

        rng = random.Random(seed + shard_index)
        sampled_paths = csv_paths if len(csv_paths) <= max_episodes else rng.sample(csv_paths, max_episodes)
        rows_checked = 0
        median_fz_by_episode: list[float] = []
        slip_extreme = 0.0
        airborne_slip_extreme = 0.0

        for csv_path in sampled_paths:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                missing = sorted(required_fields - fieldnames)
                if missing:
                    raise KeyError(f"{csv_path} is missing required fields, e.g. {missing[:6]}")
                fz_sums: list[float] = []
                for row_index, row in enumerate(reader):
                    rows_checked += 1
                    for field in required_fields:
                        value = float(row[field])
                        if not math.isfinite(value):
                            sample_index = row.get("sample_index", row_index)
                            raise ValueError(f"{csv_path}: non-finite {field} at sample {sample_index}")
                    fz_sums.append(sum(float(row[col]) for col in fz_cols))
                    for fz_col, slip_col in zip(fz_cols, slip_cols, strict=True):
                        slip = abs(float(row[slip_col]))
                        if float(row[fz_col]) > CONTACT_FZ_N:
                            slip_extreme = max(slip_extreme, slip)
                        else:
                            airborne_slip_extreme = max(airborne_slip_extreme, slip)
                if fz_sums:
                    fz_sums.sort()
                    median_fz_by_episode.append(fz_sums[len(fz_sums) // 2])

        if not median_fz_by_episode:
            raise ValueError(f"{shard_dir}: no raw rows checked")
        median_fz_by_episode.sort()
        median_fz = median_fz_by_episode[len(median_fz_by_episode) // 2]
        fz_ratio = median_fz / DEFAULT_WEIGHT_N
        if not 0.85 <= fz_ratio <= 1.15:
            raise ValueError(f"{shard_dir}: sum-Fz/weight ratio {fz_ratio:.3f} outside [0.85, 1.15]")
        if slip_extreme > 50:
            raise ValueError(f"{shard_dir}: absurd in-contact slip ratio {slip_extreme:.1f}")
        log(
            f"validated {shard_dir.name}: {len(episodes)} episodes, {rows_checked} sampled rows, "
            f"sum-Fz/weight={fz_ratio:.3f}, max contact |slip|={slip_extreme:.2f}, "
            f"airborne |slip|={airborne_slip_extreme:.2f}"
        )


def validate_processed_dataset(root: Path, *, full_finite_check: bool, chunk_rows: int) -> dict[str, Any]:
    metadata = load_json(root / "metadata.json")
    state_fields = list(metadata["state_fields"])
    if state_fields != expected_state_fields():
        raise ValueError(f"{root} has state fields {state_fields}; expected {expected_state_fields()}")
    if list(metadata["action_fields"]) != list(DEFAULT_ACTION_FIELDS):
        raise ValueError(f"{root} has unexpected action fields: {metadata['action_fields']}")
    if list(metadata["rollout_fields"]) != list(DEFAULT_ROLLOUT_FIELDS):
        raise ValueError(f"{root} has unexpected rollout fields: {metadata['rollout_fields']}")
    if abs(float(metadata["dt_s"]) - 0.01) > 1e-12:
        raise ValueError(f"{root} has dt_s={metadata['dt_s']}, expected 0.01")

    for split in ("train", "val"):
        states = np.load(root / f"{split}_states.npy", mmap_mode="r")
        actions = np.load(root / f"{split}_actions.npy", mmap_mode="r")
        targets = np.load(root / f"{split}_targets.npy", mmap_mode="r")
        rollout = np.load(root / f"{split}_rollout.npy", mmap_mode="r")
        starts = np.load(root / f"{split}_episode_starts.npy")
        lengths = np.load(root / f"{split}_episode_lengths.npy")
        split_meta = load_json(root / f"{split}_episodes.json")

        if states.shape != targets.shape:
            raise ValueError(f"{root} {split}: states shape {states.shape} != targets shape {targets.shape}")
        if states.shape[1] != len(state_fields):
            raise ValueError(f"{root} {split}: state dim {states.shape[1]} != {len(state_fields)}")
        if actions.shape != (states.shape[0], len(DEFAULT_ACTION_FIELDS)):
            raise ValueError(f"{root} {split}: unexpected actions shape {actions.shape}")
        if rollout.shape[1] != len(DEFAULT_ROLLOUT_FIELDS):
            raise ValueError(f"{root} {split}: unexpected rollout shape {rollout.shape}")
        if int(lengths.sum()) != int(states.shape[0]):
            raise ValueError(f"{root} {split}: sum(lengths) != state rows")
        if starts.shape[0] != lengths.shape[0] or starts.shape[0] != int(split_meta["episode_count"]):
            raise ValueError(f"{root} {split}: split episode metadata is inconsistent")
        if starts.size and starts[0] != 0:
            raise ValueError(f"{root} {split}: first episode start is not zero")
        if starts.size > 1 and not np.all(starts[:-1] + lengths[:-1] == starts[1:]):
            raise ValueError(f"{root} {split}: episode starts are not contiguous")

        arrays_to_check = (states, actions, targets, rollout)
        for array in arrays_to_check:
            if full_finite_check:
                for start in range(0, array.shape[0], chunk_rows):
                    if not np.isfinite(np.asarray(array[start : start + chunk_rows])).all():
                        raise ValueError(f"{root} {split}: non-finite values in {array}")
            elif not np.isfinite(np.asarray(array[: min(array.shape[0], 10_000)])).all():
                raise ValueError(f"{root} {split}: non-finite values in initial sample")

    log(
        f"validated processed cache {root}: "
        f"train={metadata['splits']['train']['transition_count']} transitions, "
        f"val={metadata['splits']['val']['transition_count']} transitions"
    )
    return metadata


def validate_checkpoint(checkpoint_path: Path, reference_metadata: dict[str, Any]) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = checkpoint.get("metadata") or {}
    if list(checkpoint_metadata.get("state_fields", [])) != list(reference_metadata["state_fields"]):
        raise ValueError(f"{checkpoint_path} state fields do not match the fine-tune dataset")
    if list(checkpoint_metadata.get("action_fields", [])) != list(reference_metadata["action_fields"]):
        raise ValueError(f"{checkpoint_path} action fields do not match the fine-tune dataset")
    if int((checkpoint.get("config") or {}).get("model", {}).get("block_size", -1)) <= 0:
        raise ValueError(f"{checkpoint_path} does not look like a full trainer checkpoint")
    log(
        f"validated warm-start checkpoint {checkpoint_path}: "
        f"epoch={checkpoint.get('epoch')}, global_step={checkpoint.get('global_step')}, "
        f"val_loss={(checkpoint.get('metrics') or {}).get('val_loss')}"
    )
    return checkpoint


def select_flat_episodes(flat_root: Path, split: str, flat_fraction: float, seed: int) -> FlatSelection:
    lengths = np.load(flat_root / f"{split}_episode_lengths.npy").astype(np.int64)
    total_transitions = int(lengths.sum())
    target_transitions = max(1, int(round(total_transitions * flat_fraction)))
    mean_length = max(float(lengths.mean()), 1.0)
    n_select = max(1, int(round(target_transitions / mean_length)))
    cumulative = np.cumsum(lengths, dtype=np.int64)
    rng = np.random.default_rng(seed)
    phase = float(rng.random())

    selected = np.array([], dtype=np.int64)
    selected_transitions = 0
    while selected_transitions < target_transitions and n_select <= len(lengths):
        positions = ((np.arange(n_select, dtype=np.float64) + phase) / n_select) * total_transitions
        candidate_indices = np.searchsorted(cumulative, positions.astype(np.int64), side="right")
        selected = np.unique(np.clip(candidate_indices, 0, len(lengths) - 1))
        selected_transitions = int(lengths[selected].sum())
        if selected_transitions < target_transitions:
            deficit = target_transitions - selected_transitions
            n_select += max(1, int(math.ceil(deficit / mean_length)))

    return FlatSelection(
        split=split,
        target_transitions=target_transitions,
        selected_episode_indices=selected.astype(np.int64).tolist(),
        selected_transitions=selected_transitions,
    )


def copy_array_rows(
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


def combine_split(
    bumpy_root: Path,
    flat_root: Path,
    output_dir: Path,
    split: str,
    selection: FlatSelection,
    chunk_rows: int,
) -> dict[str, Any]:
    bumpy_states = np.load(bumpy_root / f"{split}_states.npy", mmap_mode="r")
    bumpy_actions = np.load(bumpy_root / f"{split}_actions.npy", mmap_mode="r")
    bumpy_targets = np.load(bumpy_root / f"{split}_targets.npy", mmap_mode="r")
    bumpy_rollout = np.load(bumpy_root / f"{split}_rollout.npy", mmap_mode="r")
    bumpy_starts = np.load(bumpy_root / f"{split}_episode_starts.npy")
    bumpy_lengths = np.load(bumpy_root / f"{split}_episode_lengths.npy").astype(np.int64)
    bumpy_split = load_json(bumpy_root / f"{split}_episodes.json")

    flat_states = np.load(flat_root / f"{split}_states.npy", mmap_mode="r")
    flat_actions = np.load(flat_root / f"{split}_actions.npy", mmap_mode="r")
    flat_targets = np.load(flat_root / f"{split}_targets.npy", mmap_mode="r")
    flat_rollout = np.load(flat_root / f"{split}_rollout.npy", mmap_mode="r")
    flat_starts = np.load(flat_root / f"{split}_episode_starts.npy").astype(np.int64)
    flat_lengths = np.load(flat_root / f"{split}_episode_lengths.npy").astype(np.int64)
    flat_split = load_json(flat_root / f"{split}_episodes.json")
    flat_rollout_offsets = list(int(value) for value in flat_split["rollout_episode_offsets"])

    selected_indices = selection.selected_episode_indices
    flat_selected_lengths = flat_lengths[selected_indices] if selected_indices else np.empty((0,), dtype=np.int64)
    transition_count = int(bumpy_states.shape[0] + flat_selected_lengths.sum())
    rollout_count = int(bumpy_rollout.shape[0] + int(flat_selected_lengths.sum()) + len(selected_indices))
    episode_count = int(len(bumpy_lengths) + len(selected_indices))

    states = open_memmap(
        output_dir / f"{split}_states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, bumpy_states.shape[1]),
    )
    actions = open_memmap(
        output_dir / f"{split}_actions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, bumpy_actions.shape[1]),
    )
    targets = open_memmap(
        output_dir / f"{split}_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, bumpy_targets.shape[1]),
    )
    rollout = open_memmap(
        output_dir / f"{split}_rollout.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rollout_count, bumpy_rollout.shape[1]),
    )
    episode_starts = np.empty((episode_count,), dtype=np.int64)
    episode_lengths = np.empty((episode_count,), dtype=np.int32)

    copy_array_rows(bumpy_states, states, 0, 0, bumpy_states.shape[0], chunk_rows)
    copy_array_rows(bumpy_actions, actions, 0, 0, bumpy_actions.shape[0], chunk_rows)
    copy_array_rows(bumpy_targets, targets, 0, 0, bumpy_targets.shape[0], chunk_rows)
    copy_array_rows(bumpy_rollout, rollout, 0, 0, bumpy_rollout.shape[0], chunk_rows)
    episode_starts[: len(bumpy_starts)] = bumpy_starts
    episode_lengths[: len(bumpy_lengths)] = bumpy_lengths.astype(np.int32)

    cursor = int(bumpy_states.shape[0])
    rollout_cursor = int(bumpy_rollout.shape[0])
    episode_cursor = len(bumpy_lengths)
    for flat_episode_index in selected_indices:
        length = int(flat_lengths[flat_episode_index])
        source_start = int(flat_starts[flat_episode_index])
        copy_array_rows(flat_states, states, source_start, cursor, length, chunk_rows)
        copy_array_rows(flat_actions, actions, source_start, cursor, length, chunk_rows)
        copy_array_rows(flat_targets, targets, source_start, cursor, length, chunk_rows)

        source_rollout_start = flat_rollout_offsets[flat_episode_index]
        source_rollout_stop = flat_rollout_offsets[flat_episode_index + 1]
        copy_array_rows(
            flat_rollout,
            rollout,
            source_rollout_start,
            rollout_cursor,
            source_rollout_stop - source_rollout_start,
            chunk_rows,
        )
        episode_starts[episode_cursor] = cursor
        episode_lengths[episode_cursor] = length
        cursor += length
        rollout_cursor += source_rollout_stop - source_rollout_start
        episode_cursor += 1

    for array in (states, actions, targets, rollout):
        array.flush()

    np.save(output_dir / f"{split}_episode_starts.npy", episode_starts)
    np.save(output_dir / f"{split}_episode_lengths.npy", episode_lengths)

    episode_ids = list(bumpy_split["episode_ids"]) + [
        flat_split["episode_ids"][index] for index in selected_indices
    ]
    scenario_families = list(bumpy_split["scenario_families"]) + [
        flat_split["scenario_families"][index] for index in selected_indices
    ]
    source_datasets = list(bumpy_split["source_datasets"]) + [
        flat_split["source_datasets"][index] for index in selected_indices
    ]
    source_csv_paths = list(bumpy_split["source_csv_paths"]) + [
        flat_split["source_csv_paths"][index] for index in selected_indices
    ]
    rollout_offsets = np.cumsum(np.concatenate(([0], episode_lengths.astype(np.int64) + 1))).tolist()
    split_metadata = {
        "split": split,
        "episode_count": episode_count,
        "transition_count": transition_count,
        "episode_ids": episode_ids,
        "scenario_families": scenario_families,
        "source_datasets": source_datasets,
        "source_csv_paths": source_csv_paths,
        "rollout_episode_offsets": rollout_offsets,
    }
    write_json(output_dir / f"{split}_episodes.json", split_metadata)
    return {
        "split": split,
        "bumpy_episode_count": int(len(bumpy_lengths)),
        "bumpy_transition_count": int(bumpy_states.shape[0]),
        "flat_episode_count": int(len(selected_indices)),
        "flat_transition_count": int(flat_selected_lengths.sum()),
        "flat_target_transitions": int(selection.target_transitions),
        "transition_count": transition_count,
        "episode_count": episode_count,
    }


def copy_flat_subset_split(
    flat_root: Path,
    output_dir: Path,
    split: str,
    selection: FlatSelection,
    chunk_rows: int,
) -> dict[str, Any]:
    flat_states = np.load(flat_root / f"{split}_states.npy", mmap_mode="r")
    flat_actions = np.load(flat_root / f"{split}_actions.npy", mmap_mode="r")
    flat_targets = np.load(flat_root / f"{split}_targets.npy", mmap_mode="r")
    flat_rollout = np.load(flat_root / f"{split}_rollout.npy", mmap_mode="r")
    flat_starts = np.load(flat_root / f"{split}_episode_starts.npy").astype(np.int64)
    flat_lengths = np.load(flat_root / f"{split}_episode_lengths.npy").astype(np.int64)
    flat_split = load_json(flat_root / f"{split}_episodes.json")
    flat_rollout_offsets = list(int(value) for value in flat_split["rollout_episode_offsets"])

    selected_indices = selection.selected_episode_indices
    selected_lengths = flat_lengths[selected_indices] if selected_indices else np.empty((0,), dtype=np.int64)
    transition_count = int(selected_lengths.sum())
    rollout_count = int(
        sum(flat_rollout_offsets[index + 1] - flat_rollout_offsets[index] for index in selected_indices)
    )
    episode_count = int(len(selected_indices))

    states = open_memmap(
        output_dir / f"{split}_states.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, flat_states.shape[1]),
    )
    actions = open_memmap(
        output_dir / f"{split}_actions.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, flat_actions.shape[1]),
    )
    targets = open_memmap(
        output_dir / f"{split}_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(transition_count, flat_targets.shape[1]),
    )
    rollout = open_memmap(
        output_dir / f"{split}_rollout.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rollout_count, flat_rollout.shape[1]),
    )
    episode_starts = np.empty((episode_count,), dtype=np.int64)
    episode_lengths = np.empty((episode_count,), dtype=np.int32)

    cursor = 0
    rollout_cursor = 0
    for episode_cursor, flat_episode_index in enumerate(selected_indices):
        length = int(flat_lengths[flat_episode_index])
        source_start = int(flat_starts[flat_episode_index])
        copy_array_rows(flat_states, states, source_start, cursor, length, chunk_rows)
        copy_array_rows(flat_actions, actions, source_start, cursor, length, chunk_rows)
        copy_array_rows(flat_targets, targets, source_start, cursor, length, chunk_rows)

        source_rollout_start = flat_rollout_offsets[flat_episode_index]
        source_rollout_stop = flat_rollout_offsets[flat_episode_index + 1]
        copy_array_rows(
            flat_rollout,
            rollout,
            source_rollout_start,
            rollout_cursor,
            source_rollout_stop - source_rollout_start,
            chunk_rows,
        )
        episode_starts[episode_cursor] = cursor
        episode_lengths[episode_cursor] = length
        cursor += length
        rollout_cursor += source_rollout_stop - source_rollout_start

    for array in (states, actions, targets, rollout):
        array.flush()

    np.save(output_dir / f"{split}_episode_starts.npy", episode_starts)
    np.save(output_dir / f"{split}_episode_lengths.npy", episode_lengths)

    rollout_offsets = np.cumsum(np.concatenate(([0], episode_lengths.astype(np.int64) + 1))).tolist()
    split_metadata = {
        "split": split,
        "episode_count": episode_count,
        "transition_count": transition_count,
        "episode_ids": [flat_split["episode_ids"][index] for index in selected_indices],
        "scenario_families": [flat_split["scenario_families"][index] for index in selected_indices],
        "source_datasets": [flat_split["source_datasets"][index] for index in selected_indices],
        "source_csv_paths": [flat_split["source_csv_paths"][index] for index in selected_indices],
        "rollout_episode_offsets": rollout_offsets,
    }
    write_json(output_dir / f"{split}_episodes.json", split_metadata)
    return {
        "split": split,
        "flat_episode_count": episode_count,
        "flat_transition_count": transition_count,
        "flat_target_transitions": int(selection.target_transitions),
        "transition_count": transition_count,
        "episode_count": episode_count,
    }


def build_combined_dataset(
    bumpy_root: Path,
    flat_root: Path,
    output_dir: Path,
    checkpoint: dict[str, Any],
    flat_target_gb: float,
    flat_source_gb: float,
    flat_fraction: float,
    seed: int,
    chunk_rows: int,
    stats_chunk_rows: int,
    use_combined_normalization: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bumpy_meta = load_json(bumpy_root / "metadata.json")
    flat_meta = load_json(flat_root / "metadata.json")
    if list(bumpy_meta["state_fields"]) != list(flat_meta["state_fields"]):
        raise ValueError("bumpy and flat processed caches use different state fields")
    if list(bumpy_meta["action_fields"]) != list(flat_meta["action_fields"]):
        raise ValueError("bumpy and flat processed caches use different action fields")
    if list(bumpy_meta["rollout_fields"]) != list(flat_meta["rollout_fields"]):
        raise ValueError("bumpy and flat processed caches use different rollout fields")
    if abs(float(bumpy_meta["dt_s"]) - float(flat_meta["dt_s"])) > 1e-12:
        raise ValueError("bumpy and flat processed caches use different dt_s")

    selections = {
        split: select_flat_episodes(flat_root, split, flat_fraction, seed + split_index)
        for split_index, split in enumerate(("train", "val"))
    }
    split_summaries = {
        split: combine_split(bumpy_root, flat_root, output_dir, split, selections[split], chunk_rows)
        for split in ("train", "val")
    }

    train_states = np.load(output_dir / "train_states.npy", mmap_mode="r")
    train_actions = np.load(output_dir / "train_actions.npy", mmap_mode="r")
    train_targets = np.load(output_dir / "train_targets.npy", mmap_mode="r")
    combined_normalization = {
        "state_mean": None,
        "state_std": None,
        "action_mean": None,
        "action_std": None,
        "target_mean": None,
        "target_std": None,
    }
    combined_normalization["state_mean"], combined_normalization["state_std"] = summarize_stats(
        train_states, chunk_rows=stats_chunk_rows
    )
    combined_normalization["action_mean"], combined_normalization["action_std"] = summarize_stats(
        train_actions, chunk_rows=stats_chunk_rows
    )
    combined_normalization["target_mean"], combined_normalization["target_std"] = summarize_stats(
        train_targets, chunk_rows=stats_chunk_rows
    )

    checkpoint_metadata = checkpoint.get("metadata") or {}
    source_normalization = checkpoint_metadata.get("normalization")
    if source_normalization is None:
        raise ValueError("warm-start checkpoint metadata does not include normalization")
    training_normalization = combined_normalization if use_combined_normalization else source_normalization
    normalization_source = "combined_dataset" if use_combined_normalization else "warm_start_checkpoint"

    metadata = {
        "dataset_name": f"hmmwv_bumpy_10g_plus_flat{flat_target_gb:g}g_uniform",
        "raw_dataset_root": bumpy_meta.get("raw_dataset_root"),
        "raw_dataset_roots": sorted(
            set(bumpy_meta.get("raw_dataset_roots", [])) | set(flat_meta.get("raw_dataset_roots", []))
        ),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": float(bumpy_meta["dt_s"]),
        "state_field_preset": STATE_FIELD_PRESET,
        "state_fields": list(bumpy_meta["state_fields"]),
        "action_fields": list(bumpy_meta["action_fields"]),
        "target_fields": [f"delta_{field}" for field in bumpy_meta["state_fields"]],
        "rollout_fields": list(bumpy_meta["rollout_fields"]),
        "splits": {
            split: {
                "episode_count": int(summary["episode_count"]),
                "transition_count": int(summary["transition_count"]),
            }
            for split, summary in split_summaries.items()
        },
        "normalization": training_normalization,
        "combined_data_normalization": combined_normalization,
        "normalization_source": normalization_source,
        "finetune_mix": {
            "dataset_mode": "bumpy_flat",
            "bumpy_processed_dir": str(bumpy_root),
            "flat_processed_dir": str(flat_root),
            "flat_fraction": flat_fraction,
            "flat_target_gb": flat_target_gb,
            "flat_source_gb": flat_source_gb,
            "seed": seed,
            "state_field_preset": STATE_FIELD_PRESET,
            "split_summaries": split_summaries,
            "warm_start_checkpoint_epoch": checkpoint.get("epoch"),
            "warm_start_checkpoint_global_step": checkpoint.get("global_step"),
            "warm_start_checkpoint_val_loss": (checkpoint.get("metrics") or {}).get("val_loss"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    log(
        f"built combined cache {output_dir}: "
        f"train={metadata['splits']['train']['transition_count']} transitions, "
        f"val={metadata['splits']['val']['transition_count']} transitions, "
        f"normalization={normalization_source}"
    )
    return metadata


def build_flat_only_dataset(
    flat_root: Path,
    output_dir: Path,
    checkpoint: dict[str, Any],
    flat_target_gb: float,
    flat_source_gb: float,
    flat_fraction: float,
    seed: int,
    chunk_rows: int,
    stats_chunk_rows: int,
    use_combined_normalization: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    flat_meta = load_json(flat_root / "metadata.json")
    selections = {
        split: select_flat_episodes(flat_root, split, flat_fraction, seed + split_index)
        for split_index, split in enumerate(("train", "val"))
    }
    split_summaries = {
        split: copy_flat_subset_split(flat_root, output_dir, split, selections[split], chunk_rows)
        for split in ("train", "val")
    }

    train_states = np.load(output_dir / "train_states.npy", mmap_mode="r")
    train_actions = np.load(output_dir / "train_actions.npy", mmap_mode="r")
    train_targets = np.load(output_dir / "train_targets.npy", mmap_mode="r")
    subset_normalization = {
        "state_mean": None,
        "state_std": None,
        "action_mean": None,
        "action_std": None,
        "target_mean": None,
        "target_std": None,
    }
    subset_normalization["state_mean"], subset_normalization["state_std"] = summarize_stats(
        train_states, chunk_rows=stats_chunk_rows
    )
    subset_normalization["action_mean"], subset_normalization["action_std"] = summarize_stats(
        train_actions, chunk_rows=stats_chunk_rows
    )
    subset_normalization["target_mean"], subset_normalization["target_std"] = summarize_stats(
        train_targets, chunk_rows=stats_chunk_rows
    )

    checkpoint_metadata = checkpoint.get("metadata") or {}
    source_normalization = checkpoint_metadata.get("normalization")
    if source_normalization is None:
        raise ValueError("warm-start checkpoint metadata does not include normalization")
    training_normalization = subset_normalization if use_combined_normalization else source_normalization
    normalization_source = "flat_subset" if use_combined_normalization else "warm_start_checkpoint"

    metadata = {
        "dataset_name": f"hmmwv_flat{flat_target_gb:g}g_uniform",
        "raw_dataset_root": flat_meta.get("raw_dataset_root"),
        "raw_dataset_roots": list(flat_meta.get("raw_dataset_roots", [])),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": float(flat_meta["dt_s"]),
        "state_field_preset": STATE_FIELD_PRESET,
        "state_fields": list(flat_meta["state_fields"]),
        "action_fields": list(flat_meta["action_fields"]),
        "target_fields": [f"delta_{field}" for field in flat_meta["state_fields"]],
        "rollout_fields": list(flat_meta["rollout_fields"]),
        "splits": {
            split: {
                "episode_count": int(summary["episode_count"]),
                "transition_count": int(summary["transition_count"]),
            }
            for split, summary in split_summaries.items()
        },
        "normalization": training_normalization,
        "flat_subset_normalization": subset_normalization,
        "normalization_source": normalization_source,
        "finetune_mix": {
            "dataset_mode": "flat_only",
            "flat_processed_dir": str(flat_root),
            "flat_fraction": flat_fraction,
            "flat_target_gb": flat_target_gb,
            "flat_source_gb": flat_source_gb,
            "seed": seed,
            "state_field_preset": STATE_FIELD_PRESET,
            "split_summaries": split_summaries,
            "warm_start_checkpoint_epoch": checkpoint.get("epoch"),
            "warm_start_checkpoint_global_step": checkpoint.get("global_step"),
            "warm_start_checkpoint_val_loss": (checkpoint.get("metrics") or {}).get("val_loss"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    log(
        f"built flat-only cache {output_dir}: "
        f"train={metadata['splits']['train']['transition_count']} transitions, "
        f"val={metadata['splits']['val']['transition_count']} transitions, "
        f"normalization={normalization_source}"
    )
    return metadata


def dataset_mix_matches(metadata: dict[str, Any], args: argparse.Namespace, flat_fraction: float) -> bool:
    mix = metadata.get("finetune_mix") or {}
    dataset_mode = str(getattr(args, "dataset_mode", "bumpy_flat"))
    metadata_mode = str(mix.get("dataset_mode", "bumpy_flat"))
    if args.use_combined_normalization:
        expected_normalization_source = "flat_subset" if dataset_mode == "flat_only" else "combined_dataset"
    else:
        expected_normalization_source = "warm_start_checkpoint"
    return (
        metadata_mode == dataset_mode
        and mix.get("state_field_preset") == STATE_FIELD_PRESET
        and abs(float(mix.get("flat_fraction", -1.0)) - flat_fraction) < 1e-12
        and int(mix.get("seed", -1)) == int(args.seed)
        and metadata.get("normalization_source") == expected_normalization_source
    )


def ensure_flat_only_dataset(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    flat_fraction: float,
) -> dict[str, Any]:
    output_dir = resolve(args.combined_output_dir)
    if (output_dir / "metadata.json").exists() and not args.rebuild_dataset:
        metadata = load_json(output_dir / "metadata.json")
        if not dataset_mix_matches(metadata, args, flat_fraction):
            raise ValueError(
                f"{output_dir} already exists but does not match requested flat-only subset; rerun with "
                "--rebuild-dataset or choose a different --combined-output-dir"
            )
        validate_processed_dataset(output_dir, full_finite_check=False, chunk_rows=args.chunk_rows)
        log(f"reusing existing flat-only cache {output_dir}")
        return metadata

    return build_flat_only_dataset(
        flat_root=resolve(args.flat_processed_dir),
        output_dir=output_dir,
        checkpoint=checkpoint,
        flat_target_gb=float(args.flat_target_gb),
        flat_source_gb=float(args.flat_source_gb),
        flat_fraction=flat_fraction,
        seed=int(args.seed),
        chunk_rows=int(args.chunk_rows),
        stats_chunk_rows=int(args.stats_chunk_rows),
        use_combined_normalization=bool(args.use_combined_normalization),
    )


def ensure_combined_dataset(
    args: argparse.Namespace,
    bumpy_meta: dict[str, Any],
    checkpoint: dict[str, Any],
    flat_fraction: float,
) -> dict[str, Any]:
    output_dir = resolve(args.combined_output_dir)
    if (output_dir / "metadata.json").exists() and not args.rebuild_dataset:
        metadata = load_json(output_dir / "metadata.json")
        if not dataset_mix_matches(metadata, args, flat_fraction):
            raise ValueError(
                f"{output_dir} already exists but does not match requested mix; rerun with --rebuild-dataset "
                "or choose a different --combined-output-dir"
            )
        validate_processed_dataset(output_dir, full_finite_check=False, chunk_rows=args.chunk_rows)
        log(f"reusing existing combined cache {output_dir}")
        return metadata

    return build_combined_dataset(
        bumpy_root=resolve(args.bumpy_processed_dir),
        flat_root=resolve(args.flat_processed_dir),
        output_dir=output_dir,
        checkpoint=checkpoint,
        flat_target_gb=float(args.flat_target_gb),
        flat_source_gb=float(args.flat_source_gb),
        flat_fraction=flat_fraction,
        seed=int(args.seed),
        chunk_rows=int(args.chunk_rows),
        stats_chunk_rows=int(args.stats_chunk_rows),
        use_combined_normalization=bool(args.use_combined_normalization),
    )


def write_finetune_config(args: argparse.Namespace, combined_dir: Path) -> dict[str, Any]:
    base_config = load_json(resolve(args.base_config))
    config = json.loads(json.dumps(base_config))
    config["processed_dataset_dir"] = str(combined_dir.relative_to(REPO_ROOT) if combined_dir.is_relative_to(REPO_ROOT) else combined_dir)
    run_dir = resolve(args.run_dir)
    config["output_dir"] = str(run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir)
    base_lr = float(base_config["optimizer"]["lr"])
    base_min_lr = float(base_config["optimizer"]["min_lr"])
    lr_scale = float(args.lr_scale)
    config["optimizer"]["lr"] = float(args.finetune_lr) if args.finetune_lr is not None else base_lr * lr_scale
    config["optimizer"]["min_lr"] = (
        float(args.finetune_min_lr) if args.finetune_min_lr is not None else base_min_lr * lr_scale
    )
    config["training"]["seed"] = int(args.seed)
    config["training"]["load_dataset_into_memory"] = True
    config["training"]["pin_memory"] = False
    config["training"]["resume_from_checkpoint"] = None
    if args.num_epochs is not None:
        num_epochs = int(args.num_epochs)
        if num_epochs <= 0:
            raise ValueError("--num-epochs must be positive")
        config["training"]["num_epochs"] = num_epochs
    config["finetune"] = {
        "warm_start_checkpoint": str(resolve(args.checkpoint)),
        "warm_start_weights_only": True,
        "base_optimizer_lr": base_config["optimizer"]["lr"],
        "base_optimizer_min_lr": base_config["optimizer"]["min_lr"],
        "finetune_lr": config["optimizer"]["lr"],
        "finetune_min_lr": config["optimizer"]["min_lr"],
        "lr_scale": float(config["optimizer"]["lr"]) / base_lr,
    }
    target_weights: dict[str, float] = {}
    if args.fz_loss_weight is not None:
        fz_loss_weight = float(args.fz_loss_weight)
        if fz_loss_weight < 0.0:
            raise ValueError("--fz-loss-weight must be non-negative")
        target_weights.update({field: fz_loss_weight for field in FZ_TARGET_FIELDS})
        config["finetune"]["fz_loss_weight"] = fz_loss_weight
    if args.tracking_loss_weight is not None:
        tracking_loss_weight = float(args.tracking_loss_weight)
        if tracking_loss_weight < 0.0:
            raise ValueError("--tracking-loss-weight must be non-negative")
        target_weights.update({field: tracking_loss_weight for field in TRACKING_TARGET_FIELDS})
        config["finetune"]["tracking_loss_weight"] = tracking_loss_weight
        config["finetune"]["tracking_loss_fields"] = list(TRACKING_TARGET_FIELDS)
    if target_weights:
        config["loss"] = {"target_weights": target_weights}
    if args.train_last_n_transformer_blocks is not None:
        train_last_n_transformer_blocks = int(args.train_last_n_transformer_blocks)
        if train_last_n_transformer_blocks < 0:
            raise ValueError("--train-last-n-transformer-blocks must be non-negative")
        config["parameter_freeze"] = {
            "train_last_n_transformer_blocks": train_last_n_transformer_blocks,
            "train_backbone_final_norm": True,
            "train_head": True,
            "train_input_projection": False,
            "train_position_embedding": False,
        }
        config["finetune"]["train_last_n_transformer_blocks"] = train_last_n_transformer_blocks
    lr_divisor = base_lr / float(config["optimizer"]["lr"])
    lr_tag = f"lr{lr_divisor:g}x_lower".replace(".", "p")
    fz_tag = "_fz_weighted" if args.fz_loss_weight is not None else ""
    tracking_tag = (
        f"_track{float(args.tracking_loss_weight):g}x".replace(".", "p")
        if args.tracking_loss_weight is not None
        else ""
    )
    freeze_tag = (
        f"_last{int(args.train_last_n_transformer_blocks)}block"
        if args.train_last_n_transformer_blocks is not None
        else ""
    )
    flat_target_tag = f"{float(args.flat_target_gb):g}g"
    dataset_version = (
        f"v07_tire_normal_force_omega_flat{flat_target_tag}_finetune"
        if args.dataset_mode == "flat_only"
        else f"v07_tire_normal_force_omega_bumpy10g_flat{flat_target_tag}_finetune"
    )
    dataset_slug = (
        f"flat{flat_target_tag}"
        if args.dataset_mode == "flat_only"
        else f"bumpy10g_plus_flat{flat_target_tag}"
    )
    dataset_notes = (
        f"a uniform {float(args.flat_target_gb):g}G-equivalent flat-terrain subset."
        if args.dataset_mode == "flat_only"
        else f"the 10G bumpy dataset plus a uniform {float(args.flat_target_gb):g}G-equivalent flat subset."
    )
    config["sweep_recipe"] = {
        "version": (
            f"{dataset_version}{fz_tag}{tracking_tag}{freeze_tag}"
            if args.fz_loss_weight is not None or args.tracking_loss_weight is not None
            else dataset_version
        ),
        "slug": (
            f"{dataset_slug}_{lr_tag}{fz_tag}{tracking_tag}{freeze_tag}"
            if args.fz_loss_weight is not None or args.tracking_loss_weight is not None
            else f"{dataset_slug}_{lr_tag}{freeze_tag}"
        ),
        "notes": (
            "Fine-tune from the 300G flat-terrain tire-normal-force/omega checkpoint on "
            f"{dataset_notes}"
            + (
                f" The four wheel-frame Fz target losses are weighted by {float(args.fz_loss_weight):g}."
                if args.fz_loss_weight is not None
                else ""
            )
            + (
                " The vx, vy, and yaw-rate target losses are weighted by "
                f"{float(args.tracking_loss_weight):g}."
                if args.tracking_loss_weight is not None
                else ""
            )
            + (
                f" Only the last {int(args.train_last_n_transformer_blocks)} transformer block(s), "
                "final norm, and output head are trainable."
                if args.train_last_n_transformer_blocks is not None
                else ""
            )
        ),
    }
    write_json(resolve(args.config_out), config)
    log(
        f"wrote fine-tune config {resolve(args.config_out)} with lr={config['optimizer']['lr']} "
        f"and min_lr={config['optimizer']['min_lr']}"
        + (
            f", fz_loss_weight={float(args.fz_loss_weight):g}"
            if args.fz_loss_weight is not None
            else ""
        )
        + (
            f", tracking_loss_weight={float(args.tracking_loss_weight):g}"
            if args.tracking_loss_weight is not None
            else ""
        )
        + (
            f", train_last_n_transformer_blocks={int(args.train_last_n_transformer_blocks)}"
            if args.train_last_n_transformer_blocks is not None
            else ""
        )
        + (f", num_epochs={int(args.num_epochs)}" if args.num_epochs is not None else "")
    )
    return config


class WarmStartTrainer(HMMWVTrainer):
    def __init__(self, config: dict[str, Any], warm_start_checkpoint: Path) -> None:
        super().__init__(config)
        self.load_warm_start_checkpoint(warm_start_checkpoint)

    def load_warm_start_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        checkpoint_metadata = checkpoint.get("metadata") or {}
        if list(checkpoint_metadata.get("state_fields", [])) != list(self.metadata["state_fields"]):
            raise ValueError(f"{checkpoint_path} state fields do not match the fine-tune dataset")
        if list(checkpoint_metadata.get("action_fields", [])) != list(self.metadata["action_fields"]):
            raise ValueError(f"{checkpoint_path} action fields do not match the fine-tune dataset")
        model_state = checkpoint["model_state_dict"]
        if self.metadata.get("normalization_source") != "warm_start_checkpoint":
            filtered_state = {key: value for key, value in model_state.items() if key not in NORMALIZATION_KEYS}
            missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
            if set(missing) != NORMALIZATION_KEYS or unexpected:
                raise RuntimeError(
                    "unexpected warm-start state mismatch when preserving combined normalization: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        else:
            self.model.load_state_dict(model_state)
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        log(f"warm-started model weights from {checkpoint_path}; optimizer and LR schedule are fresh")


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def install_log_file(log_file: Path | None) -> None:
    if log_file is None:
        return
    resolved = resolve(log_file)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handle = resolved.open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, handle)  # type: ignore[assignment]
    sys.stderr = Tee(sys.stderr, handle)  # type: ignore[assignment]
    print(f"\n===== finetune_w_bumpy.py started {datetime.now(timezone.utc).isoformat()} =====", flush=True)


def run_training(config: dict[str, Any], args: argparse.Namespace) -> Path:
    run_dir = resolve(args.run_dir)
    last_checkpoint = run_dir / "checkpoints" / "last.pt"
    if args.resume and last_checkpoint.exists():
        resume_config = json.loads(json.dumps(config))
        resume_config["training"]["resume_from_checkpoint"] = str(last_checkpoint)
        log(f"resuming fine-tune from {last_checkpoint}")
        trainer = HMMWVTrainer(resume_config)
    else:
        trainer = WarmStartTrainer(config, resolve(args.checkpoint))
    return trainer.train()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    install_log_file(args.log_file)
    run_dir = resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        write_status(run_dir, "running", "validation", "validating fine-tune inputs")
        flat_fraction = float(args.flat_target_gb) / float(args.flat_source_gb)
        if flat_fraction <= 0.0 or flat_fraction > 1.0:
            raise ValueError(f"invalid flat fraction {flat_fraction}")

        if args.dataset_mode == "bumpy_flat" and not args.skip_raw_validation:
            validate_raw_bumpy_shards(
                resolve(args.bumpy_raw_root),
                max_episodes=int(args.raw_validation_episodes),
                seed=int(args.seed),
            )
        flat_meta = validate_processed_dataset(
            resolve(args.flat_processed_dir),
            full_finite_check=False,
            chunk_rows=int(args.chunk_rows),
        )

        if args.dataset_mode == "bumpy_flat":
            bumpy_meta = validate_processed_dataset(
                resolve(args.bumpy_processed_dir),
                full_finite_check=True,
                chunk_rows=int(args.chunk_rows),
            )
            checkpoint = validate_checkpoint(resolve(args.checkpoint), bumpy_meta)
            write_status(run_dir, "running", "dataset", "building or reusing combined fine-tune cache")
            combined_meta = ensure_combined_dataset(args, bumpy_meta, checkpoint, flat_fraction)
        else:
            checkpoint = validate_checkpoint(resolve(args.checkpoint), flat_meta)
            write_status(run_dir, "running", "dataset", "building or reusing flat-only fine-tune cache")
            combined_meta = ensure_flat_only_dataset(args, checkpoint, flat_fraction)
        combined_dir = resolve(args.combined_output_dir)
        validate_processed_dataset(combined_dir, full_finite_check=False, chunk_rows=int(args.chunk_rows))

        write_status(run_dir, "running", "config", "writing fine-tune config")
        config = write_finetune_config(args, combined_dir)
        log(
            "fine-tune mix summary: "
            f"train={combined_meta['splits']['train']['transition_count']} transitions, "
            f"val={combined_meta['splits']['val']['transition_count']} transitions"
        )

        if args.prepare_only:
            write_status(run_dir, "complete", "prepare", "validation, dataset, and config completed")
            log("prepare-only requested; exiting before training")
            return 0

        write_status(run_dir, "running", "training", "fine-tuning dynamics model")
        final_checkpoint = run_training(config, args)
        write_status(run_dir, "complete", "training", f"training completed: {final_checkpoint}")
        log(f"training completed; last checkpoint: {final_checkpoint}")
        return 0
    except Exception as exc:
        write_status(run_dir, "failed", "error", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
