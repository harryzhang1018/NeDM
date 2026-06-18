from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.constants import STATE_FIELD_PRESETS


UNCHANGED_FILES = [
    "train_actions.npy",
    "train_episode_lengths.npy",
    "train_episode_starts.npy",
    "train_episodes.json",
    "train_rollout.npy",
    "val_actions.npy",
    "val_episode_lengths.npy",
    "val_episode_starts.npy",
    "val_episodes.json",
    "val_rollout.npy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a processed HMMWV cache by selecting state columns from another processed cache."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--state-field-preset",
        type=str,
        choices=sorted(STATE_FIELD_PRESETS),
        required=True,
        help="Preset to select from the source cache.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows per copy chunk for state/target arrays.",
    )
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="Copy unchanged action/rollout/split files instead of symlinking them.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def replace_with_symlink_or_copy(source: Path, target: Path, copy_file: bool) -> None:
    if target.is_symlink() or target.exists():
        target.unlink()
    if copy_file:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source.resolve())


def copy_selected_columns(source_path: Path, target_path: Path, indices: list[int], chunk_rows: int) -> None:
    source = np.load(source_path, mmap_mode="r")
    target = open_memmap(
        target_path,
        mode="w+",
        dtype=source.dtype,
        shape=(source.shape[0], len(indices)),
    )
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        target[start:stop] = source[start:stop, :][:, indices]
    target.flush()


def select_values(values: list[float], indices: list[int]) -> list[float]:
    return [values[index] for index in indices]


def main() -> int:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((source_dir / "metadata.json").read_text())
    source_fields = list(metadata["state_fields"])
    selected_fields = list(STATE_FIELD_PRESETS[args.state_field_preset])
    missing = [field for field in selected_fields if field not in source_fields]
    if missing:
        raise ValueError(f"{source_dir} is missing selected state fields: {missing}")

    selected_indices = [source_fields.index(field) for field in selected_fields]
    for split in ("train", "val"):
        copy_selected_columns(
            source_dir / f"{split}_states.npy",
            output_dir / f"{split}_states.npy",
            selected_indices,
            args.chunk_rows,
        )
        copy_selected_columns(
            source_dir / f"{split}_targets.npy",
            output_dir / f"{split}_targets.npy",
            selected_indices,
            args.chunk_rows,
        )

    for name in UNCHANGED_FILES:
        replace_with_symlink_or_copy(source_dir / name, output_dir / name, copy_file=bool(args.copy_unchanged))

    normalization = dict(metadata["normalization"])
    normalization["state_mean"] = select_values(normalization["state_mean"], selected_indices)
    normalization["state_std"] = select_values(normalization["state_std"], selected_indices)
    normalization["target_mean"] = select_values(normalization["target_mean"], selected_indices)
    normalization["target_std"] = select_values(normalization["target_std"], selected_indices)

    subset_metadata = dict(metadata)
    subset_metadata["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
    subset_metadata["state_field_preset"] = args.state_field_preset
    subset_metadata["state_fields"] = selected_fields
    subset_metadata["target_fields"] = [f"delta_{field}" for field in selected_fields]
    subset_metadata["normalization"] = normalization
    subset_metadata["column_subset"] = {
        "source_processed_dataset_dir": str(source_dir),
        "source_state_field_preset": metadata.get("state_field_preset"),
        "source_state_dim": len(source_fields),
        "selected_state_dim": len(selected_fields),
        "selected_source_indices": selected_indices,
        "unchanged_files": "copied" if args.copy_unchanged else "symlinked",
    }
    (output_dir / "metadata.json").write_text(json.dumps(subset_metadata, indent=2))

    print(
        f"wrote {output_dir} from {source_dir}: "
        f"{len(source_fields)} -> {len(selected_fields)} state fields"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
