from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.references import build_reference_set, save_reference_set, summarize_reference_set


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact HMMWV RL tracking references from processed data.")
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_turn_300g_plus_base_seq_v1"),
        help="Processed HMMWV dataset cache directory.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Source split.")
    parser.add_argument("--num-references", type=int, default=20, help="Number of reference trajectories.")
    parser.add_argument(
        "--segment-nn-steps",
        type=int,
        default=1100,
        help="Number of NN-frequency transitions per compact reference segment.",
    )
    parser.add_argument("--seed", type=int, default=20260607, help="Selection seed.")
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="Optional comma-separated maneuver families to prefer.",
    )
    parser.add_argument(
        "--no-random-segment-start",
        action="store_true",
        help="Use the start of each selected source episode instead of a random segment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rl_reference_sets/hmmwv_train_refs_20_1100_seed_20260607.npz"),
        help="Output compact reference set path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested_families = None
    if args.families:
        requested_families = [family.strip() for family in args.families.split(",") if family.strip()]

    reference_set = build_reference_set(
        processed_root=args.processed_dataset_dir,
        split=args.split,
        num_references=args.num_references,
        segment_nn_steps=args.segment_nn_steps,
        seed=args.seed,
        requested_families=requested_families,
        random_segment_start=not args.no_random_segment_start,
    )
    output_path = save_reference_set(reference_set, args.output)
    summary = summarize_reference_set(reference_set)
    summary["output"] = str(output_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
