from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.training.trainer import HMMWVTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rollout evaluation for a trained HMMWV model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint produced by train_hmmwv_dynamics.py")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for evaluation metrics.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for evaluation. Use 'cpu' for portability or 'cuda' on the training machine.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu")
    checkpoint["config"]["training"]["device"] = args.device
    trainer = HMMWVTrainer(checkpoint["config"])
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.to(trainer.device)
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metrics = trainer.evaluate_windows()
    metrics.update(trainer.evaluate_rollouts())
    print(json.dumps(metrics, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
