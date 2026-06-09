from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.rl.hmmwv_tracking_env import HMMWVNeuralTrackingEnv, default_env_cfg, merge_env_cfg
from nedm.rl.references import build_reference_set, save_reference_set


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory tracking on frozen NN HMMWV dynamics.")
    parser.add_argument("--exp-name", type=str, default="hmmwv-nn-tracking")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--num-steps-per-env", type=int, default=128)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=Path("artifacts/training_runs/hmmwv_transformer_v07_context128_b64/checkpoints/best_val.pt"),
        help="Frozen NN dynamics checkpoint. Swap this to v04/v07/future checkpoints.",
    )
    parser.add_argument(
        "--processed-dataset-dir",
        type=Path,
        default=None,
        help="Optional processed dataset override for checkpoints missing embedded metadata.",
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=Path("artifacts/rl_reference_sets/hmmwv_train_refs_20_1100_seed_20260607.npz"),
        help="Compact reference set produced by build_hmmwv_rl_references.py.",
    )
    parser.add_argument(
        "--build-references-if-missing",
        action="store_true",
        help="Build the default compact reference set if --reference-path is missing.",
    )
    parser.add_argument(
        "--reference-source-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_turn_300g_plus_base_seq_v1"),
        help="Processed dataset used only when building missing references.",
    )
    parser.add_argument("--num-references", type=int, default=20)
    parser.add_argument("--reference-segment-nn-steps", type=int, default=1100)
    parser.add_argument("--reference-seed", type=int, default=20260607)
    parser.add_argument(
        "--max-position-error-m",
        type=float,
        default=20.0,
        help="Episode terminates when tracking position error exceeds this (termination.max_position_error_m).",
    )
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--obs-history-steps", type=int, default=10)
    parser.add_argument("--reference-preview-steps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--logger", type=str, default="tensorboard")
    return parser.parse_args(argv)


def get_train_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.003,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": float(args.learning_rate),
            "max_grad_norm": 1.0,
            "num_learning_epochs": int(args.num_learning_epochs),
            "num_mini_batches": int(args.num_mini_batches),
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 0.7,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": args.exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": int(args.max_iterations),
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.save_interval),
        "empirical_normalization": True,
        "logger": args.logger,
        "seed": int(args.seed),
    }


def get_env_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_env_cfg()
    cfg.update(
        {
            "num_envs": int(args.num_envs),
            "device": resolve_device(args.device),
            "dynamics_checkpoint": str(args.dynamics_checkpoint),
            "processed_dataset_dir": str(args.processed_dataset_dir) if args.processed_dataset_dir else None,
            "reference_path": str(args.reference_path),
            "action_repeat": int(args.action_repeat),
            "obs_history_steps": int(args.obs_history_steps),
            "reference_preview_steps": int(args.reference_preview_steps),
            "max_episode_steps": int(args.max_episode_steps),
            "auto_reset": True,
            "termination": {"max_position_error_m": float(args.max_position_error_m)},
        }
    )
    return merge_env_cfg(cfg)


def ensure_reference_file(args: argparse.Namespace) -> None:
    if args.reference_path.exists():
        return
    if not args.build_references_if_missing:
        raise FileNotFoundError(
            f"Reference set not found: {args.reference_path}. "
            "Run scripts/build_hmmwv_rl_references.py first or pass --build-references-if-missing."
        )
    reference_set = build_reference_set(
        processed_root=args.reference_source_dir,
        split="train",
        num_references=args.num_references,
        segment_nn_steps=args.reference_segment_nn_steps,
        seed=args.reference_seed,
    )
    save_reference_set(reference_set, args.reference_path)


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_stem = args.dynamics_checkpoint.parents[1].name
        run_name = f"{args.exp_name}_{checkpoint_stem}_{timestamp}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = resolve_device(args.device)
    ensure_reference_file(args)
    run_dir = make_run_dir(args)
    env_cfg = get_env_cfg(args)
    train_cfg = get_train_cfg(args)

    (run_dir / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (run_dir / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    env = HMMWVNeuralTrackingEnv(env_cfg, device=args.device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=str(run_dir), device=args.device)

    print(f"Starting RL tracking training in {run_dir}")
    print(f"device={args.device} num_envs={env.num_envs} action_repeat={env.action_repeat}")
    print(f"dynamics_checkpoint={Path(env_cfg['dynamics_checkpoint']).resolve()}")
    print(f"reference_path={Path(env_cfg['reference_path']).resolve()}")
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
