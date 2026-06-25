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
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.rl.defaults import (
    DEFAULT_RL_DYNAMICS_CHECKPOINT,
    DEFAULT_RL_PROCESSED_DATASET_DIR,
    DEFAULT_RL_REFERENCE_PATH,
)
from nedm.rl.references import build_reference_set, save_reference_set


class NoOpSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        return None


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def configure_torch_runtime(device: str, matmul_precision: str) -> None:
    torch.set_float32_matmul_precision(matmul_precision)
    if device.startswith("cuda") and torch.cuda.is_available():
        allow_tf32 = matmul_precision != "highest"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def parse_terrain_value(value: str | None) -> str | list[float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) > 1:
        try:
            return [float(part) for part in parts]
        except ValueError:
            pass
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trajectory tracking on frozen NN HMMWV dynamics.")
    parser.add_argument("--exp-name", type=str, default="hmmwv-nn-tracking")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="Float32 matmul precision. 'high' enables TF32 on CUDA and is faster for transformer inference.",
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--num-steps-per-env", type=int, default=128)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dynamics-checkpoint",
        type=Path,
        default=DEFAULT_RL_DYNAMICS_CHECKPOINT,
        help="Frozen NN dynamics checkpoint.",
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
        default=DEFAULT_RL_REFERENCE_PATH,
        help="Compact reference set produced by build_hmmwv_rl_references.py.",
    )
    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        help=(
            "Single terrain for all envs when using a terrain-conditioned dynamics checkpoint. "
            "Use flat/rigid for one-hot [1,0], crm for [0,1], or a comma one-hot such as 1,0."
        ),
    )
    parser.add_argument(
        "--terrain-mix",
        type=str,
        default=None,
        help=(
            "Per-env terrain allocation for terrain-conditioned generalist RL, e.g. "
            "flat:1,crm:1 for a 50/50 split or flat:0.75,crm:0.25. Each env samples refs "
            "only from its terrain domain."
        ),
    )
    parser.add_argument(
        "--reference-terrain-domains",
        type=str,
        default=None,
        help=(
            "Override reference terrain labels for old reference files. Use one value for all refs "
            "(flat or crm) or a comma-separated value per reference. Combined flat+CRM refs normally "
            "carry metadata['domains'] and do not need this."
        ),
    )
    parser.add_argument(
        "--build-references-if-missing",
        action="store_true",
        help="Build the default compact reference set if --reference-path is missing.",
    )
    parser.add_argument(
        "--reference-source-dir",
        type=Path,
        default=DEFAULT_RL_PROCESSED_DATASET_DIR,
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
    parser.add_argument(
        "--dynamics-context-steps",
        type=int,
        default=None,
        help="Trailing history tokens fed to the dynamics model each substep "
        "(<= model block_size). None uses the full model context. Smaller is much "
        "faster with near-identical rollout accuracy since the dynamics is ~Markovian.",
    )
    parser.add_argument(
        "--steering-rate-limit",
        type=float,
        default=None,
        help="Max adjacent-step change for scaled steering command. Disabled when unset.",
    )
    parser.add_argument(
        "--action-rate-weight",
        type=float,
        default=None,
        help="Reward penalty weight for squared adjacent driver-command changes. Uses env default when unset.",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=None,
        help="Reward tracking-loss weight for XY position error. Uses env default when unset.",
    )
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=None,
        help="Reward tracking-loss weight for yaw error. Uses env default when unset.",
    )
    parser.add_argument(
        "--state-error-fields",
        type=str,
        default=None,
        help="Comma-separated state fields used by the reward state-error term. Uses all fields when unset.",
    )
    parser.add_argument("--obs-history-steps", type=int, default=10)
    parser.add_argument("--reference-preview-steps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=180)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl_runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        help="RSL-RL logger: tensorboard, wandb, neptune, or none for smoke tests without scalar logging.",
    )
    return parser.parse_args(argv)


def get_train_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": float(args.entropy_coef),
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
        "torch": {
            "matmul_precision": args.matmul_precision,
        },
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
            "terrain": parse_terrain_value(args.terrain),
            "terrain_mix": args.terrain_mix,
            "reference_terrain_domains": args.reference_terrain_domains,
            "dynamics_context_steps": int(args.dynamics_context_steps)
            if args.dynamics_context_steps is not None
            else None,
            "action_repeat": int(args.action_repeat),
            "steering_rate_limit": args.steering_rate_limit,
            "obs_history_steps": int(args.obs_history_steps),
            "reference_preview_steps": int(args.reference_preview_steps),
            "max_episode_steps": int(args.max_episode_steps),
            "auto_reset": True,
            "termination": {"max_position_error_m": float(args.max_position_error_m)},
        }
    )
    if args.action_rate_weight is not None:
        cfg["reward"]["action_rate_weight"] = float(args.action_rate_weight)
    if args.position_weight is not None:
        cfg["reward"]["position_weight"] = float(args.position_weight)
    if args.yaw_weight is not None:
        cfg["reward"]["yaw_weight"] = float(args.yaw_weight)
    if args.state_error_fields is not None:
        cfg["reward"]["state_error_fields"] = [
            field_name.strip() for field_name in args.state_error_fields.split(",") if field_name.strip()
        ]
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
    args.dynamics_checkpoint = resolve_dynamics_checkpoint_path(args.dynamics_checkpoint)
    configure_torch_runtime(args.device, args.matmul_precision)
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
    if str(args.logger).lower() in {"none", "off", "disabled"}:
        runner.writer = NoOpSummaryWriter()
        runner.logger_type = "none"
        print("logger disabled; scalar logging is disabled, checkpoint saves remain enabled")

    print(f"Starting RL tracking training in {run_dir}")
    print(
        f"device={args.device} num_envs={env.num_envs} action_repeat={env.action_repeat} "
        f"matmul_precision={args.matmul_precision}"
    )
    print(f"dynamics_checkpoint={Path(env_cfg['dynamics_checkpoint']).resolve()}")
    print(f"reference_path={Path(env_cfg['reference_path']).resolve()}")
    if env.num_terrains > 0:
        print(f"terrain_vocab={env.terrain_names}")
        print(f"env_terrain_counts={env.terrain_counts()}")
        print(f"reference_terrain_counts={env.reference_terrain_counts()}")
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
