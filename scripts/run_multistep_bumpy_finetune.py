from __future__ import annotations

"""Multi-step (rollout-aware) bumpy fine-tune.

Unrolls K steps autoregressively per train sample and back-props through the
closed loop, so the model is optimized against its own predicted-state feedback
instead of teacher-forced one-step targets. Targets the compounding vx/omega
longitudinal drift identified in
artifacts/analysis/hmmwv_bumpy_finetune_diagnosis. Keeps the robust Huber loss,
Fz downweight, and dual-domain rollout-based checkpoint selection.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from nedm.training.trainer import HMMWVTrainer
from finetune_w_bumpy import WarmStartTrainer

FZ_FIELDS = [
    "tire_fl_force_wheel_fz_n",
    "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n",
    "tire_rr_force_wheel_fz_n",
]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path,
                    default=REPO_ROOT / "configs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy10g_flat90g_finetune_fz001_track10_lr3em6_last1.json")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth")
    ap.add_argument("--flat-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1")
    ap.add_argument("--bumpy-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--config-out", type=Path, default=None)
    ap.add_argument("--k-steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--min-lr", type=float, default=3e-7)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--fz-weight", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--train-last-n", type=int, default=1)
    ap.add_argument("--rollout-episodes", type=int, default=8)
    ap.add_argument("--rollout-horizon-s", type=float, default=30.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


def build_config(args) -> dict:
    config = json.loads(Path(args.base_config).read_text())
    tag = f"k{args.k_steps}_huber{args.huber_delta:g}_fz{args.fz_weight:g}_lr{args.lr:g}_last{args.train_last_n}"
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"artifacts/training_runs/hmmwv_transformer_v07_bumpy10g_flat90g_multistep_{tag}"
    if args.config_out is None:
        args.config_out = REPO_ROOT / f"configs/hmmwv_transformer_v07_bumpy10g_flat90g_multistep_{tag}.json"

    config["output_dir"] = str(args.output_dir)
    config["optimizer"]["lr"] = float(args.lr)
    config["optimizer"]["min_lr"] = float(args.min_lr)
    config["optimizer"]["grad_clip_norm"] = float(config["optimizer"].get("grad_clip_norm", 1.0))
    config["training"]["device"] = args.device
    config["training"]["resume_from_checkpoint"] = None
    config["training"]["num_epochs"] = int(args.num_epochs)
    config["training"]["steps_per_epoch"] = int(args.steps_per_epoch)
    config["training"]["batch_size"] = int(args.batch_size)
    config["training"]["multistep"] = {"enabled": True, "k_steps": int(args.k_steps)}
    config["loss"] = {
        "type": "huber",
        "huber_delta": float(args.huber_delta),
        "default_target_weight": 1.0,
        "target_weights": {f: float(args.fz_weight) for f in FZ_FIELDS},
    }
    config["parameter_freeze"] = {
        "train_last_n_transformer_blocks": int(args.train_last_n),
        "train_backbone_final_norm": True,
        "train_head": True,
        "train_input_projection": False,
        "train_position_embedding": False,
    }
    config["rollout_eval"] = {
        "dual_domain": True,
        "flat_dir": str(args.flat_dir),
        "bumpy_dir": str(args.bumpy_dir),
        "num_episodes": int(args.rollout_episodes),
        "horizon_s": float(args.rollout_horizon_s),
        "select_weight_flat": 1.0,
        "select_weight_bumpy": 1.0,
    }
    config["finetune"] = {
        "warm_start_checkpoint": str(args.checkpoint),
        "warm_start_weights_only": True,
        "objective": "multistep_rollout",
        "k_steps": int(args.k_steps),
        "loss": "huber",
        "huber_delta": float(args.huber_delta),
        "fz_weight": float(args.fz_weight),
        "selection": "dual_domain_rollout_xy",
    }
    if args.smoke:
        config["training"]["num_epochs"] = 1
        config["training"]["steps_per_epoch"] = 15
        config["training"]["max_val_batches"] = 3
        config["training"]["batch_size"] = 16
        config["rollout_eval"]["num_episodes"] = 2
        config["rollout_eval"]["horizon_s"] = 5.0
    config["sweep_recipe"] = {
        "version": f"v07_bumpy10g_flat90g_multistep_{tag}",
        "slug": tag,
        "notes": (
            f"Multi-step rollout-aware fine-tune: unroll K={args.k_steps} steps with BPTT "
            f"through closed-loop predicted-state feedback; scaled-Huber delta={args.huber_delta:g}, "
            f"Fz weight {args.fz_weight:g}; checkpoint selected by dual-domain rollout XY RMSE."
        ),
    }
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote config {args.config_out}")
    print(f"k_steps={args.k_steps} batch={config['training']['batch_size']} "
          f"epochs={config['training']['num_epochs']} steps/epoch={config['training']['steps_per_epoch']}")

    last_ckpt = args.output_dir / "checkpoints" / "last.pt"
    if args.resume and last_ckpt.exists():
        resume_config = json.loads(json.dumps(config))
        resume_config["training"]["resume_from_checkpoint"] = str(last_ckpt)
        print(f"resuming from {last_ckpt}")
        trainer = HMMWVTrainer(resume_config)
    else:
        trainer = WarmStartTrainer(config, args.checkpoint.resolve())
    final = trainer.train()
    print(f"done; last checkpoint: {final}")
    print(f"best_rollout_score={trainer.best_rollout_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
