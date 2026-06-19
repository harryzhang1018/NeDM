from __future__ import annotations

"""Robust bumpy fine-tune: Huber loss + dual-domain rollout-based checkpoint
selection. Warm-starts from the 300G flat base and trains on the existing
bumpy10g+flat90g combined cache.

Motivated by artifacts/analysis/hmmwv_bumpy_finetune_diagnosis: the rollout
failure is a longitudinal vx/omega compounding bias, and one-step val_loss
(Fz-dominated) is the wrong checkpoint-selection metric.
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
                    default=REPO_ROOT / "configs/hmmwv_transformer_v07_tire_normal_force_omega_bumpy10g_flat90g_finetune_fz001_track10_lr3em6_last1.json",
                    help="Source config for dataset dir / model / freeze.")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth")
    ap.add_argument("--flat-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1")
    ap.add_argument("--bumpy-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "artifacts/training_runs/hmmwv_transformer_v07_bumpy10g_flat90g_robust_huber_rolloutsel")
    ap.add_argument("--config-out", type=Path,
                    default=REPO_ROOT / "configs/hmmwv_transformer_v07_bumpy10g_flat90g_robust_huber_rolloutsel.json")
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--min-lr", type=float, default=3e-7)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--fz-weight", type=float, default=0.1)
    ap.add_argument("--l2sp-lambda", type=float, default=0.0,
                    help="L2-SP anchor strength (penalize drift from base weights). 0 disables.")
    ap.add_argument("--l2sp-anchor", type=Path, default=None,
                    help="Anchor checkpoint for L2-SP; defaults to --checkpoint (the base).")
    ap.add_argument("--num-epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--rollout-episodes", type=int, default=8)
    ap.add_argument("--rollout-horizon-s", type=float, default=30.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 1 epoch, 20 steps, 2 rollout eps.")
    return ap.parse_args(argv)


def build_config(args) -> dict:
    config = json.loads(Path(args.base_config).read_text())
    l2sp_on = float(args.l2sp_lambda) > 0.0
    if l2sp_on:
        tag = f"l2sp{args.l2sp_lambda:g}_huber{args.huber_delta:g}_fz{args.fz_weight:g}_lr{args.lr:g}_last1"
        if str(args.output_dir).endswith("robust_huber_rolloutsel"):
            args.output_dir = REPO_ROOT / f"artifacts/training_runs/hmmwv_transformer_v07_bumpy10g_flat90g_{tag}"
            args.config_out = REPO_ROOT / f"configs/hmmwv_transformer_v07_bumpy10g_flat90g_{tag}.json"
    config["output_dir"] = str(args.output_dir)
    config["optimizer"]["lr"] = float(args.lr)
    config["optimizer"]["min_lr"] = float(args.min_lr)
    config["optimizer"]["grad_clip_norm"] = float(config["optimizer"].get("grad_clip_norm", 1.0))
    # With L2-SP, anchor toward base (not toward zero), so disable AdamW weight decay.
    if l2sp_on:
        config["optimizer"]["weight_decay"] = 0.0
    config["training"]["device"] = args.device
    config["training"]["resume_from_checkpoint"] = None
    config["training"]["num_epochs"] = int(args.num_epochs)
    config["training"]["steps_per_epoch"] = int(args.steps_per_epoch)
    # Robust loss: Huber, full weight on body/omega/tracking, modest Fz downweight.
    config["loss"] = {
        "type": "huber",
        "huber_delta": float(args.huber_delta),
        "default_target_weight": 1.0,
        "target_weights": {f: float(args.fz_weight) for f in FZ_FIELDS},
    }
    if l2sp_on:
        anchor = args.l2sp_anchor if args.l2sp_anchor is not None else args.checkpoint
        config["loss"]["l2sp_lambda"] = float(args.l2sp_lambda)
        config["loss"]["l2sp_anchor_checkpoint"] = str(Path(anchor).resolve())
    # Dual-domain rollout-based checkpoint selection.
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
        "loss": "huber",
        "huber_delta": float(args.huber_delta),
        "fz_weight": float(args.fz_weight),
        "selection": "dual_domain_rollout_xy",
    }
    if args.smoke:
        config["training"]["num_epochs"] = 1
        config["training"]["steps_per_epoch"] = 20
        config["training"]["max_val_batches"] = 5
        config["rollout_eval"]["num_episodes"] = 2
        config["rollout_eval"]["horizon_s"] = 5.0
    config["sweep_recipe"] = {
        "version": "v07_bumpy10g_flat90g_robust_huber_rolloutsel",
        "slug": f"huber{args.huber_delta:g}_fz{args.fz_weight:g}_lr{args.lr:g}_last1_rolloutsel",
        "notes": (
            "Robust bumpy fine-tune: scaled-Huber loss (delta="
            f"{args.huber_delta:g}) caps impulsive bumpy targets; Fz weight {args.fz_weight:g}; "
            "checkpoint selected by dual-domain (flat+bumpy) open-loop rollout XY RMSE, not "
            "one-step val_loss. Last transformer block + final norm + head trainable."
        ),
    }
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote config {args.config_out}")

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
