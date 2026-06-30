from __future__ import annotations

from pathlib import Path


DEFAULT_RL_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth"
)
DEFAULT_RL_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1"
)
DEFAULT_RL_REFERENCE_PATH = Path(
    "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz"
)

DEFAULT_ARM_DYNAMICS_CHECKPOINT = Path(
    "artifacts/training_runs/arm_transformer_full_v1/checkpoints/best_val.pt"
)
DEFAULT_ARM_PROCESSED_DATASET_DIR = Path(
    "artifacts/training_datasets/arm_dyn_v3_seq16_v1"
)
DEFAULT_ARM_GEOMETRY_PATH = Path(
    "artifacts/arm_geometry/arm_geometry_v1.json"
)
