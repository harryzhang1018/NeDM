#!/bin/bash
#SBATCH --job-name=hmmwv-rl-tracking
#SBATCH --output=logs/rl_out_%j.txt
#SBATCH --error=logs/rl_err_%j.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=research
#SBATCH --time=24:00:00

# Trains the HMMWV PPO tracking policy on a cluster GPU node using the
# repository's default RL setup. The current defaults in train_hmmwv_rl_tracking.py
# use the 15-D tire-normal-force/omega dynamics checkpoint and matching reference set.
#
# Submit with:
#
#   mkdir -p logs
#   sbatch scripts/cluster/train_hmmwv_rl_tracking.sh
#
# Common overrides:
#
#   NUM_ENVS=2048 MAX_ITERATIONS=2000 sbatch scripts/cluster/train_hmmwv_rl_tracking.sh
#   RUN_NAME=hmmwv_rl_15d_a100 sbatch scripts/cluster/train_hmmwv_rl_tracking.sh
#   STEERING_RATE_LIMIT=0.08 sbatch scripts/cluster/train_hmmwv_rl_tracking.sh
#   LOGGER=none sbatch scripts/cluster/train_hmmwv_rl_tracking.sh   # if tensorboard is unavailable
#
# Extra train_hmmwv_rl_tracking.py args can be appended after --, for example:
#
#   sbatch scripts/cluster/train_hmmwv_rl_tracking.sh -- --save-interval 50

set -euo pipefail

module load conda/miniforge
bootstrap-conda
conda activate nedm

cd /srv/home/hzhang699/NeDM

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1

NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-128}"
NUM_LEARNING_EPOCHS="${NUM_LEARNING_EPOCHS:-5}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
SEED="${SEED:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
EXP_NAME="${EXP_NAME:-hmmwv-nn-tracking}"
LOGGER="${LOGGER:-tensorboard}"
MATMUL_PRECISION="${MATMUL_PRECISION:-high}"
STEERING_RATE_LIMIT="${STEERING_RATE_LIMIT:-}"
DEFAULT_DYNAMICS_CHECKPOINT="artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth"

if [[ ! -f "$DEFAULT_DYNAMICS_CHECKPOINT" ]]; then
  echo "ERROR: missing dynamics checkpoint: $DEFAULT_DYNAMICS_CHECKPOINT" >&2
  echo "Run git pull origin main on the cluster login node before submitting." >&2
  exit 1
fi

if head -n 1 "$DEFAULT_DYNAMICS_CHECKPOINT" | grep -q "git-lfs"; then
  echo "ERROR: $DEFAULT_DYNAMICS_CHECKPOINT is a Git LFS pointer, not the checkpoint payload." >&2
  echo "Run git pull origin main on the cluster login node to fetch the LFS-free .pth checkpoint." >&2
  exit 1
fi

train_args=(
  --device cuda
  --num-envs "$NUM_ENVS"
  --max-iterations "$MAX_ITERATIONS"
  --num-steps-per-env "$NUM_STEPS_PER_ENV"
  --num-learning-epochs "$NUM_LEARNING_EPOCHS"
  --num-mini-batches "$NUM_MINI_BATCHES"
  --learning-rate "$LEARNING_RATE"
  --seed "$SEED"
  --save-interval "$SAVE_INTERVAL"
  --exp-name "$EXP_NAME"
  --logger "$LOGGER"
  --matmul-precision "$MATMUL_PRECISION"
  --build-references-if-missing
)

if [[ -n "${RUN_NAME:-}" ]]; then
  train_args+=(--run-name "$RUN_NAME")
fi

if [[ -n "$STEERING_RATE_LIMIT" ]]; then
  train_args+=(--steering-rate-limit "$STEERING_RATE_LIMIT")
fi

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  train_args+=("$@")
fi

echo "starting HMMWV RL tracking training"
echo "job_id=${SLURM_JOB_ID:-local}"
echo "num_envs=$NUM_ENVS max_iterations=$MAX_ITERATIONS num_steps_per_env=$NUM_STEPS_PER_ENV"
echo "logger=$LOGGER matmul_precision=$MATMUL_PRECISION steering_rate_limit=${STEERING_RATE_LIMIT:-none}"
echo "extra_args=$*"
python scripts/train_hmmwv_rl_tracking.py "${train_args[@]}"
