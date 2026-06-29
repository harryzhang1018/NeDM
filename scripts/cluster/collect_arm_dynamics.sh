#!/bin/bash
#SBATCH --job-name=arm-dyn
#SBATCH --output=logs/arm_out_%A_%a.txt
#SBATCH --error=logs/arm_err_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=research
#SBATCH --time=08:00:00

# Collects arm-only dynamics data in 15 independent shards, intended as one
# shard per Slurm array task:
#
#   mkdir -p logs
#   sbatch --array=0-14%15 scripts/cluster/collect_arm_dynamics.sh
#
# Running without --array loops over all shards sequentially. That is useful for
# local smoke tests and for mopping up incomplete shards. Completed shards
# (dataset_index.json present) are skipped, so the job is safe to resubmit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module load conda/miniforge
  bootstrap-conda
  conda activate nedm
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

NUM_SHARDS="${ARM_NUM_SHARDS:-15}"
EPISODES_PER_SHARD="${ARM_EPISODES_PER_SHARD:-256}"
MAX_STEPS="${ARM_MAX_STEPS:-500}"
SEED_BASE="${ARM_SEED_BASE:-2026062900}"
VALIDATION_RATIO="${ARM_VALIDATION_RATIO:-0.15}"
OUTPUT_ROOT="${ARM_OUTPUT_ROOT:-artifacts/datasets/arm_dynamics_v1_shards}"

# Measured gripper-base EE row balancing. This caps how many free-space rows can
# be written per Cartesian EE bin, using Chrono-measured ee_base values only.
EE_BALANCE_GRID="${ARM_EE_BALANCE_GRID:-8,8,6}"
EE_BIN_CAP="${ARM_EE_BIN_CAP:-512}"
EE_BOUNDS_LO="${ARM_EE_BOUNDS_LO:--6.0,-6.0,-5.5}"
EE_BOUNDS_HI="${ARM_EE_BOUNDS_HI:-6.0,6.0,1.5}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  shards=("$SLURM_ARRAY_TASK_ID")
else
  shards=($(seq 0 $((NUM_SHARDS - 1))))
fi

for shard in "${shards[@]}"; do
  if (( shard < 0 || shard >= NUM_SHARDS )); then
    echo "shard $shard outside configured range 0-$((NUM_SHARDS - 1))" >&2
    exit 2
  fi

  shard_name=$(printf 'shard_%03d' "$shard")
  output_dir="$OUTPUT_ROOT/$shard_name"
  dataset_name=$(printf 'arm_dynamics_v1_s%03d' "$shard")
  episode_prefix=$(printf 'arm_s%03d_ep' "$shard")
  seed=$((SEED_BASE + 1009 * shard))

  if [[ -f "$output_dir/dataset_index.json" ]]; then
    echo "shard $shard already complete at $output_dir; skipping"
    continue
  fi

  echo "collecting arm shard $shard -> $output_dir"
  echo "  episodes=$EPISODES_PER_SHARD max_steps=$MAX_STEPS seed=$seed"
  echo "  EE balance: grid=$EE_BALANCE_GRID cap/bin=$EE_BIN_CAP bounds=$EE_BOUNDS_LO->$EE_BOUNDS_HI"

  "$PYTHON_BIN" -m nedm.arm_data \
    --episodes "$EPISODES_PER_SHARD" \
    --max-steps "$MAX_STEPS" \
    --seed "$seed" \
    --output-dir "$output_dir" \
    --dataset-name "$dataset_name" \
    --episode-prefix "$episode_prefix" \
    --validation-ratio "$VALIDATION_RATIO" \
    --ee-balance-grid="$EE_BALANCE_GRID" \
    --ee-bin-cap "$EE_BIN_CAP" \
    --ee-bounds-lo="$EE_BOUNDS_LO" \
    --ee-bounds-hi="$EE_BOUNDS_HI"

  echo "completed arm shard $shard -> $output_dir"
done

echo "done: ${shards[*]}"
