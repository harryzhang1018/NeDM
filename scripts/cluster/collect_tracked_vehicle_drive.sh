#!/bin/bash
#SBATCH --job-name=tracked-drive
#SBATCH --output=logs/tracked_drive_out_%A_%a.txt
#SBATCH --error=logs/tracked_drive_err_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=research
#SBATCH --time=02:00:00

# Collects the tracked-vehicle (M113 + arm-at-home) drive-mode dataset in
# independent shards, one shard per Slurm array task. Like arm_data.py, this
# collector reuses build_scene() from arm_data.py -- the same fragile
# single-pin track model -- and runs strictly sequentially (one Chrono scene
# per episode, no --jobs), so shards are the only parallelism knob:
#
#   mkdir -p logs
#   sbatch --array=0-14%15 scripts/cluster/collect_tracked_vehicle_drive.sh
#
# Running without --array loops over all shards sequentially, useful for
# local smoke tests and for mopping up incomplete shards. Completed shards
# (dataset_index.json present) are skipped, so the job is safe to resubmit
# after a timeout or node failure.
#
# The total scenario count is read from the config itself (not hardcoded),
# so this script stays correct if configs/tracked_vehicle_drive_v1.json's
# families change -- see TRACKED_NUM_SHARDS below to repartition.

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" && -d "$SLURM_SUBMIT_DIR/src/nedm" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
  elif [[ -d /srv/home/hzhang699/NeDM/src/nedm ]]; then
    REPO_ROOT="/srv/home/hzhang699/NeDM"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi
fi
cd "$REPO_ROOT"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  module load conda/miniforge
  bootstrap-conda
  conda activate nedm
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "repo root: $REPO_ROOT"
echo "python: $PYTHON_BIN"

CONFIG="${TRACKED_CONFIG:-configs/tracked_vehicle_drive_v1.json}"
NUM_SHARDS="${TRACKED_NUM_SHARDS:-15}"
OUTPUT_ROOT="${TRACKED_OUTPUT_ROOT:-artifacts/datasets/tracked_vehicle_drive_v1_shards}"

TOTAL_SCENARIOS=$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from nedm.tracked_vehicle_data import load_config
config = load_config(Path(sys.argv[1]))
print(len(config["scenarios"]))
PY
)
SHARD_SIZE=$(( (TOTAL_SCENARIOS + NUM_SHARDS - 1) / NUM_SHARDS ))

echo "config: $CONFIG"
echo "total scenarios: $TOTAL_SCENARIOS across $NUM_SHARDS shards (shard size $SHARD_SIZE)"

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
  start_index=$((shard * SHARD_SIZE))

  if (( start_index >= TOTAL_SCENARIOS )); then
    echo "shard $shard starts past the end of the scenario list ($start_index >= $TOTAL_SCENARIOS); nothing to do"
    continue
  fi

  if [[ -f "$output_dir/dataset_index.json" ]]; then
    echo "shard $shard already complete at $output_dir; skipping"
    continue
  fi

  echo "collecting tracked-vehicle shard $shard -> $output_dir"
  echo "  start_index=$start_index max_scenarios=$SHARD_SIZE"

  "$PYTHON_BIN" scripts/collect_tracked_vehicle_dataset.py \
    --config "$CONFIG" \
    --output-dir "$output_dir" \
    --start-index "$start_index" \
    --max-scenarios "$SHARD_SIZE"

  echo "completed tracked-vehicle shard $shard -> $output_dir"
done

echo "done: ${shards[*]}"
