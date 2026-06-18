#!/usr/bin/env bash
# Prepare and collect a 100-episode HMMWV CRM dataset, then build the
# tire_force_omega processed cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/harry/anaconda3/envs/nedm/bin/python}"
PLAN_DIR="${PLAN_DIR:-artifacts/datasets/hmmwv_crm_100_plan}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/datasets/hmmwv_crm_100}"
PROCESSED_DIR="${PROCESSED_DIR:-artifacts/training_datasets/hmmwv_crm_100_force_omega_seq_v1}"
EPISODES="${EPISODES:-100}"
DURATION_MIN_S="${DURATION_MIN_S:-12.0}"
DURATION_MAX_S="${DURATION_MAX_S:-18.0}"
TERRAIN_LENGTH_M="${TERRAIN_LENGTH_M:-150.0}"
TERRAIN_WIDTH_M="${TERRAIN_WIDTH_M:-150.0}"
CRM_SPACING_M="${CRM_SPACING_M:-0.08}"
BOUNDARY_MARGIN_M="${BOUNDARY_MARGIN_M:-5.0}"
CHRONO_THREADS="${CHRONO_THREADS:-12}"
PROGRESS_INTERVAL_S="${PROGRESS_INTERVAL_S:-5.0}"
CHRONO_DATA_ROOT="${CHRONO_DATA_ROOT:-}"
OVERWRITE="${OVERWRITE:-0}"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/run.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "python=$PYTHON_BIN"
echo "episodes=$EPISODES terrain=${TERRAIN_LENGTH_M}x${TERRAIN_WIDTH_M} spacing=$CRM_SPACING_M chrono_threads=$CHRONO_THREADS"
echo "output=$OUTPUT_DIR processed=$PROCESSED_DIR"

prepare_args=(
  --plan-dir "$PLAN_DIR"
  --output-dir "$OUTPUT_DIR"
  --episodes "$EPISODES"
  --duration-min-s "$DURATION_MIN_S"
  --duration-max-s "$DURATION_MAX_S"
  --terrain-length-m "$TERRAIN_LENGTH_M"
  --terrain-width-m "$TERRAIN_WIDTH_M"
  --crm-spacing-m "$CRM_SPACING_M"
  --boundary-margin-m "$BOUNDARY_MARGIN_M"
  --chrono-threads "$CHRONO_THREADS"
)

collect_args=(
  --config "$PLAN_DIR/configs/crm100.json"
  --progress-interval-s "$PROGRESS_INTERVAL_S"
  --resume
)

if [[ -n "$CHRONO_DATA_ROOT" ]]; then
  prepare_args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
  collect_args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
fi

if [[ "$OVERWRITE" == "1" ]]; then
  collect_args=(--config "$PLAN_DIR/configs/crm100.json" --progress-interval-s "$PROGRESS_INTERVAL_S" --overwrite)
  if [[ -n "$CHRONO_DATA_ROOT" ]]; then
    collect_args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
  fi
fi

"$PYTHON_BIN" scripts/prepare_hmmwv_crm100_generation.py "${prepare_args[@]}"
"$PYTHON_BIN" scripts/collect_hmmwv_crm_dataset.py "${collect_args[@]}"
"$PYTHON_BIN" scripts/build_hmmwv_training_dataset.py \
  --dataset-root "$OUTPUT_DIR" \
  --output-dir "$PROCESSED_DIR" \
  --state-field-preset tire_force_omega \
  --disk-backed-arrays

echo "finished_at=$(date --iso-8601=seconds)"
