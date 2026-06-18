#!/usr/bin/env bash
# Launch the 2000-episode HMMWV CRM collection in tmux.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-hmmwv_crm2000_collection}"
PYTHON_BIN="${PYTHON_BIN:-/home/harry/anaconda3/envs/nedm/bin/python}"
PLAN_DIR="${PLAN_DIR:-artifacts/datasets/hmmwv_crm_2000_plan}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/datasets/hmmwv_crm_2000}"
PROCESSED_DIR="${PROCESSED_DIR:-artifacts/training_datasets/hmmwv_crm_2000_force_omega_seq_v1}"
DATASET_NAME="${DATASET_NAME:-hmmwv_crm_2000}"
CONFIG_NAME="${CONFIG_NAME:-crm2000}"
SCENARIO_PREFIX_ROOT="${SCENARIO_PREFIX_ROOT:-crm2000}"
EPISODES="${EPISODES:-2000}"
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
BUILD_PROCESSED="${BUILD_PROCESSED:-1}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$OUTPUT_DIR/logs"
cmd="cd $(printf '%q' "$REPO_ROOT") && env"
for kv in \
  "PYTHON_BIN=$PYTHON_BIN" \
  "PLAN_DIR=$PLAN_DIR" \
  "OUTPUT_DIR=$OUTPUT_DIR" \
  "PROCESSED_DIR=$PROCESSED_DIR" \
  "DATASET_NAME=$DATASET_NAME" \
  "CONFIG_NAME=$CONFIG_NAME" \
  "SCENARIO_PREFIX_ROOT=$SCENARIO_PREFIX_ROOT" \
  "EPISODES=$EPISODES" \
  "DURATION_MIN_S=$DURATION_MIN_S" \
  "DURATION_MAX_S=$DURATION_MAX_S" \
  "TERRAIN_LENGTH_M=$TERRAIN_LENGTH_M" \
  "TERRAIN_WIDTH_M=$TERRAIN_WIDTH_M" \
  "CRM_SPACING_M=$CRM_SPACING_M" \
  "BOUNDARY_MARGIN_M=$BOUNDARY_MARGIN_M" \
  "CHRONO_THREADS=$CHRONO_THREADS" \
  "PROGRESS_INTERVAL_S=$PROGRESS_INTERVAL_S" \
  "CHRONO_DATA_ROOT=$CHRONO_DATA_ROOT" \
  "OVERWRITE=$OVERWRITE" \
  "BUILD_PROCESSED=$BUILD_PROCESSED"; do
  cmd+=" $(printf '%q' "$kv")"
done
cmd+=" bash scripts/run_hmmwv_crm2000_collection.sh"

tmux new-session -d -s "$SESSION" "$cmd"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: $OUTPUT_DIR/logs/run.log"
