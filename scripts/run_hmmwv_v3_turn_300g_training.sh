#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/hmmwv_transformer_v3_turn_300g.json}"
PROCESSED_DIR="${PROCESSED_DIR:-artifacts/training_datasets/hmmwv_turn_300g_plus_base_seq_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training_runs/hmmwv_transformer_v3_turn_300g}"
SHARD_ROOT="${SHARD_ROOT:-artifacts/datasets/hmmwv_turn_300g_shards}"
STATUS_FILE="$OUTPUT_DIR/status.json"

BASE_ROOTS=(
  "artifacts/datasets/hmmwv_overfit_6k"
  "artifacts/datasets/hmmwv_aggressive_steer_2k"
)

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

write_status() {
  local state="$1"
  local stage="$2"
  local message="$3"
  mkdir -p "$OUTPUT_DIR"
  python3 - "$STATUS_FILE" "$state" "$stage" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
payload = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "state": sys.argv[2],
    "stage": sys.argv[3],
    "message": sys.argv[4],
}
status_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

current_stage="starting"
on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -ne 0 ]]; then
    write_status "failed" "$current_stage" "exit_code=$exit_code"
  fi
}
trap on_exit EXIT

log "activating conda environment"
export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial

shard_roots=()
for shard_dir in "$SHARD_ROOT"/shard_*; do
  if [[ -f "$shard_dir/dataset_index.json" ]]; then
    shard_roots+=("$shard_dir")
  fi
done

if [[ "${#shard_roots[@]}" -eq 0 ]]; then
  log "no completed shard roots found under $SHARD_ROOT"
  exit 1
fi

if [[ ! -f "$PROCESSED_DIR/metadata.json" ]]; then
  current_stage="preprocess"
  write_status "running" "$current_stage" "building processed dataset cache"
  log "building processed dataset cache at $PROCESSED_DIR"
  log "raw roots: ${#BASE_ROOTS[@]} base roots + ${#shard_roots[@]} generated shards"
  python scripts/build_hmmwv_training_dataset.py \
    --dataset-root "${BASE_ROOTS[@]}" "${shard_roots[@]}" \
    --output-dir "$PROCESSED_DIR"
else
  log "using existing processed dataset cache at $PROCESSED_DIR"
fi

current_stage="training"
write_status "running" "$current_stage" "training v3 model"
log "training v3 model with $CONFIG"
python scripts/train_hmmwv_dynamics.py --config "$CONFIG" --device cuda

current_stage="complete"
write_status "complete" "$current_stage" "training completed"
log "training completed"
