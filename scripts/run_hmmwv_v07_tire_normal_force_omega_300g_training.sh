#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/hmmwv_transformer_v07_tire_normal_force_omega_300g.json}"
SOURCE_PROCESSED_DIR="${SOURCE_PROCESSED_DIR:-artifacts/training_datasets/hmmwv_tire_rigid_300g_force_omega_seq_v1}"
PROCESSED_DIR="${PROCESSED_DIR:-artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g}"
SHARD_ROOT="${SHARD_ROOT:-artifacts/datasets/hmmwv_tire_rigid_300g_shards}"
STATE_FIELD_PRESET="${STATE_FIELD_PRESET:-tire_normal_force_omega}"
STATUS_FILE="$OUTPUT_DIR/status.json"

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

verify_processed_fields() {
  python - "$PROCESSED_DIR" "$STATE_FIELD_PRESET" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "src"))

from nedm.training.constants import STATE_FIELD_PRESETS

processed_dir = Path(sys.argv[1])
preset = sys.argv[2]
metadata = json.loads((processed_dir / "metadata.json").read_text())
expected = list(STATE_FIELD_PRESETS[preset])
actual = list(metadata["state_fields"])
if actual != expected:
    raise SystemExit(
        f"{processed_dir} has unexpected state fields: {actual}; expected {expected}"
    )
print(f"verified {processed_dir}: {len(actual)} state fields")
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

if [[ ! -f "$PROCESSED_DIR/metadata.json" ]]; then
  current_stage="preprocess"
  if [[ -f "$SOURCE_PROCESSED_DIR/metadata.json" ]]; then
    write_status "running" "$current_stage" "building normal-force/omega cache from existing force/omega cache"
    log "building $PROCESSED_DIR from $SOURCE_PROCESSED_DIR with preset $STATE_FIELD_PRESET"
    python scripts/build_hmmwv_state_subset_dataset.py \
      --source-dir "$SOURCE_PROCESSED_DIR" \
      --output-dir "$PROCESSED_DIR" \
      --state-field-preset "$STATE_FIELD_PRESET"
  else
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

    write_status "running" "$current_stage" "building normal-force/omega processed dataset cache from raw shards"
    log "source processed cache missing; building from ${#shard_roots[@]} raw tire shards"
    python scripts/build_hmmwv_training_dataset.py \
      --dataset-root "${shard_roots[@]}" \
      --output-dir "$PROCESSED_DIR" \
      --state-field-preset "$STATE_FIELD_PRESET" \
      --disk-backed-arrays
  fi
else
  log "using existing processed dataset cache at $PROCESSED_DIR"
fi

verify_processed_fields

current_stage="training"
write_status "running" "$current_stage" "training v07 tire-normal-force/omega model"
log "training v07 tire-normal-force/omega model with $CONFIG"
train_args=(scripts/train_hmmwv_dynamics.py --config "$CONFIG" --device cuda)
if [[ -f "$OUTPUT_DIR/checkpoints/last.pt" ]]; then
  log "resuming from $OUTPUT_DIR/checkpoints/last.pt"
  train_args+=(--resume-from-checkpoint "$OUTPUT_DIR/checkpoints/last.pt")
fi
python "${train_args[@]}"

current_stage="complete"
write_status "complete" "$current_stage" "training completed"
log "training completed"
