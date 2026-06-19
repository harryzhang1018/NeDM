#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_scratch.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_scratch}"
FLAT_PROCESSED_DIR="${FLAT_PROCESSED_DIR:-artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1}"
CRM_RAW_DIR="${CRM_RAW_DIR:-artifacts/datasets/hmmwv_crm_100}"
CRM_PROCESSED_DIR="${CRM_PROCESSED_DIR:-artifacts/training_datasets/hmmwv_crm_100_normal_force_omega_seq_v1}"
STATE_FIELD_PRESET="${STATE_FIELD_PRESET:-tire_normal_force_omega}"
RESUME="${RESUME:-1}"
STATUS_FILE="$OUTPUT_DIR/status.json"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

write_status() {
  local state="$1"
  local stage="$2"
  local message="$3"
  mkdir -p "$OUTPUT_DIR"
  python - "$STATUS_FILE" "$state" "$stage" "$message" <<'PY'
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
  local processed_dir="$1"
  python - "$processed_dir" "$STATE_FIELD_PRESET" <<'PY'
import json
import sys
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "src"))

from nedm.training.constants import DEFAULT_ACTION_FIELDS, DEFAULT_ROLLOUT_FIELDS, STATE_FIELD_PRESETS

processed_dir = Path(sys.argv[1])
preset = sys.argv[2]
metadata = json.loads((processed_dir / "metadata.json").read_text())
expected_states = list(STATE_FIELD_PRESETS[preset])
actual_states = list(metadata["state_fields"])
if actual_states != expected_states:
    raise SystemExit(
        f"{processed_dir} has unexpected state fields: {actual_states}; expected {expected_states}"
    )
if list(metadata["action_fields"]) != list(DEFAULT_ACTION_FIELDS):
    raise SystemExit(f"{processed_dir} has unexpected action fields: {metadata['action_fields']}")
if list(metadata["rollout_fields"]) != list(DEFAULT_ROLLOUT_FIELDS):
    raise SystemExit(f"{processed_dir} has unexpected rollout fields: {metadata['rollout_fields']}")
if abs(float(metadata["dt_s"]) - 0.01) > 1e-12:
    raise SystemExit(f"{processed_dir} has dt_s={metadata['dt_s']}, expected 0.01")
print(
    f"verified {processed_dir}: {len(actual_states)} state fields, "
    f"train={metadata['splits']['train']['transition_count']} transitions, "
    f"val={metadata['splits']['val']['transition_count']} transitions"
)
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

log "activating conda environment: nedm"
export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate nedm

if [[ ! -f "$FLAT_PROCESSED_DIR/metadata.json" ]]; then
  log "flat processed cache missing: $FLAT_PROCESSED_DIR"
  exit 1
fi

if [[ ! -f "$CRM_PROCESSED_DIR/metadata.json" ]]; then
  current_stage="preprocess_crm"
  write_status "running" "$current_stage" "building CRM normal-force/omega processed cache"
  log "building $CRM_PROCESSED_DIR from $CRM_RAW_DIR"
  python scripts/build_hmmwv_training_dataset.py \
    --dataset-root "$CRM_RAW_DIR" \
    --output-dir "$CRM_PROCESSED_DIR" \
    --state-field-preset "$STATE_FIELD_PRESET" \
    --disk-backed-arrays
else
  log "using existing CRM processed cache at $CRM_PROCESSED_DIR"
fi

current_stage="verify"
write_status "running" "$current_stage" "verifying flat and CRM processed caches"
verify_processed_fields "$FLAT_PROCESSED_DIR"
verify_processed_fields "$CRM_PROCESSED_DIR"

current_stage="training"
write_status "running" "$current_stage" "training from scratch with fixed 75% flat / 25% CRM batches"
log "training scratch flat/CRM mixed model with $CONFIG"
train_args=(scripts/train_hmmwv_dynamics.py --config "$CONFIG" --device cuda --output-dir "$OUTPUT_DIR")
if [[ "$RESUME" == "1" && -f "$OUTPUT_DIR/checkpoints/last.pt" ]]; then
  log "resuming same scratch run from $OUTPUT_DIR/checkpoints/last.pt"
  train_args+=(--resume-from-checkpoint "$OUTPUT_DIR/checkpoints/last.pt")
elif [[ "$RESUME" != "1" && -f "$OUTPUT_DIR/checkpoints/last.pt" ]]; then
  log "last checkpoint exists and RESUME=$RESUME; choose a new OUTPUT_DIR or set RESUME=1"
  exit 1
fi
python "${train_args[@]}"

current_stage="complete"
write_status "complete" "$current_stage" "training completed"
log "training completed"
