#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET_TOTAL_GB="${TARGET_TOTAL_GB:-300}"
JOBS="${JOBS:-16}"
NUM_SHARDS="${NUM_SHARDS:-96}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-1000}"
PLAN_DIR="${PLAN_DIR:-artifacts/datasets/hmmwv_turn_300g_plan}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/datasets/hmmwv_turn_300g_shards}"
STATUS_FILE="$OUTPUT_ROOT/status.json"

BASE_ROOTS=(
  "artifacts/datasets/hmmwv_overfit_6k"
  "artifacts/datasets/hmmwv_aggressive_steer_2k"
)

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

bytes_for_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    du -sb "$path" | awk '{print $1}'
  else
    printf '0\n'
  fi
}

total_bytes() {
  local total=0
  local path
  for path in "${BASE_ROOTS[@]}" "$OUTPUT_ROOT"; do
    total=$((total + $(bytes_for_path "$path")))
  done
  printf '%s\n' "$total"
}

write_status() {
  local state="$1"
  local shard="$2"
  local bytes="$3"
  python3 - "$STATUS_FILE" "$state" "$shard" "$bytes" "$TARGET_TOTAL_GB" "$JOBS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1])
status_path.parent.mkdir(parents=True, exist_ok=True)
bytes_value = int(sys.argv[4])
target_gib = float(sys.argv[5])
target_bytes = int(target_gib * 1024**3)
payload = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "state": sys.argv[2],
    "current_shard": sys.argv[3],
    "bytes": bytes_value,
    "gib": bytes_value / 1024**3,
    "target_total_gib": target_gib,
    "target_bytes": target_bytes,
    "progress_fraction": bytes_value / target_bytes if target_bytes else None,
    "jobs": int(sys.argv[6]),
}
status_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

log "preparing shard configs"
mkdir -p "$OUTPUT_ROOT"
python3 scripts/prepare_hmmwv_300g_generation.py \
  --plan-dir "$PLAN_DIR" \
  --output-root "$OUTPUT_ROOT" \
  --num-shards "$NUM_SHARDS" \
  --episodes-per-shard "$EPISODES_PER_SHARD"

TARGET_BYTES=$(python3 - "$TARGET_TOTAL_GB" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024**3))
PY
)

export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial

for config_path in "$PLAN_DIR"/configs/shard_*.json; do
  current_bytes=$(total_bytes)
  shard_name="$(basename "$config_path" .json)"
  write_status "running" "$shard_name" "$current_bytes"

  if (( current_bytes >= TARGET_BYTES )); then
    log "target reached before $shard_name: $((current_bytes / 1024 / 1024 / 1024)) GiB"
    write_status "complete" "$shard_name" "$current_bytes"
    exit 0
  fi

  output_dir=$(python3 - "$config_path" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg["output_subdir"])
PY
)

  if [[ -f "$output_dir/dataset_index.json" ]]; then
    log "skipping completed $shard_name"
    continue
  fi

  log "starting $shard_name with $JOBS jobs"
  python scripts/collect_hmmwv_dataset.py --config "$config_path" --jobs "$JOBS"
  current_bytes=$(total_bytes)
  log "finished $shard_name; total raw size is $(python3 - "$current_bytes" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024**3:.2f} GiB")
PY
)"
done

current_bytes=$(total_bytes)
write_status "complete_no_more_shards" "none" "$current_bytes"
log "all configured shards completed; final total $(python3 - "$current_bytes" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024**3:.2f} GiB")
PY
)"
