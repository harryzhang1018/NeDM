#!/usr/bin/env bash
# Resume the HMMWV CRM2000 collection in shorter Chrono processes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
    return 0
  fi
  local candidate
  for candidate in \
    "/home/harry/miniconda3/envs/nedm/bin/python" \
    "/home/harry/anaconda3/envs/nedm/bin/python"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if command -v python >/dev/null 2>&1 && python -c 'import pychrono' >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "Could not find a Python with pychrono. Set PYTHON_BIN=/path/to/nedm/bin/python." >&2
  return 1
}

PYTHON_BIN="$(resolve_python_bin)"
PLAN_DIR="${PLAN_DIR:-artifacts/datasets/hmmwv_crm_2000_plan}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/datasets/hmmwv_crm_2000}"
PROCESSED_DIR="${PROCESSED_DIR:-artifacts/training_datasets/hmmwv_crm_2000_force_omega_seq_v1}"
CONFIG_NAME="${CONFIG_NAME:-crm2000}"
CONFIG_PATH="$PLAN_DIR/configs/${CONFIG_NAME%.json}.json"
TOTAL_EPISODES="${EPISODES:-2000}"
START_INDEX="${START_INDEX:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PROGRESS_INTERVAL_S="${PROGRESS_INTERVAL_S:-5.0}"
BUILD_PROCESSED="${BUILD_PROCESSED:-1}"
CHRONO_DATA_ROOT="${CHRONO_DATA_ROOT:-}"
LOG_DIR="$OUTPUT_DIR/logs"
LOG_PATH="$LOG_DIR/remaining_chunked.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR/episodes"
exec > >(tee -a "$LOG_PATH") 2>&1

collect_extra_args=()
if [[ -n "$CHRONO_DATA_ROOT" ]]; then
  collect_extra_args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
fi

count_completed() {
  "$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
count = 0
for sidecar in (root / "episodes").glob("*.json"):
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    csv_path = root / meta.get("csv_path", "")
    if csv_path.is_file() and int(meta.get("rows", 0)) > 0:
        count += 1
print(count)
PY
}

rebuild_index() {
  "$PYTHON_BIN" - "$CONFIG_PATH" "$OUTPUT_DIR" "$CHRONO_DATA_ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from collect_hmmwv_crm_dataset import load_config

config_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])
chrono_data_root = sys.argv[3] or None
config = load_config(config_path, chrono_data_root=chrono_data_root)

episodes = []
for scenario in config["scenarios"]:
    sidecar_path = output_root / "episodes" / f"{scenario['name']}.json"
    if not sidecar_path.is_file():
        continue
    try:
        meta = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    csv_path = output_root / meta.get("csv_path", "")
    if not csv_path.is_file() or int(meta.get("rows", 0)) <= 0:
        continue
    episodes.append(
        {
            "episode_id": str(meta["episode_id"]),
            "scenario_name": str(meta["scenario_name"]),
            "scenario_family": str(meta["scenario_family"]),
            "split": str(meta["split"]),
            "csv_path": str(csv_path.relative_to(output_root)),
            "rows": int(meta["rows"]),
            "duration_s": float(meta["duration_s"]),
            "warmup_s": float(meta["warmup_s"]),
            "terminated_near_boundary": bool(meta.get("terminated_near_boundary", False)),
        }
    )

summary = {
    "dataset_name": config["dataset_name"],
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "episode_count": len(episodes),
    "episodes": episodes,
}
(output_root / "dataset_index.json").write_text(json.dumps(summary, indent=2) + "\n")
(output_root / "collector_config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
print(len(episodes))
PY
}

echo "started_at=$(date --iso-8601=seconds)"
echo "python=$PYTHON_BIN"
echo "config=$CONFIG_PATH output=$OUTPUT_DIR processed=$PROCESSED_DIR"
echo "total=$TOTAL_EPISODES start_index=$START_INDEX chunk_size=$CHUNK_SIZE max_retries=$MAX_RETRIES build_processed=$BUILD_PROCESSED"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "missing config: $CONFIG_PATH" >&2
  echo "Run scripts/run_hmmwv_crm2000_collection.sh once to generate the plan." >&2
  exit 1
fi

completed="$(count_completed)"
echo "completed_before=$completed"

chunk_start="$START_INDEX"
while (( chunk_start < TOTAL_EPISODES )); do
  completed="$(count_completed)"
  if (( completed >= TOTAL_EPISODES )); then
    break
  fi

  if (( chunk_start < completed )); then
    chunk_start="$completed"
  fi

  remaining=$(( TOTAL_EPISODES - chunk_start ))
  chunk_count="$CHUNK_SIZE"
  if (( remaining < chunk_count )); then
    chunk_count="$remaining"
  fi

  attempt=1
  while true; do
    echo "chunk_start=$chunk_start chunk_count=$chunk_count attempt=$attempt completed=$completed"
    if "$PYTHON_BIN" scripts/collect_hmmwv_crm_dataset.py \
      --config "$CONFIG_PATH" \
      --start-index "$chunk_start" \
      --max-scenarios "$chunk_count" \
      --progress-interval-s "$PROGRESS_INTERVAL_S" \
      --resume \
      "${collect_extra_args[@]}"; then
      rebuilt="$(rebuild_index)"
      echo "chunk_done start=$chunk_start count=$chunk_count indexed_episodes=$rebuilt"
      break
    fi

    if (( attempt >= MAX_RETRIES )); then
      echo "chunk_failed start=$chunk_start count=$chunk_count attempts=$attempt" >&2
      rebuild_index >/dev/null
      exit 1
    fi
    attempt=$(( attempt + 1 ))
    sleep 10
  done

  chunk_start=$(( chunk_start + chunk_count ))
done

completed="$(rebuild_index)"
echo "completed_after=$completed"
if (( completed < TOTAL_EPISODES )); then
  echo "collection incomplete: completed $completed / $TOTAL_EPISODES" >&2
  exit 1
fi

if [[ "$BUILD_PROCESSED" == "1" ]]; then
  "$PYTHON_BIN" scripts/build_hmmwv_training_dataset.py \
    --dataset-root "$OUTPUT_DIR" \
    --output-dir "$PROCESSED_DIR" \
    --state-field-preset tire_force_omega \
    --disk-backed-arrays
else
  echo "skipped processed cache build because BUILD_PROCESSED=$BUILD_PROCESSED"
fi

echo "finished_at=$(date --iso-8601=seconds)"
