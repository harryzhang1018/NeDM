#!/usr/bin/env bash
# Local end-to-end smoke test for the 10 GB tire-force dataset pipeline.
# Runs the real collector on a 12-episode smoke shard and validates the output.
#
# Usage:
#   PYTHON_BIN=/home/harry/anaconda3/envs/nedm/bin/python bash scripts/smoke_test_hmmwv_tire10g.sh
# PYTHON_BIN must be a python with pychrono installed (defaults to `python`).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
JOBS="${JOBS:-4}"
PLAN_DIR="${PLAN_DIR:-artifacts/datasets/hmmwv_tire_rigid_10g_plan}"
CHRONO_DATA_ROOT="${CHRONO_DATA_ROOT:-}"

prepare_args=(--plan-dir "$PLAN_DIR")
if [[ -n "$CHRONO_DATA_ROOT" ]]; then
  prepare_args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
fi

echo "[1/3] writing shard + smoke configs"
"$PYTHON_BIN" scripts/prepare_hmmwv_tire10g_generation.py "${prepare_args[@]}"

smoke_config="$PLAN_DIR/configs/smoke.json"
smoke_output=$("$PYTHON_BIN" - "$smoke_config" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1]).read())["output_subdir"])
PY
)

echo "[2/3] collecting smoke shard ($smoke_config, jobs=$JOBS)"
rm -rf "$smoke_output"
"$PYTHON_BIN" scripts/collect_hmmwv_dataset.py --config "$smoke_config" --jobs "$JOBS"

echo "[3/3] validating output"
"$PYTHON_BIN" scripts/validate_hmmwv_tire_dataset.py --dataset-dir "$smoke_output"

echo
echo "smoke test passed: $smoke_output"
