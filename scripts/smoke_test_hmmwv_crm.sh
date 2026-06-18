#!/usr/bin/env bash
# Local visual smoke test for HMMWV CRM terrain collection.
# Opens the CRM/VSG visualization by default while logging a rigid-compatible
# episode CSV, sidecar JSON, collector_config.resolved.json, and dataset_index.json.
#
# Usage:
#   bash scripts/smoke_test_hmmwv_crm.sh
#   RENDER=0 bash scripts/smoke_test_hmmwv_crm.sh
#   CAMERA_STATE=track CAMERA_X_M=0 CAMERA_Y_M=-14 CAMERA_Z_M=7 bash scripts/smoke_test_hmmwv_crm.sh
#   CAMERA_X_M=0 CAMERA_Y_M=-14 CAMERA_Z_M=7 bash scripts/smoke_test_hmmwv_crm.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/harry/anaconda3/envs/nedm/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/datasets/hmmwv_crm_smoke}"
RENDER="${RENDER:-1}"
DURATION_S="${DURATION_S:-15.0}"
WARMUP_S="${WARMUP_S:-0.2}"
CRM_SPACING_M="${CRM_SPACING_M:-0.08}"
TERRAIN_LENGTH_M="${TERRAIN_LENGTH_M:-150.0}"
TERRAIN_WIDTH_M="${TERRAIN_WIDTH_M:-150.0}"
BOUNDARY_MARGIN_M="${BOUNDARY_MARGIN_M:-5.0}"
CHRONO_THREADS="${CHRONO_THREADS:-12}"
CAMERA_TRACK_Z_M="${CAMERA_TRACK_Z_M:-1.2}"
CAMERA_X_M="${CAMERA_X_M:-7.0}"
CAMERA_Y_M="${CAMERA_Y_M:--9.0}"
CAMERA_Z_M="${CAMERA_Z_M:-5.0}"
CAMERA_DISTANCE_M="${CAMERA_DISTANCE_M:-11.0}"
CAMERA_HEIGHT_M="${CAMERA_HEIGHT_M:-4.0}"
CAMERA_ANGLE_DEG="${CAMERA_ANGLE_DEG:-150.0}"
CAMERA_STATE="${CAMERA_STATE:-chase}"
CAMERA_TARGET_X_M="${CAMERA_TARGET_X_M:-}"
CAMERA_TARGET_Y_M="${CAMERA_TARGET_Y_M:-0.0}"
CAMERA_TARGET_Z_M="${CAMERA_TARGET_Z_M:-1.2}"
DRIVER_PROFILE="${DRIVER_PROFILE:-launch_brake}"
THROTTLE_PEAK="${THROTTLE_PEAK:-${THROTTLE_MAX:-0.55}}"
THROTTLE_DELAY_S="${THROTTLE_DELAY_S:-0.2}"
THROTTLE_RISE_S="${THROTTLE_RISE_S:-1.0}"
THROTTLE_HOLD_S="${THROTTLE_HOLD_S:-5.0}"
THROTTLE_RELEASE_S="${THROTTLE_RELEASE_S:-1.0}"
BRAKE_PEAK="${BRAKE_PEAK:-0.35}"
BRAKE_DELAY_S="${BRAKE_DELAY_S:-0.8}"
BRAKE_RISE_S="${BRAKE_RISE_S:-0.5}"
BRAKE_HOLD_S="${BRAKE_HOLD_S:-1.5}"
BRAKE_RELEASE_S="${BRAKE_RELEASE_S:-1.0}"
STEERING_AMPLITUDE="${STEERING_AMPLITUDE:-0.0}"
STEERING_FREQUENCY_HZ="${STEERING_FREQUENCY_HZ:-0.08}"
PROGRESS_INTERVAL_S="${PROGRESS_INTERVAL_S:-1.0}"
CHRONO_DATA_ROOT="${CHRONO_DATA_ROOT:-}"

if [[ "$RENDER" != "0" && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "RENDER requested, but no DISPLAY or WAYLAND_DISPLAY is available; continuing headless."
  echo "Run this script from an interactive desktop shell to open the VSG window."
  RENDER=0
fi

args=(
  --output-dir "$OUTPUT_DIR"
  --duration-s "$DURATION_S"
  --warmup-s "$WARMUP_S"
  --crm-spacing-m "$CRM_SPACING_M"
  --terrain-length-m "$TERRAIN_LENGTH_M"
  --terrain-width-m "$TERRAIN_WIDTH_M"
  --boundary-margin-m "$BOUNDARY_MARGIN_M"
  --chrono-threads "$CHRONO_THREADS"
  --camera-track-z-m "$CAMERA_TRACK_Z_M"
  --camera-x-m "$CAMERA_X_M"
  --camera-y-m "$CAMERA_Y_M"
  --camera-z-m "$CAMERA_Z_M"
  --camera-distance-m "$CAMERA_DISTANCE_M"
  --camera-height-m "$CAMERA_HEIGHT_M"
  --camera-angle-deg "$CAMERA_ANGLE_DEG"
  --camera-state "$CAMERA_STATE"
  --camera-target-y-m "$CAMERA_TARGET_Y_M"
  --camera-target-z-m "$CAMERA_TARGET_Z_M"
  --driver-profile "$DRIVER_PROFILE"
  --throttle-peak "$THROTTLE_PEAK"
  --throttle-delay-s "$THROTTLE_DELAY_S"
  --throttle-rise-s "$THROTTLE_RISE_S"
  --throttle-hold-s "$THROTTLE_HOLD_S"
  --throttle-release-s "$THROTTLE_RELEASE_S"
  --brake-peak "$BRAKE_PEAK"
  --brake-delay-s "$BRAKE_DELAY_S"
  --brake-rise-s "$BRAKE_RISE_S"
  --brake-hold-s "$BRAKE_HOLD_S"
  --brake-release-s "$BRAKE_RELEASE_S"
  --steering-amplitude "$STEERING_AMPLITUDE"
  --steering-frequency-hz "$STEERING_FREQUENCY_HZ"
  --progress-interval-s "$PROGRESS_INTERVAL_S"
  --overwrite
)

if [[ -n "$CAMERA_TARGET_X_M" ]]; then
  args+=(--camera-target-x-m "$CAMERA_TARGET_X_M")
fi

if [[ "$RENDER" != "0" ]]; then
  echo "VSG camera: CAMERA_STATE=$CAMERA_STATE, angle=${CAMERA_ANGLE_DEG}deg, height=${CAMERA_HEIGHT_M}m."
  echo "Controls: Left/Right rotate, Up/Down zoom in Chase mode; press 1 Chase, 2 Follow, 3 Track, 4 Inside, 5 Free."
  args+=(--render)
fi

if [[ -n "$CHRONO_DATA_ROOT" ]]; then
  args+=(--chrono-data-root "$CHRONO_DATA_ROOT")
fi

"$PYTHON_BIN" scripts/collect_hmmwv_crm_smoke.py "${args[@]}"

echo
echo "CRM smoke dataset: $OUTPUT_DIR"
