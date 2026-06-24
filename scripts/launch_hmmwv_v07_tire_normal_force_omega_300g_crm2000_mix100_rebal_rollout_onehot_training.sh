#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix100_rebal_rollout_onehot}"
LOG_DIR="$OUTPUT_DIR/logs"
SESSION_FILE="$OUTPUT_DIR/training.tmux_session"
SESSION_NAME="${SESSION_NAME:-hmmwv_v07_crm2000_mix100_rebal_rollout_onehot_training}"
RUN_LOG="$LOG_DIR/run.log"
RUN_LOG_ABS="$REPO_ROOT/$RUN_LOG"

mkdir -p "$LOG_DIR"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "training already running in tmux session $SESSION_NAME"
    echo "log: $RUN_LOG"
    echo "status: $OUTPUT_DIR/status.json"
    exit 0
  fi

  tmux new-session -d -s "$SESSION_NAME" "cd '$REPO_ROOT' && bash scripts/run_hmmwv_v07_tire_normal_force_omega_300g_crm2000_mix100_rebal_rollout_onehot_training.sh >> '$RUN_LOG_ABS' 2>&1"
  echo "$SESSION_NAME" > "$SESSION_FILE"
  echo "started HMMWV v07 rebalanced-loss + rollout-selection training in tmux session $SESSION_NAME"
  echo "log: $RUN_LOG"
  echo "status: $OUTPUT_DIR/status.json"
  echo "attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

echo "tmux is required for this long-running training job" >&2
exit 1
