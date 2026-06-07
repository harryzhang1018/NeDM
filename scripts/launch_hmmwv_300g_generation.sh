#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/datasets/hmmwv_turn_300g_shards}"
LOG_DIR="$OUTPUT_ROOT/logs"
PID_FILE="$OUTPUT_ROOT/generation.pid"
SESSION_FILE="$OUTPUT_ROOT/generation.tmux_session"
SESSION_NAME="${SESSION_NAME:-hmmwv_300g_generation}"
RUN_LOG="$LOG_DIR/run.log"
RUN_LOG_ABS="$REPO_ROOT/$RUN_LOG"

mkdir -p "$LOG_DIR"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "generation already running in tmux session $SESSION_NAME"
    echo "log: $RUN_LOG"
    exit 0
  fi

  tmux new-session -d -s "$SESSION_NAME" "cd '$REPO_ROOT' && bash scripts/run_hmmwv_300g_generation.sh >> '$RUN_LOG_ABS' 2>&1"
  echo "$SESSION_NAME" > "$SESSION_FILE"
  echo "started HMMWV 300G generation in tmux session $SESSION_NAME"
  echo "log: $RUN_LOG"
  echo "status: $OUTPUT_ROOT/status.json"
  echo "attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "generation already running with PID $existing_pid"
    echo "log: $RUN_LOG"
    exit 0
  fi
fi

nohup bash scripts/run_hmmwv_300g_generation.sh >> "$RUN_LOG" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$PID_FILE"

echo "started HMMWV 300G generation with PID $pid"
echo "log: $RUN_LOG"
echo "status: $OUTPUT_ROOT/status.json"
