#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-artifacts/training_runs/hmmwv_sweep_v04_v18}"
LOG_DIR="$RUN_ROOT/logs"
SESSION_FILE="$RUN_ROOT/sweep.tmux_session"
SESSION_NAME="${SESSION_NAME:-hmmwv_sweep_v04_v18}"
RUN_LOG="$LOG_DIR/run.log"
RUN_LOG_ABS="$REPO_ROOT/$RUN_LOG"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "sweep already running in tmux session $SESSION_NAME"
  echo "log: $RUN_LOG"
  echo "status: $RUN_ROOT/status.json"
  echo "leaderboard: $RUN_ROOT/leaderboard.md"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$REPO_ROOT' && bash scripts/run_hmmwv_sweep_v04_v18.sh >> '$RUN_LOG_ABS' 2>&1"
echo "$SESSION_NAME" > "$SESSION_FILE"
echo "started HMMWV transformer sweep in tmux session $SESSION_NAME"
echo "log: $RUN_LOG"
echo "status: $RUN_ROOT/status.json"
echo "leaderboard: $RUN_ROOT/leaderboard.md"
echo "attach: tmux attach -t $SESSION_NAME"
