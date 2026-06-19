#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_NAME="${SESSION_NAME:-hmmwv_crm100_ablation_sweep}"
SWEEP_LOG="$REPO_ROOT/artifacts/training_runs/crm100_ablation_sweep.log"
mkdir -p "$REPO_ROOT/artifacts/training_runs"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "sweep already running in tmux session $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" \
  "cd '$REPO_ROOT' && bash scripts/run_crm100_ablation_sweep.sh >> '$SWEEP_LOG' 2>&1"
echo "started CRM100 ablation sweep in tmux session $SESSION_NAME"
echo "sweep log: $SWEEP_LOG"
echo "per-run logs: artifacts/training_runs/hmmwv_transformer_v07_tnf_omega_300g_crm100_{combnorm,crm40,vx3}/logs/run.log"
echo "attach: tmux attach -t $SESSION_NAME"
