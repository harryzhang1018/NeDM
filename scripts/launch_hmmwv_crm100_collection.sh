#!/usr/bin/env bash
# Launch the 100-episode HMMWV CRM collection in tmux.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-hmmwv_crm100_collection}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

mkdir -p artifacts/datasets/hmmwv_crm_100/logs
tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && bash scripts/run_hmmwv_crm100_collection.sh"
echo "started tmux session: $SESSION"
echo "attach: tmux attach -t $SESSION"
echo "log: artifacts/datasets/hmmwv_crm_100/logs/run.log"
