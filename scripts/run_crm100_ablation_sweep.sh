#!/usr/bin/env bash
# Sequential CRM-generalist improvement ablations, each a single-variable change off
# ...crm100_mix25_rebal_rollout (flat 15.4% / CRM 9.4% @ ep80):
#   combnorm = equal-domain combined input/output normalization
#   crm40    = 60/40 flat:CRM batch mix (vs 75/25)
#   vx3      = 3x extra loss weight on vel_body_x_mps
# Runs one at a time (shared GPU + ~17 GB dataset in RAM) so they do not contend.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate nedm

CONFIGS=(
  "configs/hmmwv_transformer_v07_tnf_omega_300g_crm100_combnorm.json"
  "configs/hmmwv_transformer_v07_tnf_omega_300g_crm100_crm40.json"
  "configs/hmmwv_transformer_v07_tnf_omega_300g_crm100_vx3.json"
)

log() { printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"; }

for cfg in "${CONFIGS[@]}"; do
  out="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$cfg")"
  mkdir -p "$out/logs"
  log "=== START $cfg -> $out ==="
  if [[ -f "$out/checkpoints/last.pt" ]]; then
    log "last.pt already exists for $out; resuming"
    resume=(--resume-from-checkpoint "$out/checkpoints/last.pt")
  else
    resume=()
  fi
  python scripts/train_hmmwv_dynamics.py --config "$cfg" --device cuda --output-dir "$out" \
    "${resume[@]}" >> "$out/logs/run.log" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    log "=== DONE  $cfg (exit 0) ==="
  else
    log "=== FAILED $cfg (exit $rc) — continuing to next ==="
  fi
done

log "ablation sweep complete"
