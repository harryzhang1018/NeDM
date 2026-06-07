#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RECIPES="${RECIPES:-configs/hmmwv_transformer_sweep_v04_v18.json}"
DEVICE="${DEVICE:-cuda}"

export CONDA_NO_PLUGINS=true
export PYTHONUNBUFFERED=1
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial

python scripts/run_hmmwv_transformer_sweep.py --recipes "$RECIPES" --device "$DEVICE" "$@"
