#!/bin/bash
#SBATCH --job-name=hmmwv-tire300g
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --partition=sbel
#SBATCH --time=02:00:00

# Collects the 300 GB rigid-terrain HMMWV tire-force dataset: 128 shards x
# 256 episodes (~2.4 GB per shard, ~307 GB total -> check your disk quota
# before launching). Intended usage is one shard per array task:
#
#   sbatch --array=0-127 scripts/cluster/collect_hmmwv_tire300g.sh        # as parallel as the partition allows
#   sbatch --array=0-127%16 scripts/cluster/collect_hmmwv_tire300g.sh    # at most 16 shards at a time
#
# A shard takes roughly 30 min on 32 cpus, so the 02:00:00 walltime has wide
# margin per array task. Running without --array loops over all 128 shards
# sequentially and will hit the walltime; that mode only makes sense for
# mopping up a few leftover shards, since completed shards (those with a
# dataset_index.json) are skipped and the job is safe to resubmit after a
# timeout or node failure.
#
# Before first submit:  mkdir -p logs   (slurm won't create the output dir)

set -euo pipefail

module load conda/miniforge
bootstrap-conda
conda activate nedm

cd /srv/home/hzhang699/NeDM

# chrono data ships with the pychrono conda package; the chrono source
# checkout is not needed on the cluster
CHRONO_DATA_ROOT="$CONDA_PREFIX/share/chrono/data"
PLAN_DIR="artifacts/datasets/hmmwv_tire_rigid_300g_plan"
NUM_SHARDS=128
JOBS="${SLURM_CPUS_PER_TASK:-16}"

# idempotent; writes shard configs pointing at this machine's chrono data
python scripts/prepare_hmmwv_tire300g_generation.py --chrono-data-root "$CHRONO_DATA_ROOT"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  shards=("$SLURM_ARRAY_TASK_ID")
else
  shards=($(seq 0 $((NUM_SHARDS - 1))))
fi

for shard in "${shards[@]}"; do
  config=$(printf '%s/configs/shard_%03d.json' "$PLAN_DIR" "$shard")
  output_dir=$(python - "$config" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1]).read())["output_subdir"])
PY
)

  if [[ -f "$output_dir/dataset_index.json" ]]; then
    echo "shard $shard already complete at $output_dir; skipping"
    continue
  fi

  echo "collecting shard $shard -> $output_dir (jobs=$JOBS)"
  python scripts/collect_hmmwv_dataset.py --config "$config" --jobs "$JOBS"

  echo "validating shard $shard"
  python scripts/validate_hmmwv_tire_dataset.py --dataset-dir "$output_dir"
done

echo "done: ${shards[*]}"
