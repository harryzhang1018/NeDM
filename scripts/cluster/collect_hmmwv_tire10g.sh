#!/bin/bash
#SBATCH --job-name=hmmwv-tire10g
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --partition=sbel
#SBATCH --time=02:00:00

# Collects the 10 GB rigid-terrain HMMWV tire-force dataset (4 shards x 256
# episodes). Two ways to run:
#
#   sbatch scripts/cluster/collect_hmmwv_tire10g.sh                 # one job, all 4 shards in sequence
#   sbatch --array=0-3 scripts/cluster/collect_hmmwv_tire10g.sh    # one shard per array task
#
# Before first submit:  mkdir -p logs   (slurm won't create the output dir)
# Shards already completed (dataset_index.json present) are skipped, so the
# job is safe to resubmit after a timeout or node failure.

set -euo pipefail

module load conda/miniforge
bootstrap-conda
conda activate nedm

cd /srv/home/haoruiz/NeDM

# chrono data ships with the pychrono conda package; the chrono source
# checkout is not needed on the cluster
CHRONO_DATA_ROOT="$CONDA_PREFIX/share/chrono/data"
PLAN_DIR="artifacts/datasets/hmmwv_tire_rigid_10g_plan"
JOBS="${SLURM_CPUS_PER_TASK:-16}"

# idempotent; writes shard configs pointing at this machine's chrono data
python scripts/prepare_hmmwv_tire10g_generation.py --chrono-data-root "$CHRONO_DATA_ROOT"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  shards=("$SLURM_ARRAY_TASK_ID")
else
  shards=(0 1 2 3)
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
