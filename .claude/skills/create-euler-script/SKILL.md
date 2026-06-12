---
name: create-euler-script
description: Create or modify SLURM sbatch scripts for the user's Euler cluster (UW-Madison), especially NeDM dataset-collection array jobs. Use when asked to set up cluster data collection, write an sbatch/SLURM script, scale a dataset run, or pick partitions/CPUs/array throttles on Euler.
---

# Create Euler sbatch scripts (NeDM dataset collection)

## Cluster facts

- Login nodes `euler-login-N`; repo lives at `/srv/home/hzhang699/NeDM` (home dir is `hzhang699`).
- Env bootstrap (exactly this, in this order):
  ```bash
  module load conda/miniforge
  bootstrap-conda
  conda activate nedm
  ```
- The `nedm` env has python 3.12 + pychrono 10.0.0 + numpy. Chrono data ships
  with the conda package at `$CONDA_PREFIX/share/chrono/data` — no chrono
  source checkout exists on the cluster. Prepare scripts must receive
  `--chrono-data-root "$CONDA_PREFIX/share/chrono/data"`.
- Partitions: `sbel` (lab partition, only 144 CPUs total — small jobs only)
  and `research` (~6,200 CPUs — use for big array jobs). Check free capacity
  with `sinfo -p <partition> -o "%C"` → prints Allocated/Idle/Other/Total.
- `logs/` must exist before the first submit (`mkdir -p logs`); SLURM will
  not create the `--output` directory. A `4294967294` array id in a log
  filename means the job ran without `--array`.
- sbatch CLI flags override `#SBATCH` headers, e.g.
  `sbatch --partition=sbel --cpus-per-task=16 --mem=16G ...`.
- Workflow to deliver changes: commit + push from this machine, then give the
  user `cd /srv/home/hzhang699/NeDM && git pull && sbatch ...`. The user runs
  sbatch themselves; never submit jobs from here.

## Dataset pipeline being scripted

Three stages, all repo-relative:

1. `python scripts/prepare_hmmwv_<name>_generation.py --chrono-data-root ...`
   — idempotent; writes per-shard configs + `manifest.json` under
   `artifacts/datasets/<name>_plan/`. New prepare scripts import
   `base_config/family_config/family_counts/speed_band` from
   `prepare_hmmwv_300g_generation.py` (speed band rotates `shard_index % 4`:
   low/medium/fast/mixed).
2. `python scripts/collect_hmmwv_dataset.py --config <shard.json> --jobs N`
   — one process per episode; jobs should equal `SLURM_CPUS_PER_TASK`.
   Writes `dataset_index.json` on completion (the resumability marker).
3. `python scripts/validate_hmmwv_tire_dataset.py --dataset-dir <shard_dir>`
   — slip sanity applies only to wheels in contact (Fz > 50 N); airborne slip
   is reported but never a failure.

Existing pipelines (use as templates): `collect_hmmwv_tire10g.sh` (flat,
4×256 eps), `collect_hmmwv_tire300g.sh` (flat, 128×256), and
`collect_hmmwv_bumpy10g.sh` (heightmap; requires `assets/bumpy_terrain/`
from git and checks for it before running).

## Sizing and scheduling math

- One 256-episode shard with tire channels (105 CSV columns) ≈ **2.4 GB** and
  ≈ **30 min on 32 CPUs** (scales ~linearly with CPUs; episodes are
  single-threaded). Memory: ~1 GB per worker, so `--mem` ≈ cpus-per-task GB.
- Bumpy-terrain episodes truncate at the patch border, so they need ~340
  episodes/shard to match the bytes of 256 flat episodes.
- One shard per array task. `--array=0-(N-1)%K` runs at most K tasks at once;
  concurrent CPU footprint = K × cpus-per-task; walltime applies **per task**.
  Keep walltime ≥ 2× expected shard time (default 02:00:00 fits 32-CPU
  shards; 16-CPU shards take ~1 h, still fine).
- Seed bases already used (new datasets MUST pick a fresh one or episodes
  duplicate existing data): 2026061100 = flat tire 10G (`t10`),
  2026061200 = bumpy 10G (`b10`), 2026061300 = flat tire 300G (`t300`).
  Per-shard seed = base + 17 × shard_index; scenario prefix
  `<tag>_s{shard:03d}_{family}`.

## Script skeleton

Copy an existing `scripts/cluster/collect_*.sh`; the load-bearing parts:

```bash
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --cpus-per-task=32
#SBATCH --mem=30G
#SBATCH --partition=research      # sbel only for small jobs
#SBATCH --time=02:00:00

set -euo pipefail
# ... conda bootstrap (see above), cd /srv/home/hzhang699/NeDM ...
JOBS="${SLURM_CPUS_PER_TASK:-16}"
python scripts/prepare_..._generation.py --chrono-data-root "$CONDA_PREFIX/share/chrono/data"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then shards=("$SLURM_ARRAY_TASK_ID");
else shards=($(seq 0 $((NUM_SHARDS - 1)))); fi

for shard in "${shards[@]}"; do
  # read output_subdir out of the shard config json
  [[ -f "$output_dir/dataset_index.json" ]] && { echo "shard $shard done; skipping"; continue; }
  python scripts/collect_hmmwv_dataset.py --config "$config" --jobs "$JOBS"
  python scripts/validate_hmmwv_tire_dataset.py --dataset-dir "$output_dir"
done
```

The `dataset_index.json` skip makes resubmission after timeouts/preemption
safe: rerun the identical sbatch line and only missing shards re-collect.

## Checklist for a new dataset script

1. Write the prepare script (fresh seed base + scenario prefix; mirror an
   existing one). If terrain needs assets, commit them under `assets/` so
   `git pull` delivers them; have the script fail fast if they're missing.
2. Smoke-test locally first (`scripts/smoke_test_*.sh`, 12 episodes) —
   the user wants local verification before anything touches the cluster.
3. Copy the cluster script template, set job-name/PLAN_DIR/NUM_SHARDS/prepare
   call. Big runs: estimate total CPU-hours and check the partition with
   `sinfo -o "%C"` before recommending a `%K` throttle (~30–60% of idle CPUs).
4. Commit + push, then hand the user the exact `git pull` + `sbatch` lines,
   plus progress checks: `squeue -u hzhang699` and
   `ls artifacts/datasets/<shards>/*/dataset_index.json | wc -l`.
