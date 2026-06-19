---
name: hmmwv-nn-model-training
description: Launch, resume, monitor, and diagnose HMMWV neural dynamics model training in the NeDM repo. Use when asked to train or check HMMWV transformer dynamics models, v07-style NN dynamics runs, tire-force/omega or tire-normal-force/omega datasets, 300G processed caches, tmux launch scripts, CUDA training status, data-loader bus errors, checkpoints, or open-loop rollout overlays.
---

# HMMWV NN model training

Work from `/home/harry/NeDM`. Prefer the repo's config and launch scripts over ad hoc training commands. Long GPU jobs should run in tmux.

## First checks

Before launching or changing anything:

```bash
git status --short
pgrep -af "train_hmmwv_dynamics|run_hmmwv_v07"
tmux list-sessions
nvidia-smi
```

The sandbox may not see tmux or CUDA. If commands fail with permission/GPU visibility errors, request escalation and rerun the same direct command.

## Active v07 tire experiments

Full tire-force/omega run, 23-D state:

```text
state = 7 base states + 12 wheel-frame tire force channels + 4 spindle omega channels
config = configs/hmmwv_transformer_v07_tire_force_omega_300g.json
cache = artifacts/training_datasets/hmmwv_tire_rigid_300g_force_omega_seq_v1
run = artifacts/training_runs/hmmwv_transformer_v07_tire_force_omega_300g
tmux = hmmwv_v07_tire_force_omega_300g_training
launcher = scripts/launch_hmmwv_v07_tire_force_omega_300g_training.sh
```

Normal-force/omega run, 15-D state:

```text
state = 7 base states + 4 wheel-frame Fz normal-force channels + 4 spindle omega channels
config = configs/hmmwv_transformer_v07_tire_normal_force_omega_300g.json
cache = artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1
run = artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g
tmux = hmmwv_v07_tire_normal_force_omega_300g_training
launcher = scripts/launch_hmmwv_v07_tire_normal_force_omega_300g_training.sh
```

Interpret "normal upward tire force" as the existing wheel-frame `*_force_wheel_fz_n` channels. Do not add slip ratio by default.

## Required training settings

For large v07-style HMMWV runs, keep these config values:

```json
"load_dataset_into_memory": true,
"pin_memory": false
```

This mirrors the previous v07 data-loading fix. Without it, mmap-backed arrays plus CUDA/pinned memory have produced hard failures such as `Bus error (core dumped)`, `exit_code=135`, signal 7, or libc10 trap messages.

## Launch

Use the launcher for the requested experiment:

```bash
bash scripts/launch_hmmwv_v07_tire_force_omega_300g_training.sh
bash scripts/launch_hmmwv_v07_tire_normal_force_omega_300g_training.sh
```

The run scripts handle:

- conda env activation with `CONDA_NO_PLUGINS=true` and `conda activate tutorial`
- processed-cache creation or verification
- state-field verification against `STATE_FIELD_PRESETS`
- resume from `checkpoints/last.pt` when present
- status JSON updates under the run directory

The normal-force/omega cache should usually be built from the existing 23-D force/omega cache using:

```bash
python scripts/build_hmmwv_state_subset_dataset.py \
  --source-dir artifacts/training_datasets/hmmwv_tire_rigid_300g_force_omega_seq_v1 \
  --output-dir artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1 \
  --state-field-preset tire_normal_force_omega
```

This selects state/target columns and symlinks unchanged action, rollout, and split files. Prefer this over reparsing raw CSV shards when the 23-D cache is available.

## Monitor

For a run directory:

```bash
RUN=artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g
cat "$RUN/status.json"
tail -n 5 "$RUN/metrics.jsonl"
tail -n 40 "$RUN/logs/run.log"
pgrep -af "train_hmmwv_dynamics|run_hmmwv_v07"
nvidia-smi
```

Checkpoint summary:

```bash
export CONDA_NO_PLUGINS=true
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial
python - <<'PY'
from pathlib import Path
import torch

p = Path("artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/last.pt")
ck = torch.load(p, map_location="cpu", weights_only=False)
print(ck["epoch"], ck["global_step"], ck.get("metrics", {}).get("val_loss"))
print(ck["config"]["processed_dataset_dir"])
PY
```

A stale `status.json` after reboot does not prove the job is running. Verify tmux, process state, metrics timestamps, and GPU utilization.

## Open-loop overlays

After training finishes, run the same 20 validation overlays used for v07 comparisons:

```bash
export CONDA_NO_PLUGINS=true
export MPLCONFIGDIR=/tmp/nedm_mplconfig
mkdir -p "$MPLCONFIGDIR"
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate tutorial
python scripts/plot_hmmwv_rollout_overlay.py \
  --checkpoint artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pt \
  --split val \
  --num-random 20 \
  --seed 20260525 \
  --device cuda \
  --output-dir artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/plots/random_val_overlays_n20_seed_20260525
```

If CUDA is hidden in the sandbox, rerun outside the sandbox or fall back to `--device cpu`.

## Failure handling

For `Bus error`, signal 7, libc10 trap messages, or `exit_code=135`:

1. Inspect `logs/run.log`, `metrics.jsonl`, and whether `checkpoints/last.pt` exists.
2. Confirm the config still has `load_dataset_into_memory: true` and `pin_memory: false`.
3. Confirm processed arrays load with `np.load(..., mmap_mode="r")`.
4. Resume from `last.pt`; do not delete a partial run unless the checkpoint/config is incompatible.

For missing or incomplete processed cache:

1. Verify `metadata.json` and state fields.
2. For the 15-D normal-force run, rebuild from the 23-D cache with `scripts/build_hmmwv_state_subset_dataset.py` if possible.
3. For the 23-D force/omega run, use the raw-shard preprocessing path in `scripts/run_hmmwv_v07_tire_force_omega_300g_training.sh`.

After changing dynamics state dimension, old RL reference files may be incompatible. Rebuild RL references from the matching processed cache before RL policy training/eval.
