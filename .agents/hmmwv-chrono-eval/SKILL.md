---
name: hmmwv-chrono-eval
description: Run HMMWV RL tracking policy checkpoints in the Chrono HMMWV environment for the NeDM repo. Use when Codex needs to evaluate a policy with scripts/eval_hmmwv_rl_chrono_tracking.py, compare NN-env vs Chrono-env results, use rest-start reference sets, choose the correct nedm conda environment, handle pychrono runtime setup, or decide whether to apply a steering-rate limit during Chrono eval.
---

# HMMWV Chrono Eval

Work from `/home/harry/NeDM`.

## Environment

Use the `nedm` conda environment, not `tutorial`.

For Chrono eval, activate conda in the shell before running Python so `pychrono` picks up the conda C++ runtime:

```bash
source /home/harry/anaconda3/etc/profile.d/conda.sh
conda activate nedm
```

Avoid `conda run -n nedm` for long evals on this machine; it has previously segfaulted on the NN eval path. Avoid calling `/home/harry/anaconda3/envs/nedm/bin/python` directly for Chrono imports unless the conda runtime libraries are already active; direct `pychrono` imports can fail with a `libstdc++`/`CXXABI` error.

Check CUDA before using `--device cuda`:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

Use `--device cpu` when CUDA is not visible. Chrono stepping is CPU-bound even when policy tensors use CUDA.

## Reference Config

Do not edit a copied run's `env_cfg.json` just to evaluate another reference set. Create a temporary run config under `/tmp` with only the eval-specific values changed.

Example for the 15-D copied run and filtered validation rest-start references:

```bash
python - <<'PY'
import json, shutil
from pathlib import Path

src = Path("artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux")
dst = Path("/tmp/hmmwv_rl_15d_5090_val_rest_start_eval_cfg_model300")
dst.mkdir(parents=True, exist_ok=True)

env = json.loads((src / "env_cfg.json").read_text())
env["reference_path"] = "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz"
env["num_envs"] = 20

(dst / "env_cfg.json").write_text(json.dumps(env, indent=2))
shutil.copyfile(src / "train_cfg.json", dst / "train_cfg.json")
PY
```

For an apples-to-apples training-log comparison, use the run's original training reference set instead. In the copied 15-D run, that is:

```text
artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz
```

Training `/episode/mean_pos_error_m` is closest to eval `xy_mean_m`, not `xy_rmse_m`.

## Chrono Eval Command

Run the Chrono evaluator from an activated `nedm` shell:

```bash
MPLCONFIGDIR=/tmp/nedm_mplconfig python scripts/eval_hmmwv_rl_chrono_tracking.py \
  --run-dir /tmp/hmmwv_rl_15d_5090_val_rest_start_eval_cfg_model300 \
  --policy-checkpoint artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/model_300.pt \
  --device cpu \
  --output-dir artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/chrono_eval_tracking_model_300_val_rest_start
```

The script writes:

- `summary.json`
- `chrono_tracking_XX.png`
- `chrono_tracking_XX.npz`

Use `--num-references N` for a short scan, or `--reference-index I` to debug one trajectory.

## Steering Rate Limit

By default, Chrono eval does not clip adjacent steering commands. The saved `summary.json` records this as:

```json
"steering_rate_limit": null
```

If the user asks for steering-rate clipping or wants to match a limiter experiment, pass:

```bash
--steering-rate-limit 0.3
```

The value is command-space delta per policy step:

```text
steer_t = clamp(steer_t, last_steer - limit, last_steer + limit)
```

State clearly in the final answer whether the eval used a steering-rate limit.

## Validation Checklist

After eval:

1. Read `summary.json` and report `num_rollouts`, mean/median `xy_rmse_m`, and whether `steering_rate_limit` is null.
2. If comparing to training logs, also compute average and median rollout `xy_mean_m`.
3. Verify Chrono and NN summaries use the same reference names before comparing per-reference metrics.
4. Mention the environment used: activated `nedm`, CPU or CUDA.
5. Do not overwrite NN eval artifacts when running Chrono; use an output folder whose name starts with `chrono_eval`.
