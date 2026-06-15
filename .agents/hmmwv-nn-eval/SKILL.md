---
name: hmmwv-nn-eval
description: Build and evaluate HMMWV neural-dynamics RL tracking reference sets for the NeDM repo. Use when Codex needs to construct filtered rest-start HMMWV NN tracking references, avoid training-reference leakage, remove too-short/out-of-distribution eval trajectories, run scripts/eval_hmmwv_rl_tracking.py against an scp-copied RL run, or document/compare HMMWV RL eval artifacts under artifacts/rl_runs.
---

# HMMWV NN Eval

## Scope

Work from `/home/harry/NeDM`. This skill is for the 15-D tire-normal-force/omega HMMWV RL tracking setup that uses:

- default dynamics checkpoint: `artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth`
- default processed cache: `artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1`
- training reference set to avoid reusing: `artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz`
- copied RL run example: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux`

Use the repo's existing eval script for policy rollout. Do not edit `env_cfg.json` in a copied training run just to evaluate against a different reference set; create a temporary run config under `/tmp` with only `reference_path` changed.

## Reference Construction

Use `scripts/build_rest_start_refs.py` from this skill to create a compact validation reference set with:

- source split `val`, not `train`
- no random segment start (`local_start = 0`)
- poses expressed in the first pose's local frame
- a padded rest context so the unmodified NN env handoff index (`context_steps - 1`) corresponds to source trajectory index 0
- initial `x/y/yaw`, velocity/rates, spindle omegas, and driver action set to zero at handoff
- same-family replacements for any evaluated reference path shorter than the minimum threshold

Default command:

```bash
PYTHONPATH=src /home/harry/anaconda3/envs/tutorial/bin/python \
  .agents/hmmwv-nn-eval/scripts/build_rest_start_refs.py
```

The default output is:

```text
artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz
```

The script records selected episode IDs, family counts, rest-start metadata, path filtering, and replacements in `metadata_json`.

## Eval Workflow

Create a temporary eval config from the copied run:

```bash
/home/harry/anaconda3/envs/tutorial/bin/python - <<'PY'
import json, shutil
from pathlib import Path

src = Path("artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux")
dst = Path("/tmp/hmmwv_rl_15d_5090_val_rest_start_eval_cfg")
dst.mkdir(parents=True, exist_ok=True)
env = json.loads((src / "env_cfg.json").read_text())
env["reference_path"] = "artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz"
env["num_envs"] = 20
(dst / "env_cfg.json").write_text(json.dumps(env, indent=2))
shutil.copyfile(src / "train_cfg.json", dst / "train_cfg.json")
PY
```

Run eval:

```bash
/home/harry/anaconda3/envs/tutorial/bin/python scripts/eval_hmmwv_rl_tracking.py \
  --run-dir /tmp/hmmwv_rl_15d_5090_val_rest_start_eval_cfg \
  --policy-checkpoint artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/model_100.pt \
  --device cpu \
  --output-dir artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/eval_tracking_model_100_val_rest_start
```

Use `--device cuda` only after confirming CUDA is visible in the active env. On this host, CPU eval was used when `torch.cuda.is_available()` was false and `nvidia-smi` could not talk to the driver.

## Validation

After building references, scan the evaluated reference path lengths. The script prints min/median/max path lengths; reject sets with short near-static references if the task is intended to test meaningful tracking. In the June 15, 2026 eval, the final accepted threshold was `--min-eval-path-m 20`, which produced:

- 20 validation trajectories
- no overlap with the original training reference set
- family counts: sustained_turn 4, sine_steer 4, doublet_steer 3, multi_steer 3, chirp_steer 3, steer_brake 3
- minimum evaluated reference path about 30.54 m
- median evaluated reference path about 39.98 m

Always verify overlap against the training reference set by comparing `metadata_json["episode_ids"]`, not by assuming split names are enough.

For the detailed reconstruction notes and final replacement list from the original run, read `references/workflow.md`.
