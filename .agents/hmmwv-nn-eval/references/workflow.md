# HMMWV NN Rest-Start Eval Workflow

## Artifact Intent

The reference set should evaluate a trained HMMWV NN tracking policy outside the original training reference trajectories, while still starting from rest at the origin. The unmodified `HMMWVNeuralTrackingEnv` resets at `context_steps - 1`, not source index 0. For the 15-D normal-force/omega checkpoint, `context_steps = 128`, so the constructed reference set pads the first 127 samples with a rest state/action/pose. The env handoff index 127 then corresponds to the source episode's zero index.

## Current Accepted Artifact

Path:

```text
artifacts/rl_reference_sets/hmmwv_tire_normal_force_omega_val_refs_20_1100_rest_start.npz
```

Properties:

- `source_split`: `val`
- `random_segment_start`: `false`
- `rest_start`: `true`
- `rest_context_pad_steps`: `127`
- `rest_handoff_reference_index`: `127`
- `pose_frame`: `first_pose_relative`
- `min_eval_path_filter_m`: `20`

The accepted set uses 20 references with family counts:

```text
sustained_turn: 4
sine_steer:     4
doublet_steer:  3
multi_steer:    3
chirp_steer:    3
steer_brake:    3
```

Final evaluated path scan:

```text
min:    30.540 m
median: 39.984 m
max:    66.337 m
```

## Replacements Made

The first rest-start validation set included near-static references. They were replaced by same-family validation episodes whose evaluated path length was close to 40 m.

```text
position 01 sine_steer:    t300_s092_sine_steer_00030    12.577 m -> t300_s109_sine_steer_00002    39.960 m
position 03 multi_steer:   t300_s044_multi_steer_00059    19.491 m -> t300_s043_multi_steer_00021   39.998 m
position 05 steer_brake:   t300_s084_steer_brake_00015     2.571 m -> t300_s113_steer_brake_00012   39.971 m
position 08 doublet_steer: t300_s031_doublet_steer_00003   3.596 m -> t300_s053_doublet_steer_00013 39.924 m
position 11 steer_brake:   t300_s056_steer_brake_00006     2.468 m -> t300_s017_steer_brake_00000   39.965 m
position 15 multi_steer:   t300_s112_multi_steer_00023     6.561 m -> t300_s119_multi_steer_00053   40.005 m
```

Earlier, `tracking_04` was also replaced because it was almost stationary:

```text
position 04 chirp_steer: t300_s075_chirp_steer_00008 0.182 m -> t300_s052_chirp_steer_00025 35.006 m
```

## Final Eval Output

Output directory:

```text
artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/eval_tracking_model_100_val_rest_start
```

Final summary after filtering short references:

```text
mean_xy_rmse_m:   0.6283993072807789
median_xy_rmse_m: 0.4143199473619461
num_rollouts:     20
```

The eval was run with a temporary config at:

```text
/tmp/hmmwv_rl_15d_5090_val_rest_start_eval_cfg
```

Only `reference_path` and `num_envs` were changed in the temporary config. The copied run's `env_cfg.json` was not edited.

## Checks To Repeat

1. Confirm the new `.npz` uses 15-D state fields matching the dynamics checkpoint.
2. Confirm `rest_handoff_reference_index` pose is exactly zero and motion fields are zero.
3. Compare episode IDs against `hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz`; overlap should be zero.
4. Scan evaluated path lengths over the same sample indices used by the eval script: `handoff + action_repeat * arange(1, max_steps + 1)`.
5. Inspect any plotted trajectory that looks short or visually out-of-distribution and rebuild with a stricter `--min-eval-path-m` if needed.
