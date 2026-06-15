# NeDM Project Progress

A living log of the overall project state, so both of us can see at a glance what is done, what the headline numbers are, and what is next. Update this file whenever a milestone lands or a headline metric changes.

Last updated: 2026-06-15 (15-D tire-normal-force/omega dynamics model, RL policy, NN + Chrono eval)

## Status At A Glance

| # | Milestone | Status | Headline result |
|---|---|---|---|
| 1 | Rigid flat-terrain HMMWV dataset | Done | ~310 GB across 4 dataset generations, 100 Hz episode CSVs |
| 2 | NN dynamics model for HMMWV | Done | Upgraded from 7-D state to 15-D tire-normal-force/omega state; current RL backbone is `hmmwv_transformer_v07_tire_normal_force_omega_300g` |
| 3 | RL tracking on NN dynamics + Chrono eval | Done (first pass) | 15-D policy `hmmwv_rl_15d_5090_2048env_tmux/model_300.pt`: NN and Chrono eval both run on train refs and filtered rest-start validation refs |

## Milestone 1: Rigid Flat-Terrain HMMWV Dataset

Fixed simulation regime across all datasets: `HMMWV_Full`, flat rigid terrain, friction `mu = 0.9`, `TMEASY` tires, `SMC` contact, 2 ms simulation step, 100 Hz recording, per-scenario warmup discard, episode-level train/val splits. Pipeline documented in [data_collection_pipeline.md](data_collection_pipeline.md); collector in `src/nedm/hmmwv_data.py`.

Datasets generated (local-only, gitignored):

| Dataset | Episodes | Rows | Size | Generated | Notes |
|---|---:|---:|---:|---|---|
| `hmmwv_overfit_v1` | 6 | — | small | 2026-03 | Pilot run, one episode per maneuver |
| `hmmwv_overfit_6k` | 6,000 | 5.7 M | 4.3 GB | 2026-03-09 | Base excitation set: launch/brake, step steer, sine steer, chirp |
| `hmmwv_aggressive_steer_2k` | 2,000 | 1.9 M | 1.5 GB | 2026-05-24 | Stronger turning maneuvers to fix turn-response gaps |
| `hmmwv_turn_300g` | 82,000 (82 shards × 1,000) | — | 300 GiB | 2026-05-24 | Turning-focused; low/medium/fast speed bands; families: multi_steer, sustained_turn, sine, chirp, doublet, steer_brake |

Processed sequence caches (in `artifacts/training_datasets/`):

- `hmmwv_overfit_6k_seq_v1` — 4.6 M train / 1.1 M val transitions
- `hmmwv_overfit_6k_plus_aggressive_steer_2k_seq_v1` — 6.0 M train / 1.5 M val
- `hmmwv_turn_300g_plus_base_seq_v1` — 329 M train / 81 M val (all 82 shards + base sets)

## Milestone 2: NN Dynamics Model

GPT-style causal transformer over continuous tokens at 100 Hz. The original HMMWV dynamics stack used 10-d state+action tokens (7 state fields plus 3 controls) and predicted the 7-d next-step state delta. The current RL backbone is the upgraded 15-state tire-normal-force/omega model described below. In both versions, position and yaw are reconstructed by integration during rollout. Pipeline documented in [hmmwv_training_pipeline.md](hmmwv_training_pipeline.md); checkpoints in Git LFS per [model_checkpoints.md](model_checkpoints.md).

Training history:

- **v1 / v2_block64** (2026-04) — first models on `hmmwv_overfit_6k_seq_v1`, established the pipeline and rollout-RMSE validation protocol.
- **v04–v18 architecture sweep** (completed 2026-05-26) — 12 recipes, 80 epochs each, on the full 329 M-transition `hmmwv_turn_300g_plus_base_seq_v1` cache (≈300 GB raw pool). Ranked by median XY RMSE over a fixed set of 20 full validation rollouts:
  - **v07 `context128_b64`** won on median XY RMSE (5.96 m) — the legacy 7-state RL dynamics backbone before the 15-D upgrade.
  - **v04 `long_baseline_b32`** had the best mean/max robustness (mean 15.1 m) — the short-context fallback.
  - Lowest one-step val loss (v18, v12) did **not** give the best rollouts — long-horizon rollout error is the metric that matters.
- **v3_turn_300g** (2026-05-25) — v3 architecture on the 329 M-transition turn cache, ~20 epochs. Best val loss 0.0477; open-loop rollout XY RMSE 0.002 m @ 1 s, 0.014 m @ 2 s, 0.346 m @ 5 s.
- **v19–v30 focused sweep** — started 2026-05-26, crashed on the first model (training subprocess died with signal 6); never re-run. Open item.

### 15-D Tire-Normal-Force/Omega Upgrade (2026-06-15)

The current RL backbone has been upgraded from the earlier 7-state dynamics model to a 15-state model that includes tire vertical normal forces and wheel spindle angular velocities:

```text
artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth
```

State fields are:

```text
vel_body_x_mps, vel_body_y_mps, roll_rad, pitch_rad,
roll_rate_radps, ang_vel_body_y_radps, yaw_rate_radps,
tire_fl_force_wheel_fz_n, tire_fr_force_wheel_fz_n,
tire_rl_force_wheel_fz_n, tire_rr_force_wheel_fz_n,
tire_fl_spindle_omega_radps, tire_fr_spindle_omega_radps,
tire_rl_spindle_omega_radps, tire_rr_spindle_omega_radps
```

Actions remain the 3 driver channels: steering, throttle, braking. The model uses a 128-step context at 100 Hz (`dt_s = 0.01`). The matching RL reference sets are the `hmmwv_tire_normal_force_omega_*` compact `.npz` files under `artifacts/rl_reference_sets/`.

## Milestone 3: RL Tracking (NN Dynamics Training, NN + Chrono Eval)

PPO trajectory-tracking policy trained entirely inside the frozen NN dynamics model, then evaluated both in the NN env and against real Chrono. Documented in [rl_tracking.md](rl_tracking.md); code in `src/nedm/rl/`.

Setup of the current best run (`hmmwv_rl_tracking_v07_8192env_16steps_term1m_20260608`):

- Vectorized NN env: 8,192 parallel envs on GPU, frozen **v07** dynamics checkpoint, `next_state = state + predicted_delta`, pose integrated from body velocity and yaw rate.
- Policy: rsl-rl PPO, actor/critic MLP 512-256-128 (ELU), empirical obs normalization, 2,000 iterations, 16 steps/env per iteration.
- Control: one policy action (steering/throttle/brake) held for 5 NN steps → 20 Hz control over 100 Hz dynamics; 180 policy steps per episode (~9 s).
- Observations: 10-step state/action history + 10-step reference preview. Reward: Gaussian position/yaw/state tracking terms minus action-rate and throttle-brake penalties. Termination at 1 m position error during training.
- References: 20 fixed 1,100-transition training-set segments spanning all maneuver families (`hmmwv_train_refs_20_1100_rest_start.npz`, rest-start so Chrono can warm-start from zero speed).

Evaluation of `model_1999` over the 20 references (eval termination relaxed to 20 m):

| Eval backend | Median XY RMSE | Mean XY RMSE | Diverged |
|---|---:|---:|---:|
| NN dynamics env | 0.170 m | 0.238 m | 0 / 20 |
| Chrono (sim-to-sim) | 0.245 m | 0.929 m | 1 / 20 |

Chrono transfer detail: 16 of 20 references track under 0.5 m RMSE. The failures concentrate in braking-heavy maneuvers — `steer_brake` 6.8 m (terminated early), `launch_brake` 4.98 m, `aggressive_step_steer` 1.5 m. Turning families (sustained_turn, sine, chirp, doublet, multi_steer) transfer well.

Supporting work that landed with this milestone:

- `create_hmmwv` now honors `yaw_rad` and `fwd_vel_mps` init so Chrono eval can warm-start at the reference pose/speed.
- Chrono eval gotchas were worked through and recorded: references must start from rest, the reference line must attach to the existing terrain body (a new `ChBody` perturbs the solver), and full-loop multi-reference eval must run one reference per process due to a native stack-smash on repeated sim re-creation.
- **pychrono 10 verification (2026-06-09)**: new `nedm` conda env with pychrono 10.0.0 from the official `projectchrono` channel (replacing the 9.0.1 `bochengzou` build in `tutorial`). One API rename fixed (`veh.SetDataPath` → `SetVehicleDataPath`, compat shim in `hmmwv_data.py`). Re-ran the full 20-ref Chrono eval (`chrono_eval_model1999_reststart_pychrono10`): median 0.280 m vs 0.245 m under 9.0.1; 15/20 references match within ~0.05 m, marginal references flip both ways (launch_brake improved 4.98→0.45 m; two sine-steer refs diverged). Native fragility persists: eval processes can crash during plotting after the rollout npz is saved.
- **Steering rate-limit filter (2026-06-09)**: rendered rollout analysis showed the Chrono-10 divergences are abrupt steering reversals shoving the tires into combined-slip saturation (full throttle, vehicle decelerates to a stop). Added a `steering_rate_limit` option to the Chrono eval env (clamp steering to ±threshold of the previous policy step). At 0.3 it eliminates **all** model_1999 divergences with no cost elsewhere: mean 1.360 → 0.255 m, median 0.280 → 0.217 m, diverged 3 → 0; even `steer_brake_s010` (diverged under both pychrono versions) drops to 0.68 m. Training-side hard termination on steering jumps is the follow-up (see `.claude/lessons_learned.md`).
- **5 GB dynamics/RL scaling-law signal (2026-06-11)**: evaluated `hmmwv_rl_tracking_d005_v07_20260610_2048env_unbuf/model_1300.pt`, whose policy was trained against the `hmmwv_transformer_d005_v07_005g` NN dynamics checkpoint instead of the 300 GB backbone. On the same rest-start 20-reference set (`hmmwv_train_refs_20_1100_rest_start.npz`), NN-env tracking stayed strong: mean 0.186 m, median 0.148 m, 0/20 diverged. Raw Chrono transfer without steering clamp exposed the same steering-jump failure mode as the larger run: mean 1.802 m, median 0.442 m, 3/20 diverged. With the existing `steering_rate_limit=0.3` clamp, all 20 Chrono rollouts completed: mean 0.274 m, median 0.211 m, 0/20 diverged; worst case was `steer_brake/s010_steer_brake_00066` at 0.766 m RMSE. This is important evidence that a scaling law exists for the NN dynamics model: even the 5 GB data-scale model produces a policy whose clamped Chrono transfer is in the same regime as the 300 GB/v07 policy, while the remaining gap shows up as the same controllable action-smoothness pathology rather than broad tracking failure.

### 15-D NN-Dynamics RL Policy (2026-06-15)

The 15-D tire-normal-force/omega dynamics checkpoint now has a trained PPO tracking policy:

```text
artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/model_300.pt
```

Run setup recovered from `env_cfg.json`:

- 2,048 vectorized NN envs
- frozen dynamics checkpoint: `hmmwv_transformer_v07_tire_normal_force_omega_300g/checkpoints/best_val.pth`
- training references: `hmmwv_tire_normal_force_omega_train_refs_20_1100_seed_20260607.npz`
- 20 Hz policy control (`action_repeat = 5` over 100 Hz NN dynamics)
- 180 policy steps per episode
- no steering-rate limiter in this run (`steering_rate_limit = None`)

The NN rollout code was optimized to use the model's last-token `predict_next_delta` path; a direct check showed it is numerically identical to the old full-window `predict_delta(... )[:, -1, :]` path (`max_abs_diff = 0.0`). The NN env and eval script now use `torch.no_grad()` rather than wrapping mutable env buffers in outer `torch.inference_mode()`, which avoids PyTorch inference-tensor reset issues under the `nedm` environment.

Evaluation works in both backends:

| Eval set / backend | Metric note | Mean XY RMSE | Median XY RMSE | Mean XY mean error |
|---|---|---:|---:|---:|
| Training refs, NN env | closest comparison to training TensorBoard | 0.213 m | 0.169 m | 0.190 m |
| Filtered val rest-start refs, NN env | held-out validation set, zero/rest handoff | 0.631 m | 0.445 m | 0.462 m |
| Filtered val rest-start refs, Chrono env | `nedm` env, CPU, no steering clamp | 0.393 m | 0.287 m | 0.279 m |

The training TensorBoard scalar `/episode/mean_pos_error_m` at iteration 301 was `0.173 m`; this is closest to eval `xy_mean_m`, not eval `xy_rmse_m`. Rechecking `model_300.pt` on the original training references gives average `xy_mean_m = 0.190 m`, which is consistent with the training log. The harder held-out rest-start validation set has several outliers, so its aggregate is substantially higher.

Eval artifacts:

- NN train-ref recheck: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/eval_tracking_model_300_train_refs_recheck/`
- NN held-out rest-start eval: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/eval_tracking_model_300_val_rest_start/`
- Chrono held-out rest-start eval: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux/chrono_eval_tracking_model_300_val_rest_start/`
- Reference construction/eval workflow skill: `.agents/hmmwv-nn-eval/`
- Chrono eval workflow skill: `.agents/hmmwv-chrono-eval/`

## Bumpy-Terrain Transfer (2026-06-11)

First out-of-regime test: take the **flat-terrain-trained** `model_1999` policy and evaluate it in Chrono on **bumpy rigid-heightmap terrain** (the same `bumpy_field_*.bmp` library the 10 GB bumpy dataset was collected on, 500×500 m patches, height ±0.6 m). The Chrono env now reproduces the exact per-episode terrain: each reference's heightmap is recovered deterministically from its `episode_id` via `assign_height_map_index` (verified to match every stored `height_map_index`), and `HMMWVChronoTrackingEnv._create_sim` passes it to `create_rigid_terrain`. Setup: bumpy reference set `hmmwv_bumpy_refs_20_1100_rest_start.npz` (rest-start; 6 families — bumpy data has no launch_brake/step_steer/aggressive_*), eval config `configs/hmmwv_bumpy_eval.json`, `steering_rate_limit=0.3`, 20 m bound. See the `run-bumpy-terrain-eval` skill for the full recipe.

| Eval backend / terrain | Median XY RMSE | Mean XY RMSE | Diverged |
|---|---:|---:|---:|
| Chrono, flat terrain (smooth refs) | 0.217 m | 0.255 m | 0 / 20 |
| **Chrono, bumpy terrain (bumpy refs)** | **0.345 m** | **1.46 m** | **4 / 20** |

The flat-trained policy **does not transfer well to bumpy terrain**: mean RMSE jumps 0.26 → 1.46 m and 4/20 references diverge to the 20 m bound (refs 1 sine, 3 multi, 14 doublet, 16 chirp — all high-speed, high-travel steering maneuvers where the bumps perturb the tires most). Slow/braking refs (sustained_turn, steer_brake) still track within ~0.2–0.5 m. This is expected: both the frozen v07 NN dynamics model and the PPO policy only ever saw flat-terrain tire dynamics, so bump-induced load transfer and tire-force variation are out of distribution. **Closing this gap requires finetuning both the NN dynamics model and the policy on the bumpy dataset** — the eval harness for measuring that is now in place.

## Open Items / Next Steps

- **Braking transfer gap**: the policy tracks turning references in Chrono but diverges on braking-heavy ones — likely a dynamics-model gap (brake response) rather than a policy gap; worth checking v07 open-loop rollout error on launch_brake/steer_brake segments specifically.
- **v19–v30 sweep** crashed at the first model and was never completed.
- **RL on alternate dynamics backbones**: the current active policy uses the 15-D tire-normal-force/omega v07-style model; the older `v3_turn_300g` backbone remains untested as an RL backbone.
- **15-D held-out validation outliers**: held-out/rest-start eval now exists for the 15-D policy. The NN env and Chrono env both run, but the validation set has several NN outliers; compare `xy_mean_m` to training logs and inspect per-reference behavior before drawing conclusions from aggregate RMSE alone.
- **Bumpy-terrain finetune (next major step)**: the flat-trained policy regresses badly on bumpy terrain (mean 0.26 → 1.46 m, 4/20 diverge — see Bumpy-Terrain Transfer above). Plan: (1) finetune the v07 NN dynamics model on the `hmmwv_bumpy_10g_seq_v1` processed cache so it captures bump-induced load/tire dynamics, then (2) finetune/retrain the PPO policy against that bumpy dynamics model, and (3) re-run the bumpy Chrono eval (`run-bumpy-terrain-eval` skill) to measure recovery. The eval harness, reference set, and per-episode terrain reproduction are all done.
- **Beyond the fixed regime**: bumpy-terrain *evaluation* now exists (heightmap terrain reproduced per episode); friction variation, observation noise, and tire-channel supervision are still out of scope.
