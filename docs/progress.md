# NeDM Project Progress

A living log of the overall project state, so both of us can see at a glance what is done, what the headline numbers are, and what is next. Update this file whenever a milestone lands or a headline metric changes.

Last updated: 2026-06-09 (pychrono 10 verification added)

## Status At A Glance

| # | Milestone | Status | Headline result |
|---|---|---|---|
| 1 | Rigid flat-terrain HMMWV dataset | Done | ~310 GB across 4 dataset generations, 100 Hz episode CSVs |
| 2 | NN dynamics model for HMMWV | Done | v07 transformer: best sweep model; v3_turn_300g: 0.35 m XY RMSE on 5 s open-loop rollouts |
| 3 | RL tracking on NN dynamics + Chrono eval | Done (first pass) | NN eval median 0.17 m XY RMSE; Chrono sim-to-sim median 0.25 m over 20 references |

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

GPT-style causal transformer over continuous tokens at 100 Hz: input is a context window of 10-d state+action tokens (7 state fields: body velocities, roll, pitch, roll rate, pitch rate, yaw rate; 3 controls: steering, throttle, braking), output is the 7-d next-step state delta. Position and yaw are reconstructed by integration during rollout. Pipeline documented in [hmmwv_training_pipeline.md](hmmwv_training_pipeline.md); checkpoints in Git LFS per [model_checkpoints.md](model_checkpoints.md).

Training history:

- **v1 / v2_block64** (2026-04) — first models on `hmmwv_overfit_6k_seq_v1`, established the pipeline and rollout-RMSE validation protocol.
- **v04–v18 architecture sweep** (completed 2026-05-26) — 12 recipes, 80 epochs each, on the full 329 M-transition `hmmwv_turn_300g_plus_base_seq_v1` cache (≈300 GB raw pool). Ranked by median XY RMSE over a fixed set of 20 full validation rollouts:
  - **v07 `context128_b64`** won on median XY RMSE (5.96 m) — the default RL dynamics backbone.
  - **v04 `long_baseline_b32`** had the best mean/max robustness (mean 15.1 m) — the short-context fallback.
  - Lowest one-step val loss (v18, v12) did **not** give the best rollouts — long-horizon rollout error is the metric that matters.
- **v3_turn_300g** (2026-05-25) — v3 architecture on the 329 M-transition turn cache, ~20 epochs. Best val loss 0.0477; open-loop rollout XY RMSE 0.002 m @ 1 s, 0.014 m @ 2 s, 0.346 m @ 5 s.
- **v19–v30 focused sweep** — started 2026-05-26, crashed on the first model (training subprocess died with signal 6); never re-run. Open item.

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

## Open Items / Next Steps

- **Braking transfer gap**: the policy tracks turning references in Chrono but diverges on braking-heavy ones — likely a dynamics-model gap (brake response) rather than a policy gap; worth checking v07 open-loop rollout error on launch_brake/steer_brake segments specifically.
- **v19–v30 sweep** crashed at the first model and was never completed.
- **RL on the v3_turn_300g model**: the RL backbone is still v07 (trained on 6k+2k); the 300g-trained model has much better open-loop rollouts and is untested as an RL backbone.
- **Held-out references**: RL eval currently uses training-set segments; no held-out-trajectory evaluation yet.
- **Beyond the fixed regime**: terrain/friction variation, observation noise, and tire-channel supervision are all still deliberately out of scope.
