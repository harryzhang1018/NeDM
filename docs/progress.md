# NeDM Project Progress

A living log of the overall project state, so both of us can see at a glance what is done, what the headline numbers are, and what is next. Update this file whenever a milestone lands or a headline metric changes.

Last updated: 2026-06-30 (started the **Arm Mobile-Manipulator Study (case 2)**: trained the reach-mode arm dynamics NN `f_arm` — `arm_transformer_v1`, 15-D state, context 16, `val_loss` 0.00181, open-loop EE drift ~2% `errdist` @0.5 s — and laid out the reaching-RL (Phase 4) plan; see the new "Arm Mobile-Manipulator Study" section. Prior 2026-06-29 HMMWV steering-rate-limit entry unchanged)

## Status At A Glance

| # | Milestone | Status | Headline result |
|---|---|---|---|
| 1 | Rigid flat-terrain HMMWV dataset | Done | ~310 GB across 4 dataset generations, 100 Hz episode CSVs |
| 2 | NN dynamics model for HMMWV | Done | Upgraded from 7-D state to 15-D tire-normal-force/omega state; current RL backbone is `hmmwv_transformer_v07_tire_normal_force_omega_300g` |
| 3 | RL tracking on NN dynamics + Chrono eval | Done (first pass) | 15-D policy eval now covers flat and bumpy rest-start refs; bumpy terrain degrades Chrono transfer substantially |
| 4 | CRM (deformable soil) generalist dynamics NN | Generalist + ablations trained on 20× CRM (`crm_2000`) with one-hot terrain conditioning | One-hot 75/25 generalist hits **flat 9.1% / CRM 5.8%** open-loop 10s err/dist — improving the `crm_100` incumbent on **both** (flat 15.4%→9.1%, CRM 9.4%→5.8%). Single-domain ablations reach **flat 5.0% / CRM 3.7%** in-domain but collapse off-domain (flat-only→CRM 69%, CRM-only→flat 37%): co-training trades ~3–4 pt peak accuracy for cross-domain robustness. All three on `main` (LFS); see 2026-06-24 subsection |
| 5 | Arm mobile-manipulator (case 2): dynamics NN + reaching RL | Dynamics model **done**; reaching RL planned | `f_arm` (`arm_transformer_v1`, 15-D state `[q,qd,qcmd,ee_base]`, context 16) reuses the HMMWV stack: `val_loss` **0.00181**, open-loop EE drift **~1.9% `errdist` @0.5 s**, 2.7% @1 s. Reaching policy `π_reach` on the frozen model is the next step; see the "Arm Mobile-Manipulator Study" section |

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

### 15-D Bumpy-Terrain Data and Transfer Check (2026-06-16)

The existing bumpy raw shards did include tire channels, but the old processed bumpy cache and compact references were 7-D. A new 15-D cache was built from the same raw bumpy heightmap data using the tire-normal-force/omega state preset:

```text
artifacts/training_datasets/hmmwv_bumpy_10g_normal_force_omega_seq_v1
```

The cache has 1,104 train episodes / 3.67 M train transitions and 256 val episodes / 0.84 M val transitions. State arrays are 15-D and match the current `hmmwv_transformer_v07_tire_normal_force_omega_300g` checkpoint exactly. New compact 20-reference sets were also built:

- `artifacts/rl_reference_sets/hmmwv_bumpy_10g_normal_force_omega_train_refs_20_1100_seed_20260607.npz`
- `artifacts/rl_reference_sets/hmmwv_bumpy_10g_normal_force_omega_train_refs_20_1100_rest_start.npz`
- `artifacts/rl_reference_sets/hmmwv_bumpy_10g_normal_force_omega_val_refs_20_1100_rest_start.npz`

Chrono bumpy evaluation reproduces the terrain per trajectory, not just per rollout index: `HMMWVChronoTrackingEnv._create_sim` resolves the bumpy heightmap from each reference's `episode_id`, and the selected 20 bumpy validation references were verified against the raw episode JSON `height_map_index` values with 0/20 mismatches.

Flat-vs-bumpy comparison for the newer 15-D run used:

```text
artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux_v2/model_500.pt
```

The flat references are the existing rigid-terrain validation rest-start set (`t300_*`), while the bumpy references are held-out 15-D bumpy validation rest-start refs (`b10_*`). They are not trajectory-paired, but they use the same policy and the same 20-rollout family mix.

| Eval backend / terrain | Mean XY RMSE | Median XY RMSE | Mean reward | Notes |
|---|---:|---:|---:|---|
| NN env, flat refs | 0.429 m | 0.342 m | 162.67 | `eval_tracking_model_500_val_rest_start` |
| NN env, bumpy refs | 0.458 m | 0.401 m | 146.84 | mild degradation: +7% mean RMSE, +17% median RMSE |
| Chrono, flat refs | 0.246 m | 0.217 m | 161.44 | `chrono_eval_tracking_model_500_val_rest_start` |
| Chrono, bumpy refs | 0.615 m | 0.523 m | 135.41 | large degradation: +150% mean RMSE, +142% median RMSE |

Finding: the 15-D NN env shows only mild degradation on bumpy references, but real Chrono bumpy transfer degrades strongly. The worst Chrono bumpy cases were `doublet_steer/b10_s002_doublet_steer_00023` at 1.95 m RMSE and `steer_brake/b10_s002_steer_brake_00021` at 1.74 m RMSE. This indicates that adding tire normal force and spindle omega to the flat-terrain dynamics state helps the policy interface, but it does not by itself close the terrain-domain gap. The dynamics model and policy still need bumpy-terrain adaptation.

Artifacts:

- Flat NN summary: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux_v2/eval_tracking_model_500_val_rest_start/summary.json`
- Flat Chrono summary: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux_v2/chrono_eval_tracking_model_500_val_rest_start/summary.json`
- Bumpy NN summary: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux_v2/eval_bumpy15d_model500_val_rest_start/summary.json`
- Bumpy Chrono summary: `artifacts/rl_runs/hmmwv_rl_15d_5090_2048env_tmux_v2/chrono_eval_bumpy15d_model500_val_rest_start/summary.json`

### Dynamics-Context Speedup for RL Training (2026-06-19)

RL training throughput on the 15-D run was stuck at ~5K steps/s (4090) / ~6.6K (5090) despite 2,048 envs. Root cause, isolated from the RL loop with two new benchmark scripts (`scripts/bench_dynamics_inference.py`, `scripts/bench_context_accuracy.py`): every `_nn_substep` ran the dynamics transformer over the **full `block_size = 128` history** (`state_hist` is allocated at `context_steps`) but only consumes the last token — O(seq²) attention on ~128× more tokens than needed, ×`action_repeat = 5` substeps per policy step. Two findings:

- **Batching saturates around batch = 64**: at seq=128 the GPU is already compute-bound, so 64 → 2,048 envs gives the *same* steps/s. Large batch buys decorrelated experience, not throughput. The "(15,1)→(15,n) is ~free" intuition only holds up to GPU saturation.
- **The dynamics is near-Markovian**: pose RMSE over 300-step open-loop rollouts is flat (0.09–0.23 m) for context K from 128 down to 1; K=16 is as good as / better than the full 128.

Fix: new `dynamics_context_steps` env config knob (+ `--dynamics-context-steps` CLI flag in `scripts/train_hmmwv_rl_tracking.py`) feeds only the last K tokens to the model. Buffers stay full-size and reset still warm-starts from 128 reference steps — only the model input is sliced (`None` = full context, backward-compatible). Measured on the same 4090, identical config otherwise:

| Dynamics context | steps/s | collection / iter | 2,000-iter ETA |
|---|---:|---:|---:|
| Full 128 (baseline) | ~5,080 | 51.4 s | ~29 h |
| **K = 16** | **~34,000** | **7.6 s** | **~4.4 h** |

≈**6.8× end-to-end speedup** (PPO learning is ~0.2 s/iter, negligible); a bit under the 9× pure-inference gain because the full-size buffer roll + obs assembly are now a larger share of the cost.

Relaunched the `hmmwv_rl_15d_5090_2048env_tmux_v2` config verbatim plus `dynamics_context_steps = 16` as `artifacts/rl_runs/hmmwv_rl_15d_4090_2048env_K16` (on a 4090; the original "5090" run was on a different host). Eval of the new `model_500.pt` vs the original full-context `model_500.pt`, same val rest-start refs (20), same `hmmwv_overfit_v1.json` chrono config, no steering clamp:

| Eval backend | Run | Mean XY RMSE | Median XY RMSE | Diverged |
|---|---|---:|---:|---:|
| Chrono (ground truth) | Original (full ctx) | 0.246 m | 0.2167 m | 0 / 20 |
| Chrono (ground truth) | **K=16** | 0.330 m | **0.2168 m** | 0 / 20 |
| NN env | Original (full ctx) | 0.429 m | 0.342 m | 0 / 20 |
| NN env | **K=16** | 0.631 m | 0.324 m | 1 / 20 |

K=16 tracking quality is **on par with full context**: identical Chrono median and zero Chrono divergences. The higher K=16 means come from two hard refs (`steer_brake_s111`, `sustained_turn_s065`); on the other 18 the two runs are within a few cm. The single NN-env divergence (`steer_brake_s111`, 11.9 m in NN dynamics) is an NN-rollout artifact — that same trajectory tracks at 1.30 m in Chrono. Caveat: both are iteration 500 / 2000 (unconverged) from *different* training runs (different hardware/RNG), so the small mean gap is within run-to-run variance; re-compare at a later/converged checkpoint to confirm.

Artifacts:

- K=16 run (in progress): `artifacts/rl_runs/hmmwv_rl_15d_4090_2048env_K16/` (tmux `rl_k16`)
- K=16 NN eval: `artifacts/rl_runs/hmmwv_rl_15d_4090_2048env_K16/eval_tracking_model_500_val_rest_start/`
- K=16 Chrono eval: `artifacts/rl_runs/hmmwv_rl_15d_4090_2048env_K16/chrono_eval_tracking_model_500_val_rest_start/`
- Benchmarks: `scripts/bench_dynamics_inference.py`, `scripts/bench_context_accuracy.py`

### One-Hot Terrain-Conditioned RL 3-Terrain Chrono Eval (2026-06-25)

> **Superseded for the headline by the 2026-06-29 subsection below.** This first eval used the
> original policies (trained *without* a steering clamp) and a *mixed* eval config — rigid-flat and
> bumpy re-evaluated with `steering_rate_limit = 0.1`, CRM evaluated with no clamp. The newer
> `…_steerlim010` policies are trained *and* evaluated with `steering_rate_limit = 0.1` uniformly
> across all three terrains, and their run dirs / plots are the ones actually committed to `main`.
> The original (no-suffix) run dirs referenced just below were not synced to this box; only the
> `rigid_flat_bumpy_3policies_steerlim010_summary.json` aggregate is committed.

The terrain-conditioned dynamics/RL stack now has three comparable `model_500.pt` policies:

- **Mixture generalist**: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2/`
- **Rigid-only specialist**: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2/`
- **CRM-only specialist**: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2/`

All three use the same PPO architecture and `K=16`, 64 steps/env/update setup. Rigid-flat and bumpy Chrono evals were rerun with `steering_rate_limit = 0.1` after the bumpy runs showed that abrupt policy steering changes can trip Chrono solver/vehicle failures. CRM Chrono evals use the existing CRM runs without the rigid-terrain steering clamp. The plotted metric is **median XY RMSE** over 20 trajectories; the error bars are IQR (25th to 75th percentile). Lower is better.

| Chrono terrain | Policy | Mean XY RMSE | Median XY RMSE | IQR XY RMSE | Early terminations |
|---|---|---:|---:|---:|---:|
| Rigid flat | Mixture | 0.184 m | 0.161 m | 0.130-0.214 m | 0 / 20 |
| Rigid flat | Rigid-only | **0.158 m** | **0.143 m** | 0.116-0.166 m | 0 / 20 |
| Rigid flat | CRM-only | 0.204 m | 0.172 m | 0.167-0.239 m | 0 / 20 |
| CRM | Mixture | 0.164 m | 0.136 m | 0.100-0.195 m | 0 / 20 |
| CRM | Rigid-only | 0.786 m | 0.505 m | 0.356-0.848 m | 0 / 20 |
| CRM | CRM-only | 0.165 m | **0.131 m** | 0.120-0.209 m | 0 / 20 |
| Bumpy | Mixture | **0.182 m** | **0.146 m** | 0.104-0.216 m | 0 / 20 |
| Bumpy | Rigid-only | 0.242 m | 0.169 m | 0.128-0.332 m | 0 / 20 |
| Bumpy | CRM-only | 0.231 m | 0.202 m | 0.156-0.267 m | 0 / 20 |

Takeaway: the mixture policy is a strong generalist. It is close to the matching specialists on rigid-flat and CRM, and it is best on the held-out bumpy rigid-heightmap eval after steering-rate limiting. The rigid-only specialist still has the best rigid-flat median, and the CRM-only specialist still has the best CRM median, but each specialist degrades outside its terrain regime; the rigid-only policy is especially poor on CRM.

Artifacts:

- Plot: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_model500.png`
- PDF: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_model500.pdf`
- Stats JSON: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_model500.json`
- CSV: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_model500.csv`
- Rate-limited rigid/bumpy aggregate: `artifacts/rl_runs/chrono_eval_comparisons/rigid_flat_bumpy_3policies_steerlim010_summary.json`

### Steering-rate-limited one-hot RL — uniform 3-terrain Chrono eval (2026-06-29)

The cleaner, self-consistent version of the comparison above. Three new `model_500.pt` policies were
**trained** with `steering_rate_limit = 0.1` (env config `steering_rate_limit = 0.1`, not just an
eval-time clamp), and **all three eval terrains apply the same `steering_rate_limit = 0.1`** — so
rigid-flat, CRM, and bumpy are now directly comparable under one control regime. Same one-hot
terrain-conditioned dynamics backbones, same PPO architecture, `K=16`, 64 steps/env/update. Plotted
metric is **median XY RMSE** over 20 trajectories; error bars are IQR (25th–75th percentile). Lower is
better.

| Chrono terrain | Policy | Mean XY RMSE | Median XY RMSE | IQR XY RMSE | Early terminations |
|---|---|---:|---:|---:|---:|
| Rigid flat | Mixture | **0.168 m** | **0.125 m** | 0.109–0.215 m | 0 / 20 |
| Rigid flat | Rigid-only | 0.219 m | 0.164 m | 0.135–0.240 m | 0 / 20 |
| Rigid flat | CRM-only | 0.272 m | 0.204 m | 0.168–0.273 m | 0 / 20 |
| CRM | Mixture | **0.289 m** | 0.191 m | 0.168–0.362 m | 0 / 20 |
| CRM | Rigid-only | 0.627 m | 0.561 m | 0.374–0.809 m | 0 / 20 |
| CRM | CRM-only | 0.331 m | **0.166 m** | 0.145–0.333 m | 0 / 20 |
| Bumpy | Mixture | **0.201 m** | **0.144 m** | 0.107–0.226 m | 0 / 20 |
| Bumpy | Rigid-only | 0.274 m | 0.229 m | 0.150–0.362 m | 0 / 20 |
| Bumpy | CRM-only | 0.491 m | 0.254 m | 0.190–0.498 m | 0 / 20 |

Takeaway: under uniform steering-rate limiting the **mixture generalist is the clear best all-rounder** —
lowest mean XY RMSE on *every* terrain, and lowest median on rigid-flat and bumpy. On CRM it trails the
CRM-only specialist on median (0.191 vs 0.166 m) but beats it on mean (0.289 vs 0.331 m), i.e. the
generalist has the tighter tail. Each specialist still degrades off its training terrain — the rigid-only
policy collapses on CRM (median 0.561 m), and the CRM-only policy is worst on rigid-flat and bumpy. Note
the flip vs the 2026-06-25 mixed-clamp eval: training the policies with the clamp moves the rigid-flat
winner from the rigid-only specialist to the mixture generalist (median 0.125 vs 0.164 m), strengthening
the generalist story; all 9 cells complete 20/20 with zero early terminations.

Policies (all committed to `main`; `model_500.pt` plus full checkpoint history under each):

- Mixture generalist: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010/`
- Rigid-only specialist: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010/`
- CRM-only specialist: `artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010/`

Artifacts:

- Plot: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_model500.png`
- PDF: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_model500.pdf`
- Stats JSON: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_model500.json`
- CSV: `artifacts/rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_model500.csv`

## Bumpy-Terrain Transfer (2026-06-11)

First out-of-regime test: take the **flat-terrain-trained** `model_1999` policy and evaluate it in Chrono on **bumpy rigid-heightmap terrain** (the same `bumpy_field_*.bmp` library the 10 GB bumpy dataset was collected on, 500×500 m patches, height ±0.6 m). The Chrono env now reproduces the exact per-episode terrain: each reference's heightmap is recovered deterministically from its `episode_id` via `assign_height_map_index` (verified to match every stored `height_map_index`), and `HMMWVChronoTrackingEnv._create_sim` passes it to `create_rigid_terrain`. Setup: bumpy reference set `hmmwv_bumpy_refs_20_1100_rest_start.npz` (rest-start; 6 families — bumpy data has no launch_brake/step_steer/aggressive_*), eval config `configs/hmmwv_bumpy_eval.json`, `steering_rate_limit=0.3`, 20 m bound. See the `run-bumpy-terrain-eval` skill for the full recipe.

| Eval backend / terrain | Median XY RMSE | Mean XY RMSE | Diverged |
|---|---:|---:|---:|
| Chrono, flat terrain (smooth refs) | 0.217 m | 0.255 m | 0 / 20 |
| **Chrono, bumpy terrain (bumpy refs)** | **0.345 m** | **1.46 m** | **4 / 20** |

The flat-trained policy **does not transfer well to bumpy terrain**: mean RMSE jumps 0.26 → 1.46 m and 4/20 references diverge to the 20 m bound (refs 1 sine, 3 multi, 14 doublet, 16 chirp — all high-speed, high-travel steering maneuvers where the bumps perturb the tires most). Slow/braking refs (sustained_turn, steer_brake) still track within ~0.2–0.5 m. This is expected: both the frozen v07 NN dynamics model and the PPO policy only ever saw flat-terrain tire dynamics, so bump-induced load transfer and tire-force variation are out of distribution. **Closing this gap requires finetuning both the NN dynamics model and the policy on the bumpy dataset** — the eval harness for measuring that is now in place.

## CRM Co-Trained Generalist Dynamics Model — First Sign of Life (2026-06-18)

CRM (Continuum Representation Method — Chrono SPH/FSI deformable granular soil) is the **real**
domain shift, qualitatively harder than bumpy: the flat-trained v07 base does **not** zero-shot to
it. On a 22-episode CRM validation set its full-episode open-loop XY error is ~44% of distance
traveled (vs ~6% on flat), because CRM adds wheel slip + sinkage that rigid-terrain data never
contains (analysis in `artifacts/analysis/hmmwv_crm_15d_distribution/README.md`). Every sequential
finetune off the flat base had previously made things *worse*, so the plan switched to **co-training
one generalist** across terrains (balanced sampling + rollout-based selection) rather than
sequential finetuning.

First co-train attempt — `hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_scratch`
(75% flat / 25% CRM batches, from scratch) — exposed a pipeline bug: its loss was a plain
flat-normalized MSE, in which the CRM tire normal-force (Fz) delta is ~30× the flat per-step std →
~900× in MSE, so the loss and the `val_mixed_loss` checkpoint metric were both dominated by an
essentially-aleatoric channel. That metric *increased* as the model actually learned, so it
selected the **epoch-1** checkpoint.

Three fixes (all in `src/nedm/training/trainer.py`, backward-compatible behind config flags) →
new run `..._crm100_mix25_rebal_rollout`:
1. **Rebalanced loss** — per-channel weights from the equal-domain combined (flat+CRM) target std
   + Huber (`_build_channel_weights`, `_compute_loss`, config `loss` block). The flat-std term
   cancels, so the loss is effectively normalized by the combined scale; CRM-Fz stops dominating.
2. **Checkpoint selection on `rollout_sel`** (combined flat+CRM open-loop err/dist), not one-step val.
3. **Dual-domain rollout eval** during training (flat+CRM, 5 s / 10 s horizons; 10 s clears the
   rest-start warm-up so it measures real maneuvering).

Verification (`scripts/verify_rebal_vs_baseline.py`) uses full-episode open-loop **aggregate**
err/dist = `sqrt(Σ pos²/Σ steps) / mean episode distance` (distance-weighted, robust; the
mean-of-ratios variant is junk on CRM because immobilized episodes — e.g. one where ground truth
moves 0.1 m — blow it up):

| checkpoint | FLAT err/dist | CRM err/dist |
|---|---:|---:|
| flat-only 300 GB base (gold flat reference) | 6.1% | 44.2% |
| `mix25_scratch` baseline `last` (the run to beat) | 27.1% | 31.2% |
| generalist `best_val` (ep25, auto-selected) | 23.6% | **7.4%** |
| **generalist `last` (ep80, best all-round)** | **15.4%** | 9.4% |

Findings:

- **Sign of life confirmed.** The co-trained generalist beats the naive baseline on **both**
  domains at once — `last` (ep80) is flat 1.8× and CRM 3.3× better than the baseline. CRM forward-
  speed prediction (open-loop vx RMSE) drops from 2.46 → 0.55 m/s. The three changes work as
  intended: rebalancing let the model actually learn CRM, and rollout selection captured a
  checkpoint ~4× better on the metric that matters, where the old metric shipped epoch 1.
- **Flat tax remains** vs the dedicated flat-only base (15.4% vs 6.1%) — the expected co-training
  cost. The flat-only base + its RL policy stay the shippable flat/bumpy result; this generalist
  *adds* CRM. The baseline's flat dynamics were actually corrupted too (open-loop omega RMSE
  50 rad/s, Fz 31 kN); the generalist fixes that (omega ~5, Fz ~0.4 kN).
- **Selection metric mis-ranked.** More flat training shrank the flat tax (ep25 23.6% → ep80 15.4%)
  for a tiny CRM cost (7.4% → 9.4%), so `last` (ep80) is the better generalist — but the noisy 10 s,
  12-episode `rollout_sel` picked ep25. CRM Fz stays aleatoric (~4.7 kN, by design down-weighted).

Artifacts:

- Run: `artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_rebal_rollout/`
  (config, `logs/run.log`, 80 epochs, `checkpoints/{best_val,last}.pt`)
- Config: `configs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_rebal_rollout.json`
- Verify / plot: `scripts/verify_rebal_vs_baseline.py`, `scripts/plot_rebal_vs_baseline_overlay.py`
  → `artifacts/analysis/hmmwv_crm_15d_distribution/rebal_overlay_{crm,flat}.png`
- Writeup: `artifacts/analysis/hmmwv_crm_15d_distribution/README.md` §7

### Improvement ablations (2026-06-18)

Three single-variable ablations off `..._rebal_rollout` (each 80 epochs, same selection/eval),
to see whether the generalist could be pushed further. Trainer support added (backward-compatible):
`model_normalization` override + `_equal_domain_normalization` (combined input/output norm), and
`loss.channel_weight_overrides` (per-channel emphasis). Sweep: `scripts/{launch,run}_crm100_ablation_sweep.sh`.

Gold full-episode open-loop err/dist (aggregate; best checkpoint per run):

| run | change | FLAT | CRM | flat vx RMSE | verdict |
|---|---|---:|---:|---:|---|
| `..._rebal_rollout` (incumbent) | — | **15.4%** | 9.4% | 2.46 | best overall |
| `..._crm100_combnorm` | equal-domain combined input/output norm | 19.9% | 9.2% | 2.23 | worse flat — de-centers the dominant flat domain |
| `..._crm100_crm40` | 60/40 flat:CRM batch (vs 75/25) | 15.8% | 12.8% | 2.44 | worse both — CRM overfit |
| `..._crm100_vx3` | 3× loss weight on vx | 15.5% | **8.6%** | **1.85** | ~tie flat, marginal CRM win, tighter vx |

Findings: none is a decisive win. **More CRM batch weight ≠ better CRM** (overfits the ~96k-row
CRM set → the bottleneck is CRM *data*). **Combined input normalization is the wrong lever**
(it de-centers flat, which is 75% of the mix and where the do-no-harm bar lives; per-domain
specialization should come from *terrain conditioning*, not a shared-norm shift). **vx upweight**
is the only upside — small CRM gain at equal flat plus genuinely tighter vx — but it's fragile under
the noisy 10s/12-episode `rollout_sel`, which kept mis-ranking checkpoints (e.g. it picked vx3 ep80,
but ep71 is the good one). De-noising selection (full-episode in-loop eval, more episodes) is now a
higher-value fix than further loss tuning. (combnorm crashed once at ep11 with SIGBUS — the known
degraded-14900KF flakiness — and was resumed to 80.)

### Terrain conditioning — planned next direction (2026-06-19)

The ablations confirmed the flat tax is a *shared-network compromise* problem, not a loss-scaling or
data-ratio problem. Root cause: one network `f(state, action) → Δstate` must serve physically
different terrains — given the same observable 15-D state + action, the true delta differs by terrain
(rigid: vx ≈ ωR, no slip; CRM: vx < ωR, ~25% slip + sinkage), so with no terrain signal the net
*averages* across terrains, and that average is the flat tax. Combined-std loss fixed the loss
domination but not this. **Terrain conditioning** gives the net an explicit terrain signal so a shared
backbone can make terrain-specific predictions `f(state, action, terrain) → Δstate` — shared capacity
for kinematics/integration, specialized capacity for slip/sinkage/Fz. Expected win: flat returns
toward the flat-only gold (6.1%) while CRM keeps ~8–9%, i.e. flat tax removed without hurting CRM.

Variants, simplest → richest: (1) **one-hot terrain ID** concatenated to each token — trivial, and the
label is *free* (each training sample's source cache, flat-seq vs CRM-seq, is its terrain); (2) **FiLM**
— terrain code → per-layer scale/shift modulating hidden features (more expressive than concat);
(3) **continuous soil params** (stiffness/cohesion/density the CRM sim sets per episode) — interpolates
to unseen soils, estimable online; (4) **inferred context** — an encoder reads a short (s,a,s′) window
and infers the terrain latent from the observed slip/sinkage signature, *needing no label at inference*
(the deployment-honest version for RL/real Chrono on soft soil). Plan: start with one-hot/FiLM as a
clean A/B vs the current generalist (small changes — grow the transformer input or add a FiLM head;
tag each sub-batch with its terrain id in `mixed_infinite_loader` before the merge; supply the id per
episode at eval), then graduate to inferred context. Caveat: conditioning helps *allocation*, not
*information* — it does not reduce CRM data needs (**the 20× CRM set `hmmwv_crm_2000` has since
landed — see the 2026-06-22 subsection below**). This is design-rule #4 from the original co-train
plan, now justified by the confirmed flat tax.

### 20× CRM data collected + processed — `hmmwv_crm_2000` (2026-06-22)

The ablations above pinned the headline CRM limiter on **data, not loss/ratio tuning** (the `crm40`
batch-weight bump just overfit the ~96k-row crm_100 set). That larger set is now in hand.

- **Raw**: `artifacts/datasets/hmmwv_crm_2000` — **2000 episodes** (20× `hmmwv_crm_100`), same CRM
  collector and scenario family (`crm2000_*` prefix), identical maneuver mix (chirp 300 / doublet 200 /
  multi 600 / sine 300 / steer_brake 200 / sustained_turn 400 — same proportions as crm_100) and terrain
  (150×150 m, CRM spacing 0.08 m, depth 0.25 m). Collected via `scripts/run_hmmwv_crm2000_collection.sh`.
- **Processed (15-D, the combined-model pipeline)**:
  `artifacts/training_datasets/hmmwv_crm_2000_normal_force_omega_seq_v1` — built with the *same*
  `tire_normal_force_omega` preset as crm_100
  (`scripts/build_hmmwv_training_dataset.py --state-field-preset tire_normal_force_omega`).
  **1582 train episodes / 2,280,431 transitions; 418 val / 602,530 transitions** — ~22× the ~128k-row
  crm_100 cache, ~23.6× its 96k training rows. State/action/target/rollout fields verified
  *field-for-field identical* to `hmmwv_crm_100_normal_force_omega_seq_v1`, so it is a drop-in swap.
- **QA**: no NaN/Inf across all 2000 CSVs, no truncated episodes; episode-length and per-channel
  (actions, body velocities, yaw rate, tire forces, slip, spindle ω) distributions match crm_100
  closely. The ~18% boundary-cutoff / ~15% immobilized episodes therefore persist at similar
  *proportions*, but there is now ~20× more mobile data in absolute terms (curation still open).
- A 23-D `tire_force_omega` variant (`hmmwv_crm_2000_force_omega_seq_v1`) was also auto-built by the
  collection script; **use the 15-D `_normal_force_omega_` cache** for the flat+CRM generalist (the 23-D
  one does not match the established pipeline).

**Next step (DONE 2026-06-24 — see the next subsection): retrain the flat+CRM generalist on crm_2000.** Fork
`configs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm100_mix25_rebal_rollout.json` and swap the
**four** `hmmwv_crm_100_normal_force_omega_seq_v1` references → `hmmwv_crm_2000_normal_force_omega_seq_v1`
(`train_mix` crm dataset, `validation_datasets` crm, `loss.channel_weight_datasets[1]`, `rollout_eval`
crm dataset); the launch/run scripts default `CRM_PROCESSED_DIR` to the crm_100 cache, so override that
env var or add a `crm2000_mix25` config+script. This re-tests the flat:CRM batch-ratio question on the
20× set (the `crm40` overfit should ease) and is the do-no-harm/CRM-gain headline for Milestone 4. Run
on Euler, not the degraded local box.

### Flat+CRM generalist retrained on crm_2000 + one-hot terrain conditioning — three-way ablation (2026-06-24)

The two planned next-steps above — **retrain the generalist on `hmmwv_crm_2000`** and **add terrain
conditioning** — landed together. The new generalist implements **one-hot terrain conditioning**
(variant 1 of the planned direction): a 2-D one-hot `[flat, crm]` code is concatenated to every
(state, action) token, growing the transformer input 18 → 20. The label is free (each `train_mix`
source is one terrain) and is supplied per-domain at window/rollout eval. Everything else matches the
`crm_100` incumbent recipe (flat-based input/output normalization, equal-domain-combined-std Huber
loss, `rollout_sel` checkpoint selection, 80 epochs × 2000 steps), with the four `crm_100` → `crm_2000`
references swapped.

To isolate the contribution of co-training, two **single-domain ablation arms** were trained with
*identical* architecture / normalization / loss — only the train mix differs:

| model | run dir suffix | train mix | checkpoint selection |
|---|---|---|---|
| 75/25 generalist | `…crm2000_mix25_rebal_rollout_onehot` | 75% flat / 25% CRM | combined 0.5·flat + 0.5·CRM |
| flat-only | `…crm2000_mix00_rebal_rollout_onehot` | 100% flat | flat only |
| CRM-only | `…crm2000_mix100_rebal_rollout_onehot` | 100% CRM | CRM only |

All three keep the one-hot `[flat, crm]` head so `input_dim` stays 20 (the specialists simply always
emit their own one-hot); each specialist keeps the off-domain rollout/val at weight 0 for monitoring
only, so selection is in-domain.

**Three-way result** (best-`rollout_sel` checkpoint, open-loop 10s rollout err/dist; off-domain = zero-shot):

| model | flat | CRM |
|---|---|---|
| 75/25 generalist | **9.1%** | **5.8%** |
| flat-only | 5.0% | 69.2% (zero-shot) |
| CRM-only | 37.2% (zero-shot) | 3.7% |

Headline: the one-hot generalist improves on the `crm_100` `_rebal_rollout` incumbent on **both**
domains (flat 15.4% → 9.1% on the same flat val; CRM 9.4% → 5.8% on the larger crm_2000 val) — the 20×
data + explicit terrain signal together largely close the flat tax (flat-only gold ≈ 5–6%) while
pushing CRM well below 9%. Each specialist still wins its own domain (flat 5.0 < 9.1; CRM 3.7 < 5.8) and
collapses on the other (flat-only → CRM 69%, CRM-only → flat 37%), confirming that one shared network
*can't* serve both terrains for free: co-training buys cross-domain robustness at a ~3–4 pt
peak-accuracy cost. (CRM eval episodes differ between crm_100 and crm_2000 val, so the CRM
incumbent comparison is indicative, not strict.)

**Eval metric (`rollout_sel` / err-dist) — the numbers above.** Once per epoch, for each terrain we draw
**12 val-split episodes** and roll the dynamics model **open-loop** over a **10 s horizon** (1000 steps
at dt = 0.01 s): the model's own predicted body-frame Δstate is fed back each step and integrated to a
predicted (x, y, yaw) pose. We take the RMS xy position error vs ground truth over all rollout steps
(`xy_rmse_m`), then divide by the mean ground-truth path length per episode (`mean_dist_m`) to get
**`errdist = xy_rmse / mean_dist`** — a dimensionless, distance-normalized open-loop position error
(reported as %). Normalizing by distance traveled is what makes flat and CRM comparable, since CRM
episodes are shorter/slower (≈ 25 m vs ≈ 35 m of ground-truth path per 10 s). The scalar
**`rollout_sel`** used for checkpoint selection is the weight-averaged 10 s `errdist` across the model's
selection domains, and `best_val.pt` is the epoch that minimizes it (here ep 69 / 67 / 70 for
generalist / flat-only / CRM-only). The table cells are the per-domain `errdist` at each model's
`best_val.pt`. Also logged per epoch but **not** the headline here: one-step window loss (Huber, channel
re-weighted) and 5 s rollout err/dist.

Caveats: these are **in-loop training rollout numbers on the val split**, not Chrono closed-loop
transfer; and `best_val` is selected by peeking at those same rollout episodes, so it carries a mild
optimistic bias vs a fully disjoint test set (use a disjoint reference set or `last.pt` for an unbiased
read). For downstream RL use **`best_val.pt`** on all three — it beats `last.pt` on every model (largest
gap on the generalist, `rollout_sel` 0.074 vs 0.139).

**Provenance.** Run dirs under
`artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_{mix25,mix00,mix100}_rebal_rollout_onehot/`.
All three pushed to `main` (checkpoints via Git LFS; `last.pt` force-added past `.gitignore`): commits
`a239ff6` (one-hot code in `model.py`/`trainer.py`/`rl/dynamics.py` + flat-only run + both ablation
configs/scripts), `a196a29` (75/25 generalist), `7aa90c9` (CRM-only run). Processed datasets stay
gitignored — rsync `hmmwv_tire_rigid_300g_normal_force_omega_seq_v1` +
`hmmwv_crm_2000_normal_force_omega_seq_v1` to retrain elsewhere (and `git lfs pull` to fetch the real
checkpoints). Trained on the local box this time; both ablation runs completed cleanly.

## Arm Mobile-Manipulator Study (Case 2): Arm Dynamics Model + Reaching RL

Second NeDM study case, parallel to the HMMWV traversal work above: a 4-DOF gripper arm welded to the front of an M113 tracked vehicle. The goal is **locomanipulation** — drive the base until a target enters the arm's workspace, then reach it with the arm — split into a **drive mode** and a **reach mode** so we never have to learn the full coupled vehicle+arm dynamics at once. Design doc: [arm-dyn-model.md](arm-dyn-model.md). This milestone delivers the **reach-mode arm dynamics model** `f_arm` and the plan to train the reaching policy on it. Chrono-side collector details: `src/nedm/arm_data.py` and project memory `arm-dynamics-data-collection`.

### Arm dynamics data + quality (2026-06-30)

Raw data: `artifacts/datasets/arm_dynamics_v3_home_reset_fulltraj_shards/` — 15 shards × 1,000 episodes = **15,000 episodes / 920,640 transitions / 799 MB**, 50 Hz control (`control_dt = 0.02 s`), base pinned after settle, **torque + per-substep PD** joint actuation (so `q` genuinely lags `qcmd` — real dynamics, not a hard angle constraint). Each CSV row is a full transition: `q, qd, qcmd, act (=Δqcmd), q_next, qd_next, ee` (world) and `ee_base` (arm-base frame), plus `collision`/`collision_kind`.

Quality probe (episode stats over all 15k + per-channel stats from a 1,200-episode CSV sample):

- **Dynamics are non-degenerate.** PD tracking gap `|qcmd−q|` mean/joint = [0.37, 0.10, 0.065, 0.043] rad; `|qd|` mean/joint = [2.06, 1.02, 0.95, 0.85] rad/s; `qcmd+act==qcmd_next` exactly. The torque/PD swap worked — there is dynamics to learn.
- **Plenty of windows at context 16.** A 16-step window needs ≥17 rows; yield **703,027 windows** (train 592k / val 111k) from 65.8% of episodes. (block 8 → 800k from 100% of eps; block 32 → 573k from 51%.)
- **Coverage skew — the one caveat that shapes the RL phase.** 71% of episodes terminate on **ground** contact and 34% are <17 rows: the home pose + random downward shoulder commands drive the arm into the floor fast, so trajectories are short and the **lower workspace is under-sampled** — shoulder `q_1` only covers [−0.61, +1.57] of its ±1.57 rad limit (base yaw `q_0` covers full ±π; elbow/wrist full ±1.57). EE (world) z ∈ [0.16, 7.28] m, all above ground. ⟹ **reaching goals must be sampled in the well-covered upper/forward workspace**, not deep-down/near-ground.

### Arm dynamics NN model `f_arm` (2026-06-30)

Reuses the HMMWV training stack **unchanged** (`model.py`, `model_transformer.py`, `dataset.py`, `trainer.py`, `rl/dynamics.py`) — the same GPT-style causal continuous-token transformer, only the dims and context differ. Two tiny backward-compatible edits to `preprocess.py` were all the arm data needed: (1) `compute_dt_s` falls back to the dataset index's `control_dt_s` when there is no `collector_config.resolved.json` (arm dt = 0.02 s; without this it silently returned 0.01); (2) `main()` defaults the missing `scenario_family` from `collision_kind`.

I/O design (state == target, 15-D):

| Group | Fields | Dim |
|---|---|---:|
| state == target | `q0..3, qd0..3, qcmd0..3, ee_base_{x,y,z}` | 15 |
| action | `act0..3` (applied Δqcmd) | 4 |
| context (`block_size`) | 16 steps ≈ 0.32 s @ 50 Hz | — |

- **`qcmd` is in the state** because during a step the PD targets `qcmd_{t+1}=qcmd_t+act_t`; the model needs the absolute command, not just the increment (the increment alone is ambiguous across window boundaries).
- **`ee_base` is predicted as a state channel** so the reaching RL reads the end-effector straight from the model with **no torch arm-FK** (the reward is `‖ee−goal‖`).
- Because the pipeline ties `target_dim==state_dim`, the model also predicts `Δqcmd` — a trivial identity (`Δqcmd≈act`; its `target_std` equals `action_std`). Harmless, and at RL rollout the qcmd channels are overwritten deterministically rather than trusted.

Config `configs/arm_transformer_v1.json`: compact 4-layer / 4-head / 128-embd transformer, `dropout 0`, no `rollout_eval` (the HMMWV rollout integrator is pose-specific and would KeyError on arm fields; checkpoint on windowed `val_loss`). Processed cache `artifacts/training_datasets/arm_dyn_v3_seq16_v1` (763,886 train / 141,754 val transitions; `rollout_fields = ee_base`). Run dir `artifacts/training_runs/arm_transformer_v1`.

Results — best `val_loss = 0.00181` @ epoch 30 (converged monotonically, not overfit):

| Metric | Value |
|---|---|
| 1-step val RMSE | q 0.3–1.0 mrad · qd 0.009–0.043 rad/s · ee_base 0.0014–0.0029 |
| Open-loop EE drift `errdist` (EE err ÷ EE travel) | **1.9% @0.25 s · 1.9% @0.5 s · 2.7% @1 s · 4.0% @2 s** |

Eval via `scripts/eval_arm_rollout.py` (its own open-loop EE-drift rollout over held-out val episodes — seeds 16 steps, rolls `next = state + predict_next_delta`, compares predicted `ee_base` to ground truth; `artifacts/training_runs/arm_transformer_v1/rollout_eval.json`). The displacement-normalized drift holds ~2% over the horizons a reaching policy plans over → the model is **RL-ready**. (Absolute EE units are the collector's 2×-scaled-arm meters; `errdist` is the scale-invariant headline. Chrono sim-to-real transfer is validated in Phase 4.) **Not yet committed** — local edits + artifacts on `main`.

### Reaching RL plan — Phase 4 (planned, not started)

Train a **reach policy** `π_reach` entirely inside the frozen `f_arm` (base fixed, matching the data), then validate transfer in Chrono. Follows [arm-dyn-model.md](arm-dyn-model.md) §7, §9.

- **Env** — new `src/nedm/rl/arm_reaching_env.py` (rsl_rl `VecEnv`), **simpler** than the HMMWV tracking env: no world-pose integration and no reference trajectory, just a goal point. Reuse `load_frozen_dynamics` and the `state_hist`/`action_hist` roll + `predict_next_delta` substep pattern from `hmmwv_tracking_env.py`.
- **Obs:** `q, qd, qcmd, goal_base, ee_base, goal−ee, d_safe_min`.
- **Action (4):** `Δqcmd`, scaled by per-joint `DQ_MAX`; the env computes `qcmd_next = clip(qcmd+act)` **deterministically** and writes it into the qcmd channels (ignores the model's qcmd prediction — exact, drift-free). EE for the reward is read directly from the model's predicted `ee_base`.
- **Reward:** `−w·‖ee−goal‖ − w·‖a‖² − w·‖Δa‖² − collision penalty + success bonus`; success when `‖ee−goal‖ < ε` for several consecutive steps.
- **Goals:** sampled in the **well-covered upper/forward workspace** (per the coverage caveat above), not near-ground.
- **Safety filter (doc §7):** lightweight geometric self/track/ground check applied before each model step (block + penalize unsafe). Separate module, needed for RL but not for the dynamics model; Chrono stays the ground-truth checker only during data collection / final validation.
- **Then:** `scripts/train_arm_rl_reaching.py` (mirror `train_hmmwv_rl_tracking.py`), and a Chrono reaching-validation env (parallel to `hmmwv_chrono_tracking_env.py`) to confirm `π_reach` reaches goals on the real simulator with a fixed base.
- **Later (out of scope for v1):** the **drive** policy + base dynamics model (HMMWV-style), and the rule-based **mode selector** that switches drive↔reach (doc §10–§12).

### Drive-mode data collection pipeline (2026-07-01)

Base-motion data collector for the drive-mode NN-ROM described in [tracked_vehicle_nn_rom_rl_plan.md](tracked_vehicle_nn_rom_rl_plan.md): `src/nedm/tracked_vehicle_data.py` (`scripts/collect_tracked_vehicle_dataset.py` wrapper), driven by `configs/tracked_vehicle_drive_v1.json`.

Reuses rather than rebuilds. The M113+arm+terrain scene is exactly `arm_data.build_scene()` — the real deployed mounted-arm configuration, so the base ROM's mass/inertia/CG match reality instead of a bare M113 — with the arm's four `ChLinkMotorRotationAngle` motors left completely untouched: no angle target is ever set, so they hold `q=0` (home) as a hard constraint and the arm rides as fixed dead weight ("drive mode" per [arm-dyn-model.md](arm-dyn-model.md) §3.1), needing neither the PD actuator nor the arm collision setup `arm_data.py` builds for reach-mode data. The maneuver library (launch/brake, coast-down, steering arcs, S-turns, pivot-like turns, brake-while-steering, broad random commands, stop-and-go — plan §4.3) reuses the HMMWV collector's `scenario_generator`/driver-profile machinery (`generated_scenarios.py`, `hmmwv_data.sample_channel`) unchanged; the tracked-vehicle-specific families are config-level template overrides (`pivot_like_turn`/`coast_down`/`stop_and_go_straight` reuse the `step_steer`/`launch_brake`/`multi_steer` builders with different parameter ranges), not new generator code. Driver commands are evaluated directly from the continuous profile at each recorded step rather than through a `ChDataDriver` table, dropping the `driver_sample_step_s` knob entirely.

Logged per row at 50 Hz (`record_step_s=0.02`; physics stepped at the same `STEP_SIZE=5e-4` the single-pin track needs): pose, quaternion, roll/pitch/yaw, world+body velocity/acceleration, world+body angular velocity, speed, roll/yaw rate, left/right sprocket speed, driver commands. A 0.5 s braked settle (matching `arm_data.py`'s handling of this same track model) runs before each scenario's own clock starts, so `time_s` in the CSV is scenario-local, not wall/sim-absolute. Per-row divergence guards (non-finite state, >45° roll/pitch, >45 m from origin) flag and truncate rare solver hiccups instead of silently polluting the dataset.

`configs/tracked_vehicle_drive_v1.json` materializes 540 episodes across 10 families. Verified so far with ad hoc single-episode runs, not yet the full 540-episode collection (that's cluster-only — see below): sustained throttle=0.8 accelerates 0→3.9 m/s and is still climbing at 10 s (M113 top speed ≈16 m/s, so this is plausible, not capped/broken); a steering step to ±0.4 flips `yaw_rate_radps` sign with the expected inertial lag; throttle-then-release with no brake gives a smooth multi-second coast-down decay rather than an instant stop. Zero divergences across 5 test episodes. Each episode currently costs tens of seconds of wall-clock (e.g. a 13.7 s episode took ~52 s), so collecting the full 540-episode config is a cluster job (`create-euler-script` skill / Euler), never a local run.

Two things surfaced while testing:

- **Terrain-size fix.** `arm_data.build_scene()`'s flat patch is a genuine finite rigid box (`RigidTerrain.AddPatch`), not an infinite plane. Its 100 m default is fine for reach-mode (base pinned, never moves) but far too small once the base actually drives — a 30 s sustained-throttle test reached pos_y≈45 m, which the old 100 m patch (and a first-pass 45 m bounds guard) would have falsely flagged as diverged/out-of-bounds. `build_scene()` now takes a `terrain_size_m` param (default 100 unchanged, so reach-mode/arm data collection is unaffected); the drive-mode collector passes `TERRAIN_SIZE_M = 600` and derives its divergence-guard bound from that.
- **Zero-steering curving is expected, not a bug.** With `driver_steering` held at exactly 0.0 for a full 30 s run, the vehicle still curves substantially (left/right sprocket speed diverge, yaw rate grows to ~0.25 rad/s) — confirmed via the logged columns, not a scenario-evaluation bug. User confirmed this is a known asymmetry (likely the welded arm's mass/CG), so it's left as-is: the ROM should learn it as real behavior, and "steering=0" families (`coast_down`, `stop_and_go_straight`) mean *commanded* straight, not *actually* straight.

## Open Items / Next Steps

- **Arm reaching RL (Phase 4 — next for case 2)**: build `src/nedm/rl/arm_reaching_env.py` on the frozen `f_arm` and train `π_reach`, then validate in Chrono — full plan in the "Arm Mobile-Manipulator Study" section. Sample goals in the upper/forward workspace (the lower workspace is under-sampled). Also: commit the Phase 0–3 arm work (currently local on `main`).
- **Tracked-vehicle drive-mode base ROM (next for the base side of case 2)**: the data collection pipeline (`src/nedm/tracked_vehicle_data.py`, `configs/tracked_vehicle_drive_v1.json`) is built and spot-verified — see "Drive-mode data collection pipeline" above. Next: (1) scale the 540-episode config up on Euler (small-scale local runs cost tens of seconds per episode), (2) preprocess to `[vx, vy, r]` state / `[throttle, steering, brake]` action windows via the existing `training/preprocess.py` (field-name agnostic, no code changes needed), (3) train the memoryless residual MLP `f_base` per plan §6, (4) build the goal-reaching RL env per plan §8.
- **Braking transfer gap**: the policy tracks turning references in Chrono but diverges on braking-heavy ones — likely a dynamics-model gap (brake response) rather than a policy gap; worth checking v07 open-loop rollout error on launch_brake/steer_brake segments specifically.
- **v19–v30 sweep** crashed at the first model and was never completed.
- **RL on alternate dynamics backbones**: the current active policy uses the 15-D tire-normal-force/omega v07-style model; the older `v3_turn_300g` backbone remains untested as an RL backbone.
- **15-D held-out validation outliers**: held-out/rest-start eval now exists for the 15-D policy. The NN env and Chrono env both run, but the validation set has several NN outliers; compare `xy_mean_m` to training logs and inspect per-reference behavior before drawing conclusions from aggregate RMSE alone.
- **Bumpy-terrain finetune (next major step)**: the flat-trained policies regress on bumpy terrain. The older 7-D/v07 check degraded from mean 0.26 → 1.46 m with 4/20 divergences, and the newer 15-D `model_500.pt` check degraded from mean 0.25 → 0.62 m in Chrono. Plan: (1) finetune the current 15-D tire-normal-force/omega NN dynamics model on `hmmwv_bumpy_10g_normal_force_omega_seq_v1`, then (2) finetune/retrain the PPO policy against that bumpy dynamics model, and (3) re-run the 15-D bumpy NN + Chrono eval to measure recovery. The 15-D bumpy cache, reference sets, eval helper config, and per-episode terrain reproduction are all done.
- **Beyond the fixed regime**: bumpy-terrain *evaluation* now exists (heightmap terrain reproduced per episode); friction variation, observation noise, and tire-channel supervision are still out of scope.
- **CRM generalist follow-ups** (informed by the 2026-06-18 ablations): (1) **de-noise checkpoint selection** — make the in-loop rollout eval full-episode (or longer horizon) with more episodes so `rollout_sel` stops mis-ranking (it picked ep25/vx3-ep80 when the gold metric prefers different epochs); highest-value fix. (2) **More CRM data — LANDED 2026-06-22** — `hmmwv_crm_2000` (2000 eps) collected and processed to `hmmwv_crm_2000_normal_force_omega_seq_v1` (~2.28M train / 0.60M val transitions, ~23.6× the crm_100 training rows; see the dedicated subsection above). The `crm40` ablation showed more CRM *batch weight* just overfits the ~96k-row set (CRM 9.4%→12.8%), so the limiter was data, not weight — now addressable. **Actionable next step: retrain the flat+CRM generalist swapping crm_100→crm_2000** (4 config refs) and re-test the flat:CRM ratio on the 20× set. (Still ~18% boundary cutoffs / ~15% immobilized episodes to curate.) (3) **Terrain conditioning** (one-hot → FiLM → inferred context) for per-domain specialization to attack the flat tax — see the dedicated subsection above; combined input normalization (`combnorm`) is the *wrong* lever (it de-centers the dominant flat domain → flat 15.4%→19.9%). (4) Keep the **vx loss upweight** (marginal CRM win + tighter vx, free). (5) Extend flat+CRM to the full **tri-domain** (add bumpy) generalist; train/eval a PPO policy against it and run CRM Chrono transfer.
