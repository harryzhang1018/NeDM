# Lessons Learned

Running notes on what worked and what to change next time. Add to this file as new lessons land.

## 1. Prefer the 2048env_unbuf RL configuration when retraining

The configuration in `artifacts/rl_runs/hmmwv_rl_tracking_v07_20260607_2048env_unbuf` produced a better tracking policy than `artifacts/rl_runs/hmmwv_rl_tracking_v07_8192env_16steps_term1m_20260608`. Use it as the starting point for retraining.

The two runs differ in exactly three settings:

| Setting | 2048env_unbuf (better) | 8192env_16steps_term1m |
|---|---|---|
| `num_envs` | 2048 | 8192 |
| `num_steps_per_env` (PPO rollout length) | 128 | 16 |
| Training termination `max_position_error_m` | 20.0 | 1.0 |

Everything else (reward weights, action repeat 5, obs history/preview 10, PPO hyperparameters, v07 dynamics checkpoint, 20 rest-start references) is identical.

Why this combination is plausibly better: 128-step rollouts cover ~6.4 s of the 9 s episode per PPO iteration (vs 0.8 s with 16 steps), so credit assignment sees whole maneuver segments; and the loose 20 m termination lets the policy learn recovery from large errors instead of being reset the moment it drifts 1 m.

Evidence (Chrono eval, 20 rest-start references, eval termination 20 m):

- pychrono 10.0: model_1150 (2048env_unbuf) mean 1.08 m / 2 diverged vs model_1999 (8192env) mean 1.36 m / 3 diverged.
- model_1150 also tracks `aggressive_sine_steer_00341` (0.33 m) where model_1999 bogs down mid-corner and diverges to 20 m.

## 2. Add a hard termination on sudden steering changes during RL training

When training RL, terminate the episode if the steering command jumps too far in one policy step — e.g. terminate when `|steer_t - steer_{t-1}| > 0.5`. Near-instant steering reversals can break/destabilize the Chrono solver when the policy is later evaluated on the real simulator, and they are physically unrealistic driver inputs.

Current setup only discourages this softly (`action_rate_weight: 0.02` in the reward), which still allows large single-step steering offsets. A hard termination makes the constraint binding, so the trained policy never relies on steering slews that Chrono cannot tolerate.

Implementation note: the env already tracks both tensors — `self.actions` and `self.last_actions` in `src/nedm/rl/hmmwv_tracking_env.py` (line ~155). Add `(torch.abs(self.actions[:, 0] - self.last_actions[:, 0]) > 0.5)` to the existing position/roll/pitch termination expression (line ~353), and make the threshold an entry in the `termination` config dict so it lands in `env_cfg.json` for reproducibility.

**Empirically validated (2026-06-09).** A steering rate-limit *filter* in the Chrono eval env (`steering_rate_limit` cfg option / `--steering-rate-limit` flag: clamp steering to ±threshold of the previous policy step's command) at 0.3 fixed every model_1999 divergence under pychrono 10 with no cost to the good references:

| 20-ref Chrono eval (model_1999, pychrono 10) | mean | median | diverged |
|---|---:|---:|---:|
| no filter | 1.360 m | 0.280 m | 3 |
| steering rate limit 0.3 | **0.255 m** | **0.217 m** | **0** |

Even `steer_brake_s010`, which diverged under both pychrono 9 and 10 (~6.8 m), drops to 0.68 m. This confirms abrupt steering reversals are the root cause of the tire-saturation stalls — so the training-side fix (this lesson) should pay off, and the eval-side filter is a safe deployment guard in the meantime.
