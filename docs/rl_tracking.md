# NN Dynamics RL Tracking

This package trains a trajectory-tracking policy against a frozen HMMWV neural dynamics model. The policy controls driver steering, throttle, and brake at a slower rate than the NN model: by default one policy action is held for 5 NN steps.

Build the compact 20-trajectory reference set:

```bash
conda activate tutorial
python scripts/build_hmmwv_rl_references.py
```

The default output is:

```text
artifacts/rl_reference_sets/hmmwv_train_refs_20_1100_seed_20260607.npz
```

It contains 20 fixed-length training-set reference segments, 1100 NN transitions each. The default set is short enough to include the shorter straight/step/aggressive maneuver families while still giving roughly 9 seconds of policy-controlled tracking after the v07 model context window.

Train PPO with the default v07 dynamics checkpoint:

```bash
conda activate tutorial
python scripts/train_hmmwv_rl_tracking.py \
  --device cuda \
  --num-envs 1024 \
  --max-iterations 2000
```

Swap the frozen NN dynamics checkpoint with:

```bash
python scripts/train_hmmwv_rl_tracking.py \
  --dynamics-checkpoint artifacts/training_runs/hmmwv_transformer_v04_long_baseline_b32/checkpoints/best_val.pt
```

Evaluate and plot a trained policy:

```bash
python scripts/eval_hmmwv_rl_tracking.py \
  --run-dir artifacts/rl_runs/<run-name>
```

Evaluate the same policy against the real Chrono HMMWV model:

```bash
python scripts/eval_hmmwv_rl_chrono_tracking.py \
  --run-dir artifacts/rl_runs/<run-name> \
  --policy-checkpoint artifacts/rl_runs/<run-name>/model_50.pt \
  --device cpu
```

The Chrono evaluator is intentionally for policy evaluation only. It creates a Chrono HMMWV using the same vehicle and terrain setup as the data-collection pipeline, initializes the rollout near each reference pose and forward speed, applies the policy's steering/throttle/brake commands through `DriverInputs`, advances Chrono at the collector simulation step size, and then reads back the same 7 NN state fields from the vehicle.

The vectorized environment is implemented in `src/nedm/rl/hmmwv_tracking_env.py`. It keeps batched state/action history buffers on the selected device, runs the frozen NN model in batched inference, updates state with `next_state = current_state + predicted_delta`, and integrates pose from body velocity and yaw rate using the same convention as the dynamics rollout evaluator.
