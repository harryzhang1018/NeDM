# HMMWV Sequence Training Pipeline

This is the first NN training pipeline for the dataset in [artifacts/datasets/hmmwv_overfit_6k](/home/harry/NeDM/artifacts/datasets/hmmwv_overfit_6k).

It is intentionally narrow:

- one vehicle: `HMMWV_Full`
- one terrain: flat rigid
- one friction regime: `mu = 0.9`
- one contact/tire setup: `SMC` + `TMEASY`

The point of this first model is not broad generalization. It is to learn a strong sequential baseline on one consistent vehicle-dynamics regime.

## Model Type

The model is a causal transformer with GPT-style decoder blocks over continuous-valued tokens.

The implementation is in:

- [model_transformer.py](/home/harry/NeDM/src/nedm/training/model_transformer.py)
- [model.py](/home/harry/NeDM/src/nedm/training/model.py)
- [trainer.py](/home/harry/NeDM/src/nedm/training/trainer.py)

This matches the design direction used in the `neural-robot-dynamics` reference repo: a causal transformer over fixed-length state/action sequences rather than shuffled single rows.

## Input And Output

The model uses one token per logged timestep at `100 Hz`.

Per-step input state:

- `vel_body_x_mps`
- `vel_body_y_mps`
- `roll_rad`
- `pitch_rad`
- `roll_rate_radps`
- `ang_vel_body_y_radps`
- `yaw_rate_radps`

Per-step control:

- `driver_steering`
- `driver_throttle`
- `driver_braking`

So the input token dimension is `10`.

The model predicts the next-step delta of the state channels:

- `delta_vel_body_x_mps`
- `delta_vel_body_y_mps`
- `delta_roll_rad`
- `delta_pitch_rad`
- `delta_roll_rate_radps`
- `delta_ang_vel_body_y_radps`
- `delta_yaw_rate_radps`

So the output token dimension is `7`.

This is a sequential input/output model:

- input: a context window of `T` state-action tokens
- output: `T` predicted delta-state tokens

During rollout, the model is used autoregressively: predicted next state is fed back as the next state token, while controls come from the known command sequence.

## Why This State Definition

This first state is local and dynamic rather than global:

- body-frame velocities are better learning targets than world-frame `x/y`
- roll, pitch, and rate channels capture important load-transfer dynamics
- yaw rate is more stable to learn than absolute yaw angle

The first model intentionally does not directly predict:

- `pos_x_m`, `pos_y_m`
- `yaw_rad`
- `body_slip_rad`

`x/y/yaw` are reconstructed during rollout evaluation by integrating predicted body velocity and yaw rate. `body_slip_rad` is available in the raw data, but it becomes unstable near zero forward speed and is not a good primary supervised target for the first pass.

## Pipeline Stages

1. Raw episode CSVs are converted into compact train/val arrays with [build_hmmwv_training_dataset.py](/home/harry/NeDM/scripts/build_hmmwv_training_dataset.py).
2. The processed dataset stores contiguous `states`, `actions`, `targets`, episode boundaries, and normalization statistics.
3. [train_hmmwv_dynamics.py](/home/harry/NeDM/scripts/train_hmmwv_dynamics.py) samples fixed-length windows from those arrays and trains the transformer on normalized delta-state loss.
4. Validation reports:
   - one-step sequence RMSE on held-out windows
   - open-loop rollout RMSE over `1 s`, `2 s`, and `5 s`
5. [eval_hmmwv_rollout.py](/home/harry/NeDM/scripts/eval_hmmwv_rollout.py) can reload a checkpoint and rerun validation later.

## Default Run

The default config is [hmmwv_transformer_v1.json](/home/harry/NeDM/configs/hmmwv_transformer_v1.json).

Build the processed dataset:

```bash
conda activate tutorial
python scripts/build_hmmwv_training_dataset.py \
  --dataset-root artifacts/datasets/hmmwv_overfit_6k \
  --output-dir artifacts/training_datasets/hmmwv_overfit_6k_seq_v1
```

Train the model:

```bash
conda activate tutorial
python scripts/train_hmmwv_dynamics.py --config configs/hmmwv_transformer_v1.json
```

Run rollout evaluation from a saved checkpoint:

```bash
conda activate tutorial
python scripts/eval_hmmwv_rollout.py \
  --checkpoint artifacts/training_runs/hmmwv_transformer_v1/checkpoints/best_val.pt
```

## Current Scope

This first pipeline is deliberately conservative:

- deterministic simulation only
- no observation noise model
- no tire-channel supervision
- no terrain variation
- no friction variation
- no multi-vehicle training

That is appropriate for a first overfit-oriented baseline. Once this is stable, the next extension should be adding friction and terrain variation while keeping the same sequential training structure.
