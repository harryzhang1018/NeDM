# Tracked Vehicle NN-ROM and RL Goal-Reaching Plan

## 1. Purpose

This plan defines a simplified tracked-vehicle reduced-order neural-network dynamics model and a reinforcement-learning setup for goal reaching in the NeDM locomanipulation project.

The main design choice is to keep the tracked-vehicle case simpler than the earlier HMMWV-style vehicle model. The immediate RL objective is not full trajectory tracking. The objective is to learn goal-reaching behavior using a compact NN-ROM of the tracked base.

The arm dynamics NN-ROM and arm-side RL can remain separate for now. The tracked vehicle should first become a reliable, low-dimensional base-motion model that can support RL policies for moving the platform to a desired goal pose or goal region.

---

## 2. Scope and Key Simplifications

### In scope

- Tracked base motion on mostly flat terrain.
- Goal reaching rather than reference trajectory tracking.
- Low-dimensional vehicle dynamics state.
- Continuous control actions: throttle, steering, braking.
- NN-ROM trained from simulation or logged vehicle-dynamics data.
- RL policy trained against the NN-ROM environment.

### Out of scope for the first version

- Full HMMWV-style suspension, wheel, tire, or high-fidelity multibody state.
- Detailed terrain-map input.
- Full vehicle-arm coupling.
- Obstacle avoidance.
- Exact path tracking.
- Manipulation while moving.

### Design philosophy

Start with the smallest model that can support meaningful goal reaching:

\[
\text{NN-ROM: } (v_x, v_y, r, u_{thr}, u_{steer}, u_{brake}) \rightarrow (v_x^+, v_y^+, r^+)
\]

where:

- \(v_x\): body-frame longitudinal velocity
- \(v_y\): body-frame lateral velocity
- \(r\): yaw rate
- \(u_{thr}\): throttle command
- \(u_{steer}\): steering command, interpreted as left-right track differential or normalized turn command
- \(u_{brake}\): braking command

Vehicle global pose is not part of the NN dynamics state. Instead, pose is propagated using a simple kinematic integrator outside the NN-ROM:

\[
\dot{x} = v_x \cos\psi - v_y \sin\psi
\]

\[
\dot{y} = v_x \sin\psi + v_y \cos\psi
\]

\[
\dot{\psi} = r
\]

This keeps the learned dynamics compact while still allowing the RL environment to compute distance-to-goal and heading-to-goal.

---

## 3. Recommended Baseline Model

### 3.1 State input to the NN-ROM

Use the compact body-frame velocity state:

```text
s_t = [v_x, v_y, r]
```

Recommended units:

```text
v_x: m/s
v_y: m/s
r: rad/s
```

### 3.2 Action input to the NN-ROM

Use the normalized actuator command vector:

```text
u_t = [throttle, steering, brake]
```

Recommended ranges:

```text
throttle ∈ [0, 1]
steering ∈ [-1, 1]
brake    ∈ [0, 1]
```

Interpretation:

- `throttle = 0`: no propulsion command.
- `throttle = 1`: maximum forward propulsion command.
- `steering = -1`: maximum left turn or left/right track differential in one direction.
- `steering = 1`: maximum right turn or opposite track differential.
- `brake = 0`: no braking.
- `brake = 1`: maximum braking.

If reverse motion is needed later, use one of these extensions:

```text
Option A: throttle ∈ [-1, 1]
Option B: keep throttle ∈ [0, 1] and add reverse/gear command
Option C: use left_track_cmd and right_track_cmd instead of throttle/steering
```

For the first goal-reaching model, use forward throttle only unless the task requires reversing.

### 3.3 NN-ROM output

Prefer residual prediction instead of direct next-state prediction:

```text
NN input:  [v_x, v_y, r, throttle, steering, brake]
NN output: [Δv_x, Δv_y, Δr]
```

Then update:

```text
v_x_next = v_x + Δv_x
v_y_next = v_y + Δv_y
r_next   = r   + Δr
```

Residual prediction is usually easier to stabilize because the network learns local changes instead of the entire state value.

### 3.4 Optional short-history input

Tracked vehicles can have contact and actuator memory effects. If the memoryless model is too weak, use a short history window:

```text
z_t = [s_t, s_{t-1}, s_{t-2}, u_t, u_{t-1}, u_{t-2}]
```

This keeps the model simple while giving the network enough information to infer short-term acceleration, braking lag, and skid behavior.

Recommended progression:

```text
Version 0: memoryless MLP
Version 1: MLP with 2-3 step history
Version 2: GRU/LSTM only if history MLP fails
```

---

## 4. Data Collection Plan

The vehicle NN-ROM should be trained on broad control excitation rather than expert trajectories. Because RL will use the ROM for goal reaching, the model needs to understand how actions change velocity and yaw rate across the operating envelope.

### 4.1 Data source

Use the best available source in this order:

1. High-fidelity tracked-vehicle simulation logs.
2. Existing Chrono/vehicle simulator logs, if available.
3. Real vehicle logs, if safe and available.
4. Hybrid data: simulation first, real data later for fine-tuning.

### 4.2 Logged variables

Minimum required log fields:

```text
time
x, y, yaw
v_x, v_y, yaw_rate
throttle, steering, brake
```

Useful additional fields:

```text
left_track_speed or left_track_command
right_track_speed or right_track_command
terrain type
slope estimate
normal load estimate
slip estimate
contact flags
```

The first ROM should not require the additional fields. They are useful for debugging and future extensions.

### 4.3 Maneuver library

Collect data from simple, diverse maneuvers:

| Maneuver type | Purpose |
|---|---|
| Straight acceleration | Learn throttle-to-speed response |
| Coast-down | Learn passive drag and decay |
| Straight braking | Learn brake response |
| Constant steering arcs | Learn steady turning behavior |
| S-turns | Learn steering reversal dynamics |
| Pivot-like turns | Learn high-yaw-rate behavior |
| Throttle plus steering sweeps | Learn coupled speed/yaw behavior |
| Brake while steering | Learn stopping during turns |
| Random smooth commands | Improve coverage for RL |
| Stop-and-go commands | Learn low-speed behavior |

### 4.4 Action excitation

Use smooth randomized commands rather than purely white-noise commands. White noise may overrepresent unrealistic actuator switching.

Recommended command generation:

```text
sample target throttle, steering, brake every 0.5-2.0 s
apply first-order smoothing or rate limits
hold command briefly
repeat over many randomized episodes
```

Suggested constraints:

```text
avoid high throttle and high brake simultaneously in most samples
include some throttle-brake overlap only if the real controller allows it
include low-speed and near-zero-speed samples heavily
include steering near zero as well as saturated steering
```

### 4.5 Dataset coverage targets

The dataset should cover:

```text
v_x: from 0 to expected maximum goal-reaching speed
v_y: expected lateral slip range
r: expected yaw-rate range
throttle: 0 to 1
steering: -1 to 1
brake: 0 to 1
```

For the initial RL model, prioritize data quality around:

```text
low speed
starting from rest
moderate forward motion
turning while moving
braking near the goal
```

These regions matter most for goal reaching.

---

## 5. Preprocessing Plan

### 5.1 Coordinate frame

Convert velocity to the vehicle body frame before training:

```text
v_x = forward velocity in body frame
v_y = lateral velocity in body frame
r   = yaw rate
```

Do not train directly on global \(\dot{x}\), \(\dot{y}\), because the same local vehicle behavior appears different depending on yaw angle.

### 5.2 Time step

Choose a fixed model time step:

```text
Recommended NN-ROM Δt: 0.05 s to 0.10 s
Recommended RL control Δt: 0.10 s to 0.20 s
```

A practical setup is:

```text
NN-ROM integration step: 0.05 s
RL action hold: 2-4 NN steps
RL effective control rate: 5-10 Hz
```

### 5.3 Filtering and cleaning

Apply the following preprocessing checks:

```text
remove corrupted samples
unwrap yaw before differentiating
clip impossible velocity or yaw-rate spikes
check sign convention for yaw rate and steering
smooth only if the raw signal is noisy; avoid over-smoothing real dynamics
normalize all inputs and outputs using training-set statistics
```

### 5.4 Train/validation/test split

Split by episodes, not by individual time steps. This prevents near-identical adjacent samples from appearing in both train and test sets.

Recommended split:

```text
70% training episodes
15% validation episodes
15% test episodes
```

Hold out at least one maneuver family for generalization testing, for example:

```text
train on random steering sweeps and arcs
test on S-turns or stop-and-go goal-like episodes
```

---

## 6. NN-ROM Training Plan

### 6.1 Baseline architecture

Start with a small MLP:

```text
input dimension: 6 for memoryless model
hidden layers: 3-4
hidden width: 128-256
activation: SiLU, GELU, or ReLU
output dimension: 3
output target: [Δv_x, Δv_y, Δr]
```

Recommended initial model:

```text
MLP(6 → 128 → 128 → 128 → 3)
```

For the history version with 3 state/action steps:

```text
input dimension = 3 states × 3 + 3 actions × 3 = 18
MLP(18 → 256 → 256 → 256 → 3)
```

### 6.2 Loss function

Use weighted MSE on residual dynamics:

```text
L_one_step = w_vx MSE(Δv_x_pred, Δv_x_true)
           + w_vy MSE(Δv_y_pred, Δv_y_true)
           + w_r  MSE(Δr_pred,   Δr_true)
```

Because \(v_y\) may be smaller in magnitude than \(v_x\), normalize targets before training or use weights to prevent the network from ignoring lateral velocity and yaw-rate behavior.

Suggested starting weights after normalization:

```text
w_vx = 1.0
w_vy = 1.0
w_r  = 1.0
```

If training on unnormalized targets:

```text
increase w_vy and w_r until turning behavior is captured well
```

### 6.3 Multi-step rollout loss

After one-step training works, add a short rollout loss:

```text
L_total = L_one_step + λ L_rollout
```

Suggested rollout horizon:

```text
H = 10-30 steps
```

Suggested rollout weight:

```text
λ = 0.1 initially
increase if open-loop rollouts drift too quickly
```

This matters because RL will repeatedly roll the model forward, so stable multi-step behavior is more important than only one-step accuracy.

### 6.4 Physical regularization

Add simple checks or penalties if needed:

```text
brake should generally reduce v_x when moving forward
zero throttle and zero brake should not create strong acceleration
large steering should affect yaw rate more than straight speed
velocity and yaw-rate outputs should remain within plausible bounds
```

Do not over-constrain the first model. Use these as diagnostics first, then penalties only if the model learns physically impossible behavior.

### 6.5 Ensemble option

Train an ensemble of 3-5 small networks if RL exploits model errors.

Uses:

```text
estimate model uncertainty
randomly select one ensemble member per episode during RL
penalize high-disagreement states
detect out-of-distribution rollouts
```

This is useful when the RL policy discovers unrealistic actions that were not well covered in the training data.

---

## 7. NN-ROM Validation Plan

Validation should focus on whether the ROM is good enough for goal-reaching RL, not whether it perfectly matches every high-fidelity detail.

### 7.1 One-step metrics

Report:

```text
RMSE(v_x_next)
RMSE(v_y_next)
RMSE(r_next)
MAE(v_x_next)
MAE(v_y_next)
MAE(r_next)
```

Also report normalized errors:

```text
NRMSE relative to standard deviation of each target
```

### 7.2 Open-loop rollout metrics

Evaluate 2-5 second rollouts using held-out action sequences.

Report:

```text
velocity rollout error over time
yaw-rate rollout error over time
integrated x-y-yaw error over time
failure cases by maneuver type
```

### 7.3 Goal-relevant validation

Use simple scripted controllers or random action sequences to test whether the ROM supports goal-reaching behavior.

Test cases:

```text
start from rest, goal straight ahead
start from rest, goal left/right
moving start, goal ahead
moving start, need to brake near goal
small heading correction
large heading correction
```

The ROM is acceptable for first RL experiments if it preserves these qualitative behaviors:

```text
throttle increases forward motion
brake reduces forward motion
steering changes yaw rate with correct sign
turning radius is plausible
vehicle can slow down near a goal
model remains stable during repeated rollout
```

---

## 8. RL Environment Design

The RL environment should wrap the NN-ROM and add pose integration, goal definition, reward calculation, and termination conditions.

### 8.1 Environment state for RL policy

The NN-ROM state is only:

```text
[v_x, v_y, r]
```

The RL observation should include goal information:

```text
obs_t = [
    v_x,
    v_y,
    r,
    x_goal_body,
    y_goal_body,
    distance_to_goal,
    sin(heading_error),
    cos(heading_error),
    previous_throttle,
    previous_steering,
    previous_brake
]
```

Here:

```text
x_goal_body, y_goal_body = goal position expressed in vehicle body frame
heading_error = desired goal heading - current yaw
```

Using body-frame goal coordinates makes the policy easier to learn because the goal direction is represented relative to the vehicle.

### 8.2 RL action

Use the same action vector as the NN-ROM:

```text
a_t = [throttle, steering, brake]
```

Action bounds:

```text
throttle ∈ [0, 1]
steering ∈ [-1, 1]
brake    ∈ [0, 1]
```

Recommended action post-processing:

```text
apply rate limits
clip actions to bounds
optionally suppress brake when throttle is high
optionally suppress throttle when brake is high
```

A simple brake-throttle conflict rule:

```text
if brake > 0.2:
    effective_throttle = throttle * (1 - brake)
else:
    effective_throttle = throttle
```

This prevents the RL policy from exploiting unrealistic simultaneous full throttle and full brake behavior.

### 8.3 Pose propagation

At each NN-ROM step:

```text
[v_x_next, v_y_next, r_next] = NN_ROM([v_x, v_y, r], action)

x_next   = x + Δt * (v_x_next cos(yaw) - v_y_next sin(yaw))
y_next   = y + Δt * (v_x_next sin(yaw) + v_y_next cos(yaw))
yaw_next = yaw + Δt * r_next
```

Use normalized yaw wrapping:

```text
yaw ∈ [-π, π]
```

### 8.4 Goal definition

Start with position-only goal reaching:

```text
success if distance_to_goal < d_tol
```

Recommended initial tolerance:

```text
d_tol = 0.5-1.0 m
```

Then add heading-aware reaching:

```text
success if distance_to_goal < d_tol and abs(heading_error) < yaw_tol
```

Recommended heading tolerance:

```text
yaw_tol = 10-20 degrees
```

### 8.5 Reward function

Use dense progress reward plus terminal success reward.

Recommended first reward:

```text
r_t = k_progress * (d_prev - d_curr)
    - k_dist     * d_curr
    - k_yaw      * abs(heading_error)
    - k_action   * ||a_t||^2
    - k_smooth   * ||a_t - a_{t-1}||^2
    - k_spin     * max(0, abs(r) - r_safe)^2
    + r_success
```

Suggested starting values:

```text
k_progress = 5.0
k_dist     = 0.1
k_yaw      = 0.05
k_action   = 0.01
k_smooth   = 0.05
k_spin     = 0.05
r_success  = 50.0
```

For position-only goals, set `k_yaw = 0` initially.

### 8.6 Termination conditions

Terminate an episode if:

```text
goal is reached
maximum episode time is reached
vehicle moves too far from the workspace
state becomes numerically invalid
velocity or yaw rate exceeds safe model range
```

Recommended first episode length:

```text
10-20 seconds
```

### 8.7 Recommended RL algorithm

Use a continuous-action off-policy algorithm:

```text
Primary recommendation: SAC
Alternative: TD3
Alternative for simpler implementation: PPO with continuous actions
```

SAC is recommended because it is usually robust for continuous control and benefits from replay-buffer sample efficiency.

---

## 9. RL Training Curriculum

### Stage 0: NN-ROM sanity environment

Goal:

```text
confirm the environment rolls forward correctly
```

Use random actions and scripted actions. Plot:

```text
x-y path
yaw over time
v_x, v_y, r over time
actions over time
```

### Stage 1: Easy forward goals

Initial condition:

```text
vehicle starts at rest
vehicle yaw = 0
goals sampled in front of vehicle
```

Goal sampling:

```text
x_goal ∈ [2, 8] m
y_goal ∈ [-1, 1] m
```

Success criterion:

```text
distance_to_goal < 1.0 m
```

Expected learned behavior:

```text
accelerate forward
reduce throttle near goal
use brake if needed
```

### Stage 2: 2D local goals

Goal sampling:

```text
x_goal ∈ [2, 10] m
y_goal ∈ [-5, 5] m
```

Expected learned behavior:

```text
steer toward target
control speed during turning
brake near target
```

### Stage 3: Wider goal distribution

Goal sampling:

```text
x_goal ∈ [-5, 12] m
y_goal ∈ [-8, 8] m
```

Only include behind-the-vehicle goals if reverse motion or pivot-turn behavior is supported.

Expected learned behavior:

```text
turn in place or arc-turn before approaching target
avoid excessive spinning
```

### Stage 4: Heading-aware goals

Add desired final yaw:

```text
yaw_goal ∈ [-π, π]
```

Success criterion:

```text
distance_to_goal < d_tol
abs(heading_error) < yaw_tol
```

Expected learned behavior:

```text
approach goal while aligning final heading
```

### Stage 5: Robustness and domain randomization

Randomize:

```text
initial velocity
initial yaw rate
mass or drag coefficient if represented
NN ensemble member
action delay
control-rate variation
small process noise
```

This reduces policy brittleness and helps prevent overfitting to one exact NN-ROM.

---

## 10. Integration Path Toward Locomanipulation

The tracked base should first be validated alone. Then integrate with the arm side gradually.

### Phase A: Base-only NN-ROM and RL

Deliverable:

```text
tracked base can reach local goals reliably in the NN-ROM environment
```

### Phase B: Base plus frozen arm configuration

Keep the arm fixed in a neutral pose. Treat the vehicle mass/inertia as constant.

Deliverable:

```text
base goal-reaching policy still works when deployed in the combined simulation wrapper
```

### Phase C: Base with arm configuration as disturbance

If arm motion noticeably changes base dynamics, add compact arm context to the base ROM:

```text
optional extra inputs:
arm center-of-mass offset
arm joint summary
arm payload flag
base mass/inertia mode
```

Avoid this until base-only performance is proven.

### Phase D: Coordinated locomanipulation

Use the base policy as a primitive:

```text
move base to reachable manipulation region
stop or stabilize
execute arm policy
optionally reposition base
```

This keeps the problem modular and avoids training one large policy too early.

---

## 11. Risk Register and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| State is too simple | Good one-step error but bad rollouts | Add short history of states/actions |
| RL exploits NN errors | Policy finds unrealistic spinning or braking behavior | Use ensemble, uncertainty penalty, action limits, data expansion |
| Poor low-speed behavior | Policy fails near goal | Oversample start/stop and braking data |
| Brake/throttle conflict | Policy uses both commands unrealistically | Add action post-processing or penalty |
| Steering sign error | Policy turns away from goals | Validate with scripted steering tests |
| Distribution mismatch | RL visits states absent from data | Add OOD detector and iterative data collection |
| Open-loop drift | Rollouts diverge after a few seconds | Add multi-step rollout loss |
| Model ignores lateral velocity | Turning looks wrong despite good v_x | Normalize/weight v_y and yaw-rate losses |
| Unstable yaw rate | Excessive spinning | Clip actions, penalize spin, expand turn data |
| Hard transfer back to simulator | Policy works in ROM but not high-fidelity sim | Validate periodically in high-fidelity sim and fine-tune data |

---

## 12. Iterative Development Milestones

### Milestone 1: Data and preprocessing pipeline

Deliverables:

```text
logged vehicle maneuver dataset
body-frame velocity conversion
fixed-step resampling
train/val/test split by episode
normalization statistics
basic dataset coverage plots
```

Exit criteria:

```text
commands and states have correct signs
coverage includes straight, turning, braking, and low-speed data
no obvious corrupted samples
```

### Milestone 2: First memoryless NN-ROM

Deliverables:

```text
MLP residual dynamics model
one-step validation metrics
open-loop rollout plots
scripted maneuver comparisons
```

Exit criteria:

```text
correct qualitative response to throttle, steering, and brake
stable 2-5 second rollouts on held-out maneuvers
```

### Milestone 3: History NN-ROM if needed

Deliverables:

```text
short-history MLP model
comparison against memoryless model
rollout improvement report
```

Exit criteria:

```text
turning, braking, and stop-go behavior improve over memoryless baseline
```

### Milestone 4: NN-ROM RL environment

Deliverables:

```text
Gymnasium-style environment wrapper
pose integration
goal sampler
reward function
termination logic
random-action test plots
scripted-controller sanity tests
```

Exit criteria:

```text
environment produces reasonable trajectories
reward increases when moving toward goal
success condition triggers correctly
```

### Milestone 5: RL goal-reaching policy

Deliverables:

```text
SAC or TD3 training script
training curves
success-rate evaluation
failure-case rollouts
policy videos or path plots
```

Exit criteria:

```text
high success rate on local 2D goals
reasonable braking or slowing near goals
no obvious model exploitation
```

### Milestone 6: High-fidelity simulator validation

Deliverables:

```text
policy replay in high-fidelity tracked-vehicle simulator
ROM-vs-sim trajectory comparison
failure-case dataset for retraining
```

Exit criteria:

```text
policy transfers qualitatively to the higher-fidelity simulator
major failure modes are identified and fed back into dataset collection
```

---

## 13. Recommended First Implementation Checklist

1. Define normalized action interface:

```text
u = [throttle, steering, brake]
```

2. Define NN-ROM state:

```text
s = [v_x, v_y, r]
```

3. Build data logger with:

```text
time, x, y, yaw, v_x, v_y, r, throttle, steering, brake
```

4. Generate maneuver dataset:

```text
straight, turning, braking, S-turns, random smooth commands, stop-go
```

5. Train memoryless residual MLP:

```text
[s_t, u_t] -> Δs_t
```

6. Validate one-step and rollout behavior.

7. If rollouts are weak, train short-history MLP:

```text
[s_t, s_{t-1}, s_{t-2}, u_t, u_{t-1}, u_{t-2}] -> Δs_t
```

8. Wrap NN-ROM in RL environment with pose integration.

9. Train SAC on position-only goal reaching.

10. Add heading goals, domain randomization, and high-fidelity validation only after the simple position-goal task works.

---

## 14. Proposed Repository Structure

```text
tracked_vehicle_rom/
  data/
    raw/
    processed/
    splits/
  configs/
    rom_memoryless.yaml
    rom_history.yaml
    rl_goal_reach.yaml
  scripts/
    collect_maneuvers.py
    preprocess_logs.py
    train_rom.py
    eval_rom_rollout.py
    train_rl_goal_reach.py
    eval_rl_policy.py
  src/
    rom/
      dataset.py
      model.py
      train.py
      rollout.py
    envs/
      tracked_goal_env.py
      reward.py
      goal_sampler.py
    utils/
      frames.py
      normalization.py
      plotting.py
  outputs/
    checkpoints/
    figures/
    logs/
```

---

## 15. Baseline Configuration Draft

```yaml
rom:
  dt: 0.05
  state: [vx, vy, yaw_rate]
  action: [throttle, steering, brake]
  action_bounds:
    throttle: [0.0, 1.0]
    steering: [-1.0, 1.0]
    brake: [0.0, 1.0]
  prediction_type: residual
  history_steps: 1
  model:
    type: mlp
    hidden_sizes: [128, 128, 128]
    activation: silu
  training:
    batch_size: 1024
    learning_rate: 0.0003
    max_epochs: 300
    early_stopping_patience: 25
    loss: weighted_mse
    rollout_loss:
      enabled: false
      horizon: 20
      weight: 0.1

rl_env:
  control_dt: 0.10
  nn_steps_per_action: 2
  observation:
    - vx
    - vy
    - yaw_rate
    - goal_x_body
    - goal_y_body
    - distance_to_goal
    - sin_heading_error
    - cos_heading_error
    - previous_throttle
    - previous_steering
    - previous_brake
  action: [throttle, steering, brake]
  goal:
    position_tolerance_m: 0.75
    heading_enabled: false
    heading_tolerance_deg: 15
  episode:
    max_time_s: 15.0
    workspace_radius_m: 25.0
  reward:
    progress_weight: 5.0
    distance_weight: 0.1
    heading_weight: 0.0
    action_weight: 0.01
    smoothness_weight: 0.05
    spin_weight: 0.05
    success_bonus: 50.0
  algorithm:
    name: SAC
    total_steps: 1000000
    replay_buffer_size: 1000000
```

---

## 16. Main Recommendation

Use this initial formulation:

```text
NN-ROM input:  [v_x, v_y, yaw_rate, throttle, steering, brake]
NN-ROM output: [Δv_x, Δv_y, Δyaw_rate]
RL observation: [vehicle velocities, body-frame goal, heading error, previous action]
RL action:      [throttle, steering, brake]
RL objective:   reach local 2D goal region first, add heading later
```

This gives a clean base-motion model that is much simpler than a full HMMWV-style model, while still containing the minimum dynamics needed for tracked-vehicle goal reaching.
