# Arm Dynamics Model and Two-Mode RL Plan

## 1. Purpose of this document

This document describes the proposed data-collection, dynamics-model training, and RL practice for the tracked-vehicle mobile-manipulator case study.

The main design decision is to avoid learning full simultaneous vehicle-plus-arm coupled dynamics in the first version. Instead, we split the problem into two modes:

1. **Drive mode**: the tracked vehicle moves while the arm is held in a safe folded pose.
2. **Reach mode**: the vehicle is stopped while the 4-DOF arm moves to reach the target.

This makes data collection, model training, and RL training significantly easier while still solving the main locomanipulation behavior:

```text
far goal  -> drive vehicle until target enters arm workspace -> stop -> reach with arm
near goal -> keep vehicle still -> reach with arm
```

The full controller is therefore a hybrid controller:

```text
mode selector + drive policy + reach policy
```

The first version should use a rule-based mode selector. A learned mode selector can be added later.

---

## 2. Overall architecture

```text
Chrono vehicle-only data
        ↓
Train base dynamics model f_base

Chrono arm-only data
        ↓
Train arm dynamics model f_arm

f_base + f_arm
        ↓
Hybrid learned simulator

Train policies:
    π_drive: steering / throttle / brake
    π_reach: 4 arm joint commands

Rule-based mode selector:
    Drive if target outside arm workspace
    Reach if target inside arm workspace and base nearly stopped

Final validation:
    run complete switched controller in Chrono
```

The important point is that the learned dynamics models are trained separately:


a. base-only model:

\[
x^b_{t+1} = f_b(x^b_t, a^b_t)
\]

b. arm-only model:

\[
x^a_{t+1} = f_a(x^a_t, a^a_t)
\]

The full hybrid learned environment combines them using the current mode.

---

## 3. Mode definitions

### 3.1 Drive mode

In drive mode, the arm is held fixed in a safe folded pose.

Action:

\[
a^b_t = [steering, throttle, brake]
\]

Objective:

\[
d_{ws} \rightarrow 0
\]

where \(d_{ws}\) is the distance from the target to the approximate reachable workspace of the arm, expressed relative to the vehicle base.

In this mode:

```text
base moves
arm does not move
```

The drive policy learns to move the vehicle so that the target becomes reachable by the arm.

---

### 3.2 Reach mode

In reach mode, the vehicle is stopped or heavily braked, and only the arm moves.

Action:

\[
a^a_t = [\Delta q^{cmd}_1, \Delta q^{cmd}_2, \Delta q^{cmd}_3, \Delta q^{cmd}_4]
\]

Objective:

\[
p_{ee} \rightarrow p_{goal}
\]

In this mode:

```text
base is held still
arm moves
```

The reach policy learns 4-DOF arm control for end-effector position reaching.

---

## 4. Recommended training strategy

The recommended first implementation is to train two separate policies:

```text
π_drive: 3D action, trained only on drive-mode task
π_reach: 4D action, trained only on reach-mode task
```

This is preferred over training one combined 6D or 7D policy with action masking.

A single masked policy could output:

\[
[steering, throttle, q_1, q_2, q_3, q_4]
\]

and then the environment could ignore part of the action depending on the mode. However, this makes learning less efficient because the policy continuously outputs actions that may be discarded.

The cleaner first version is:

```text
train π_drive separately
train π_reach separately
combine them with a mode selector
```

Later, a high-level learned mode policy can be trained:

\[
\pi_{mode}(o_t) \rightarrow \{Drive, Reach\}
\]

while keeping the low-level policies fixed or fine-tuning them jointly.

---

## 5. Arm dynamics model

This section focuses on the arm dynamics model, since the main question is how to collect arm data, train the model, and handle self-collision.

### 5.1 What the arm model should learn

The arm dynamics model should learn safe free-space arm motion only.

It should not learn self-collision dynamics.

The target mapping is:

\[
(q_t, \dot q_t, q^{cmd}_t, a^a_t) \rightarrow (q_{t+1}, \dot q_{t+1}, q^{cmd}_{t+1})
\]

or, preferably, a delta form:

\[
\Delta \hat{x}^a_t = f_a(x^a_t, a^a_t)
\]

\[
\hat{x}^a_{t+1} = x^a_t + \Delta \hat{x}^a_t
\]

where:

\[
x^a_t = [q_t, \dot q_t, q^{cmd}_t]
\]

and:

\[
a^a_t = [\Delta q^{cmd}_1, \Delta q^{cmd}_2, \Delta q^{cmd}_3, \Delta q^{cmd}_4]
\]

The command update is:

\[
q^{cmd}_{t+1} = clip(q^{cmd}_t + \Delta q^{cmd}_t, q_{min}, q_{max})
\]

Then a low-level PD controller in Chrono drives the physical joints toward \(q^{cmd}\).

---

### 5.2 Recommended arm-model input

For a simple feedforward model:

\[
input_t = [q_t, \dot q_t, q^{cmd}_t, a^a_t]
\]

where:

| Term | Dimension | Meaning |
|---|---:|---|
| \(q_t\) | 4 | current arm joint angles |
| \(\dot q_t\) | 4 | current arm joint velocities |
| \(q^{cmd}_t\) | 4 | current joint command targets |
| \(a^a_t\) | 4 | policy output, usually command increments |

Total input dimension:

\[
4 + 4 + 4 + 4 = 16
\]

For better robustness, use a short-history sequence model:

\[
input_t = [x^a_{t-K:t}, a^a_{t-K:t}]
\]

Example with history length \(K=4\):

```text
[q_{t-4}, qdot_{t-4}, qcmd_{t-4}, action_{t-4}]
[q_{t-3}, qdot_{t-3}, qcmd_{t-3}, action_{t-3}]
[q_{t-2}, qdot_{t-2}, qcmd_{t-2}, action_{t-2}]
[q_{t-1}, qdot_{t-1}, qcmd_{t-1}, action_{t-1}]
[q_t,   qdot_t,   qcmd_t,   action_t]
```

The sequence version is recommended if actuator delay, motor lag, or controller memory becomes important.

---

### 5.3 Recommended arm-model output

The model should predict the next-state delta:

\[
output_t = \Delta x^a_t
\]

where:

\[
\Delta x^a_t = [\Delta q_t, \Delta \dot q_t, \Delta q^{cmd}_t]
\]

Total output dimension:

\[
4 + 4 + 4 = 12
\]

Then reconstruct:

\[
\hat{x}^a_{t+1} = x^a_t + \Delta \hat{x}^a_t
\]

For \(q^{cmd}\), we may also update it deterministically instead of predicting it:

\[
q^{cmd}_{t+1} = clip(q^{cmd}_t + a^a_t, q_{min}, q_{max})
\]

If we do this, the model only predicts:

\[
[\Delta q_t, \Delta \dot q_t]
\]

and the output dimension becomes 8.

Recommended first version:

```text
input:  [q, qdot, qcmd, action]
output: [Δq, Δqdot]
qcmd_next is computed deterministically
```

This is simpler and avoids forcing the neural network to learn something that is already known.

---

## 6. Arm data collection in Chrono

### 6.1 Goal of data collection

The goal is to collect safe free-space arm transitions:

\[
(x^a_t, a^a_t, x^a_{t+1})
\]

The vehicle base is fixed during this data collection.

```text
base pose fixed
base velocity = 0
arm moves with safe random commands
```

---

### 6.2 Arm data-collection loop

```python
for episode in range(num_episodes):
    env.reset_base_fixed()
    q = sample_safe_initial_arm_pose()
    env.set_arm_pose(q)
    qcmd = q.copy()

    for step in range(max_steps):
        x_t = env.get_arm_state()  # q, qdot, qcmd

        raw_action = sample_random_arm_action()
        safe_action, safety_info = safety_filter(q, qcmd, raw_action)

        if not safety_info.is_safe:
            # Option A: resample
            raw_action = sample_random_arm_action()
            safe_action, safety_info = safety_filter(q, qcmd, raw_action)

            # Option B: shrink/project action
            # safe_action = project_to_safe_action(q, qcmd, raw_action)

        env.apply_arm_action(safe_action)
        env.step(control_dt)

        x_next = env.get_arm_state()

        chrono_collision = env.check_chrono_collision()

        if chrono_collision:
            # Do not train the free-space dynamics model on this transition.
            # Save it separately for debugging the safety filter.
            save_collision_case(x_t, safe_action, x_next)
            break

        save_transition(x_t, safe_action, x_next)
```

The saved action must be the action that was actually applied:

```text
save safe_action, not raw_action
```

---

### 6.3 Random action design

Avoid independent white-noise commands at every step. Use smooth random commands so the data looks like realistic actuator usage.

Recommended:

```text
sample target joint command
move toward target with bounded Δq per step
occasionally resample target
```

Example:

\[
a^a_t = clip(q^{target} - q^{cmd}_t, -\Delta q_{max}, \Delta q_{max})
\]

with:

\[
\Delta q_{max} = 1^\circ \text{ to } 5^\circ \text{ per control step}
\]

The exact value should be chosen based on control frequency and actuator limits.

Use multiple data regimes:

| Regime | Purpose |
|---|---|
| small random motions | local linear behavior |
| large but safe motions | workspace coverage |
| near joint limits | learn boundary behavior |
| slow motions | stable reaching behavior |
| faster motions | velocity response |

---

## 7. Self-collision handling

### 7.1 Key principle

Self-collision should not be part of the learned arm dynamics model.

The model should learn:

```text
safe free-space arm motion
```

not:

```text
link-link collision/contact dynamics
```

Collision behavior is discontinuous and contact-rich. Including self-collision in the training data would make the dynamics model harder to train and less useful for RL.

Therefore:

```text
safety is handled outside the learned dynamics model
```

---

### 7.2 Use both Chrono and a custom safety filter

Use two collision systems for two different purposes.

#### Chrono collision detection

Use Chrono as the ground-truth collision checker during:

```text
data collection
final validation
debugging the safety filter
```

In Chrono, define collision geometry for:

```text
arm links
end effector
arm base/mount
vehicle chassis
```

Use simple shapes when possible:

```text
capsules
cylinders
boxes
spheres
```

Avoid relying on detailed visual meshes for collision checking unless necessary.

#### Custom lightweight safety filter

Use a custom geometric safety filter during:

```text
learned-model RL training
fast batched rollout
```

The reason is speed. RL training inside the learned dynamics model should not call the full Chrono simulator at every step.

The safety filter should check:

```text
joint limits
self-collision
arm-base collision
motion interpolation between q_t and proposed q_next
```

---

### 7.3 Safety-filter geometry

Approximate the arm using simple geometry:

```text
link 1 -> capsule or cylinder
link 2 -> capsule or cylinder
link 3 -> capsule or cylinder
link 4 -> capsule or cylinder
end effector -> sphere or short capsule
vehicle body -> box or set of boxes
arm mount -> box/cylinder
```

Check non-adjacent link pairs, for example:

```text
link 1 vs link 3
link 1 vs link 4
link 2 vs link 4
link 2 vs vehicle body
link 3 vs vehicle body
link 4 vs vehicle body
end effector vs vehicle body
```

Adjacent links usually share joints, so their collision pairs can often be ignored or handled with special margins.

A configuration is unsafe if:

\[
d_{ij} < d_{safe}
\]

where \(d_{ij}\) is the minimum distance between two collision primitives and \(d_{safe}\) is a safety margin.

Recommended first margin:

\[
d_{safe} = 0.03m \text{ to } 0.05m
\]

---

### 7.4 Check the proposed motion, not only the current pose

The filter should not only check \(q_t\). It should check the proposed movement from current command to next command.

Given:

\[
q^{cmd}_{next} = q^{cmd}_t + a^a_t
\]

check interpolated configurations:

\[
q^{check}_\alpha = (1-\alpha)q^{cmd}_t + \alpha q^{cmd}_{next}
\]

for:

\[
\alpha \in \{0, 0.25, 0.50, 0.75, 1.0\}
\]

If any interpolated configuration is unsafe, reject or modify the action.

---

### 7.5 Data collection safety policy

During Chrono data collection:

```text
raw random action
        ↓
custom safety pre-filter
        ↓
apply safe action in Chrono
        ↓
Chrono collision check
        ↓
if no collision: save transition
if collision: discard transition and save as safety-filter failure case
```

If the custom safety filter says safe but Chrono detects collision, that means the custom filter is not conservative enough. Increase the safety margin or improve the primitive geometry.

---

### 7.6 RL training safety policy

During learned-model RL training:

```text
policy action
        ↓
custom safety filter
        ↓
if safe:
    step learned arm model
else:
    block or project action
    apply penalty
    optionally terminate episode
```

Recommended first version:

```text
unsafe action -> block action + penalty
```

For example:

\[
a^{safe}_t = 0
\]

and:

\[
r_t \leftarrow r_t - w_{collision}
\]

Alternative:

```text
unsafe action -> project to nearest safe action + smaller penalty
```

This is smoother but harder to implement.

The first version can simply block unsafe actions.

---

## 8. Arm dynamics model loss

### 8.1 State prediction loss

The core supervised loss is:

\[
\mathcal{L}_{state}
=
\left\|W_x(\Delta \hat{x}^a_t - \Delta x^a_t)\right\|^2
\]

where \(W_x\) balances different units and magnitudes.

For the simple output version:

\[
\Delta x^a_t = [\Delta q_t, \Delta \dot q_t]
\]

---

### 8.2 End-effector auxiliary loss

Because the RL task is end-effector reaching, add an auxiliary loss on predicted end-effector position.

Use forward kinematics:

\[
\hat{p}_{ee,t+1} = FK(\hat{q}_{t+1})
\]

Compare against the Chrono end-effector position:

\[
\mathcal{L}_{ee}
=
\left\|\hat{p}_{ee,t+1} - p_{ee,t+1}\right\|^2
\]

Total loss:

\[
\mathcal{L}
=
\mathcal{L}_{state}
+
\lambda_{ee}\mathcal{L}_{ee}
\]

Recommended first version:

```text
train with state loss first
then add EE auxiliary loss if rollout EE error is too high
```

---

## 9. Reach policy training

### 9.1 Reach-policy observation

The reach policy should observe enough information to move the arm to the goal while avoiding unsafe configurations.

Recommended observation:

\[
o^a_t = [q_t, \dot q_t, q^{cmd}_t, p^{base}_{goal}, p^{base}_{ee}, p_{goal} - p_{ee}, d_{safe,min}]
\]

where:

| Term | Meaning |
|---|---|
| \(q_t\) | current joint angles |
| \(\dot q_t\) | current joint velocities |
| \(q^{cmd}_t\) | current joint command targets |
| \(p^{base}_{goal}\) | goal expressed in arm/base frame |
| \(p^{base}_{ee}\) | end-effector position in base frame |
| \(p_{goal} - p_{ee}\) | Cartesian reaching error |
| \(d_{safe,min}\) | minimum safety distance to self/base collision |

For angles, use either raw joint angles normalized by limits or sine/cosine encoding.

---

### 9.2 Reach-policy action

The policy outputs normalized actions:

\[
a^a_t \in [-1,1]^4
\]

Convert to command increments:

\[
\Delta q^{cmd}_t = \Delta q_{max} a^a_t
\]

Then:

\[
q^{cmd}_{t+1}=clip(q^{cmd}_t+\Delta q^{cmd}_t,q_{min},q_{max})
\]

The safety filter is applied before the learned model step.

---

### 9.3 Reach reward

Recommended reward:

\[
r^a_t =
-w_{ee}\|p_{ee}-p_{goal}\|
-w_a\|a^a_t\|^2
-w_{\Delta a}\|a^a_t-a^a_{t-1}\|^2
-w_{near}\sum_{i,j}\max(0,d_{safe}-d_{ij})^2
-w_{col}\mathbb{1}_{unsafe}
+r_{success}
\]

where:

```text
first term: reach the target
second term: avoid excessive joint commands
third term: smooth control
fourth term: stay away from collision boundaries
fifth term: penalize unsafe actions
sixth term: success bonus
```

Success condition:

\[
\|p_{ee}-p_{goal}\| < \epsilon_{ee}
\]

for several consecutive steps.

Recommended first threshold:

\[
\epsilon_{ee} = 0.03m \text{ to } 0.05m
\]

---

## 10. Drive policy training

Although this document focuses on the arm dynamics model, the drive policy is part of the full system.

### 10.1 Drive-model input/output

Base dynamics model:

\[
x^b_{t+1}=f_b(x^b_t,a^b_t)
\]

where:

\[
x^b_t = [p_b, yaw_b, v_b, \omega_b, previous\ action]
\]

and:

\[
a^b_t = [steering, throttle, brake]
\]

For the learned model, use local-frame deltas:

\[
output = [\Delta p^{base}_b, \Delta yaw_b, \Delta v_b, \Delta \omega_b]
\]

This can follow the same pipeline used in the HMMWV traversing problem.

---

### 10.2 Drive-policy observation

Recommended drive observation:

\[
o^b_t = [p^{base}_{goal}, v_b, \omega_b, d_{ws}, yaw\ error]
\]

The key term is:

\[
d_{ws}=dist(p^{base}_{goal}, \mathcal{W}_{arm})
\]

The drive policy should learn to reduce \(d_{ws}\), not directly reach the end-effector target.

---

### 10.3 Drive reward

\[
r^b_t=
-w_{ws}d_{ws}
-w_v\|a^b_t\|^2
-w_{\Delta a}\|a^b_t-a^b_{t-1}\|^2
-w_{roll}|roll|
-w_{pitch}|pitch|
+r_{workspace}
\]

Success for drive mode:

\[
d_{ws} < \epsilon_{ws}
\]

and:

\[
\|v_b\| < v_{threshold}
\]

This means the vehicle has moved to a region where the arm can reach the target and is nearly stopped.

---

## 11. Rule-based mode selector

The first version should use a rule-based mode selector.

Use two thresholds to avoid mode chattering:

\[
\epsilon_{in} < \epsilon_{out}
\]

Example:

```text
Drive if d_ws > 0.20 m
Reach if d_ws < 0.10 m and vehicle speed is small
```

Pseudo-code:

```python
if mode == "Drive":
    if d_ws < eps_in and norm(base_velocity) < v_switch:
        mode = "Reach"

elif mode == "Reach":
    if d_ws > eps_out:
        mode = "Drive"
```

Actions sent to the full system:

```python
if mode == "Drive":
    base_action = pi_drive(obs_drive)
    arm_action = hold_safe_folded_pose()

if mode == "Reach":
    base_action = [0.0, 0.0, 1.0]  # steering, throttle, brake
    arm_action = pi_reach(obs_reach)
```

Important:

```text
Do not switch to Reach while the base is still moving fast.
```

If the base is still drifting, the reach policy will have difficulty because it was trained assuming the base is fixed.

---

## 12. Full learned environment step

The hybrid learned environment step is:

```python
def step(action_or_policy_output):
    mode = mode_selector(state, goal)

    if mode == "Drive":
        a_base = pi_drive(obs_base)
        a_arm = hold_arm_action()

        base_next = f_base(base_state, a_base)
        arm_next = arm_state

    elif mode == "Reach":
        a_base = brake_action()
        raw_a_arm = pi_reach(obs_arm)

        safe_a_arm, safety_info = safety_filter(arm_state, raw_a_arm)

        if safety_info.is_safe:
            arm_next = f_arm(arm_state, safe_a_arm)
            collision_penalty = 0.0
        else:
            arm_next = arm_state
            collision_penalty = -w_collision

        base_next = base_state

    state_next = combine(base_next, arm_next)
    p_ee_next = FK(state_next)
    reward = compute_reward(state_next, goal, mode, collision_penalty)
    done = check_done(state_next, goal)

    return state_next, reward, done, info
```

---

## 13. Final Chrono validation

After training the policies in the learned models, the final controller must be evaluated in Chrono.

Validation cases:

| Case | Description | Expected behavior |
|---|---|---|
| Near goal | target already inside arm workspace | no base motion, arm reaches |
| Far goal | target outside workspace | drive first, then reach |
| Boundary goal | target near workspace boundary | minimal base adjustment, then reach |
| Unsafe arm pose | target requires awkward arm motion | policy avoids self-collision |
| Slight terrain variation | base dynamics changes slightly | drive policy remains stable |

Metrics:

| Metric | Meaning |
|---|---|
| success rate | percentage of reached goals |
| final EE error | final \(\|p_{ee}-p_{goal}\|\) |
| base distance for near goals | should be close to zero |
| workspace entry time for far goals | how fast drive mode makes goal reachable |
| number of unsafe arm actions | should decrease during training |
| Chrono collision count | final safety metric |
| learned-model rollout error | model accuracy |
| Chrono transfer success | whether policy trained in model works in Chrono |

---

## 14. Recommended implementation order

### Phase 1: Arm-only data and safety

```text
1. Define arm joint limits.
2. Define simplified arm collision geometry.
3. Implement custom safety filter.
4. Validate custom filter against Chrono collision detection.
5. Collect safe arm-only Chrono trajectories.
6. Train arm dynamics model.
7. Test multi-step arm rollout accuracy.
```

### Phase 2: Reach policy

```text
1. Build learned arm-only RL environment.
2. Add safety filter to every RL step.
3. Train π_reach.
4. Validate π_reach in Chrono with fixed base.
```

### Phase 3: Base model and drive policy

```text
1. Collect vehicle-only Chrono trajectories with arm folded.
2. Train base dynamics model using HMMWV-style pipeline.
3. Train π_drive to reduce workspace distance d_ws.
4. Validate π_drive in Chrono with arm folded.
```

### Phase 4: Hybrid controller

```text
1. Combine π_drive and π_reach using rule-based mode selector.
2. Test near-goal cases.
3. Test far-goal cases.
4. Debug switching thresholds.
5. Add failed Chrono rollouts to the relevant dataset.
6. Retrain models if needed.
```

---

## 15. Key design choices

Recommended first-version choices:

| Design item | Recommendation |
|---|---|
| Full coupled vehicle-arm dynamics | Do not use initially |
| Number of policies | Two separate low-level policies |
| Mode switch | Rule-based first |
| Arm dynamics data | Safety-filtered random free-space motion |
| Self-collision in dynamics model | Do not model it |
| Self-collision in RL | External safety filter + penalty |
| Chrono collision detection | Use for data collection and validation |
| RL collision checking | Use custom lightweight geometry |
| Arm action | \(\Delta q^{cmd}\), not direct torque |
| Arm model output | \(\Delta q, \Delta \dot q\) |
| End-effector loss | Optional auxiliary loss |

---

## 16. Most important conclusions

1. The arm dynamics model should learn only safe free-space dynamics.
2. Self-collision should be prevented by a safety filter, not learned by the model.
3. Chrono collision detection should be used as the ground-truth safety check during data collection and final validation.
4. A custom lightweight collision filter should be used during learned-model RL training to preserve rollout speed.
5. Two separate policies are easier and cleaner than one large masked policy.
6. The first hybrid controller should use rule-based switching between Drive and Reach.
7. The coupled vehicle-plus-arm dynamics case can be added later as an advanced extension, but it is not necessary for the first locomanipulation case study.
