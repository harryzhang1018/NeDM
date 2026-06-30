# Arm EE-Reaching RL — Implementation Plan (Phase 4)

> Companion to [arm-dyn-model.md](arm-dyn-model.md) (overall two-mode design) and the
> "Arm Mobile-Manipulator Study" section of [progress.md](progress.md) (status log).
> This is the implementation plan for the **reach-mode RL policy** on the trained `f_arm`.

## Context

Phases 0–3 are **done**: the arm dynamics model `f_arm` is trained
(`artifacts/training_runs/arm_transformer_full_v1`, 6L/256, ctx16, val_loss 0.00192, EE
open-loop drift ~1.9% errdist @0.5 s; overlay plots confirm tight multi-second tracking).
State==target 15-D `[q,qd,qcmd,ee_base]`, action = `Δqcmd` (4-D). Now train an
**end-effector reaching policy** `π_reach` that rolls out inside the frozen `f_arm` (base
fixed, matching the data), and that is **collision-free in Chrono** (link-vs-link and
link-vs-track) on deployment.

The core problem: `f_arm` was trained on **collision-free** transitions only, so it is only
valid in free space and knows nothing about contact. The policy must therefore be kept out
of collision both at train time (to keep the model in-distribution) and at deploy time (the
hard requirement). Confirmed design decisions:
- **Geometric safety filter** — a torch forward-kinematics + primitive-distance check whose
  geometry is extracted once from the real Chrono arm. Used as a training penalty/termination
  AND as a runtime shield in Chrono → hard, Chrono-verifiable guarantee.
- **FK-sampled safe goals** — goals are FK of random collision-free joint configs in the
  well-sampled upper/forward workspace; episodes seed their history from recorded episode
  prefixes (in-distribution rest starts).

## Architecture

```
recorded episode prefix ─seed→ ┌─────────── ArmReachingEnv (rsl_rl VecEnv) ───────────┐
FK-sampled goal ──────────────→│ obs → π_reach → Δqcmd → SafetyFilter(block if unsafe) │
                               │      → qcmd_next=clip(qcmd+Δq) → f_arm.predict_next   │
                               │      → roll state_hist; EE & clearance via torch FK   │→ PPO
                               └──────────────────────────────────────────────────────┘
deploy: same π_reach + same SafetyFilter shield, stepped in Chrono → verify 0 collisions
```

Reuse: `load_frozen_dynamics` (`src/nedm/rl/dynamics.py`); the `state_hist`/`action_hist`
roll + `predict_next_delta` substep, `reset_idx`, obs-assembly, `default_env_cfg`/
`merge_env_cfg`, and rsl_rl `VecEnv` contract from `src/nedm/rl/hmmwv_tracking_env.py`; the
PPO/`OnPolicyRunner` setup (`get_train_cfg`/`get_env_cfg`) from
`scripts/train_hmmwv_rl_tracking.py`; the Chrono scene + PD actuator + ground-truth collision
(`build_and_prepare`, `setup_arm_collision`, `ArmPdActuator`, `arm_contact`,
`gripper_center`, `ADJACENT_LINK_PAIRS`) from `src/nedm/arm_data.py`.

## Phase 4.1 — Arm forward-kinematics geometry (critical path)

The dynamics model outputs only EE; collision needs every link's pose, so we need batched FK.
The chain is a SolidWorks import (non-DH joint frames), so **extract the geometry from Chrono
once**, then reimplement FK in torch and validate.

- **`scripts/extract_arm_geometry.py`** (nedm env, uses Chrono): call
  `arm_data.build_and_prepare()` to get the settled M113+arm at home, then dump
  `artifacts/arm_geometry/arm_geometry_v1.json`:
  - the measured home config `q_home` (the settle sag) — FK uses `Δq = q − q_home` so the
    reference need not be exactly zero.
  - per joint i (`actuator.motors[i]`): world axis (frame Z) + pivot at home, from
    `GetFrame2Abs()`; the serial-chain distal-link membership (j0⊃j1⊃j2⊃j3).
  - per collision link (`setup_arm_collision` output): home world REF pose + box
    `(center, half)` in REF. (wrist is locked to endeffector, fingers ride it — all distal to j3.)
  - grasp point (`gripper_center`) expressed in the endeffector REF frame, for `ee(q)`.
  - vehicle obstacle: world AABB box(es) over the M113 collision bodies (chassis + track
    shoes via `GetTrackAssembly(...).GetTrackShoe(i).GetShoeBody()`), conservative; ground
    plane `z=0`; base→world transform.
  - **self-validate**: drive random qcmd, and at each step compare torch `fk(measured q)`
    link/EE world positions to Chrono's actual `GetFrameRefToAbs()`/`gripper_center`
    (and to the model's `ee_base` channel). Require max error ≲1 cm.
- **`src/nedm/rl/arm_kinematics.py`**: `ArmKinematics` (torch) loads the JSON; `fk(q:(B,4))`
  is product-of-exponentials — nested Rodrigues rotations about each home joint axis/pivot
  (using `Δq = q − q_home`) left-multiplying the home link poses; `ee(q)` = grasp-center;
  `link_points(q)` = box corners+center per link (B,L,P,3). All batched on GPU.

## Phase 4.2 — Safety filter

- **`src/nedm/rl/arm_safety.py`**: `ArmSafetyFilter(kin, margins)`:
  - `clearance(q:(B,4)) -> (B,)`: min over {ground (point z − ground_z), vehicle (point→OBB
    distance), self (non-adjacent link point-pairs, reuse `ADJACENT_LINK_PAIRS`)}.
  - `filter(q, qcmd, raw_dq) -> (safe_dq, unsafe:(B,), clearance)`: check the interpolated
    qcmd path α∈{0,.25,.5,.75,1} (doc §7.4); if any config breaches a margin, **block**
    (`safe_dq=0`) and flag unsafe.
  - Validate against Chrono labels: the terminal `collision=1` configs in the dataset must
    register as unsafe; well inside the workspace must be safe. (Self-collision is
    essentially unreachable within the ±π/2 pitch limits — joint-limit clipping covers it —
    but the pairwise check is kept as a cheap guard.)

## Phase 4.3 — Reaching env

- **`src/nedm/rl/arm_reaching_env.py`**: `ArmReachingEnv(VecEnv)` + `default_env_cfg`/
  `merge_env_cfg`. `num_actions=4`, `action_repeat=1` (50 Hz Δqcmd).
  - `reset_idx`: seed `state_hist`/`action_hist` from a recorded episode prefix (first
    `context` steps; build a seed cache from the processed dataset, à la references but
    prefix-only); sample a goal via FK of a random collision-free joint config in the
    upper/forward workspace; set `qcmd` from the seed tail.
  - `_nn_substep`: scale action → `raw_dq = tanh(a)·DQ_MAX`; `safe_dq = filter(...)`;
    `qcmd_next = clip(qcmd+safe_dq)`; `action_hist[-1]=safe_dq`; `predict_next_delta`;
    `next_state = state+delta` but **overwrite qcmd channels with `qcmd_next`**; roll.
    EE and clearance computed from torch `fk(next q)` (consistent with the filter).
  - obs (doc §9.1): `[q, qd, qcmd, goal_base, ee_base, goal−ee, clearance, last_action]`
    (normalized).
  - reward (doc §9.3): `−w_ee‖ee−goal‖ − w_a‖a‖² − w_Δa‖Δa‖² − w_near·max(0,margin−clr)²
    − w_col·unsafe + r_success`; success when `‖ee−goal‖<ε` for K steps.
  - termination: success / actual collision of predicted q (clearance<0) / timeout / non-finite.
  - Smoke test with a random policy: filter blocks unsafe Δq, EE & reward finite, episodes
    reset cleanly.

## Phase 4.4 — PPO training

- **`scripts/train_arm_rl_reaching.py`** + **`configs/arm_reach_rl_v1.json`**: mirror
  `train_hmmwv_rl_tracking.py` (`OnPolicyRunner`, `get_train_cfg` PPO, `get_env_cfg`), but
  drop the reference machinery (goals are sampled in-env). Smaller actor/critic
  (`[256,128,64]`), `--dynamics-checkpoint artifacts/training_runs/arm_transformer_full_v1`.
  Run under `artifacts/rl_runs/`. Watch: success rate, final EE error, unsafe-action rate
  (should fall toward 0), episode length.

## Phase 4.5 — Chrono validation (the guarantee)

- **`src/nedm/rl/arm_reaching_chrono_env.py`** + **`scripts/eval_arm_reaching_chrono.py`**:
  reuse the `arm_data` Chrono scene + PD actuator; step the trained `π_reach` with the
  **same `ArmSafetyFilter` shield**; obs from Chrono state (`actuator.read_state`,
  `gripper_center`, FK clearance); ground-truth collision via `arm_contact`. Report per-goal
  success (`‖ee−goal‖<ε`) and **Chrono collision count (must be 0)**. Keep runs modest (cf.
  memory `chrono-eval-multiref-stack-smash`: re-create the sim sparingly / few goals per process).

## Files
- **Add:** `scripts/extract_arm_geometry.py`, `src/nedm/rl/arm_kinematics.py`,
  `src/nedm/rl/arm_safety.py`, `src/nedm/rl/arm_reaching_env.py`,
  `scripts/train_arm_rl_reaching.py`, `configs/arm_reach_rl_v1.json`,
  `src/nedm/rl/arm_reaching_chrono_env.py`, `scripts/eval_arm_reaching_chrono.py`.
- **Reuse unchanged:** `rl/dynamics.py`, `rl/hmmwv_tracking_env.py` (pattern),
  `arm_data.py`, the trained `f_arm` checkpoint.
- **Out of scope (later):** drive policy + base dynamics model, the rule-based mode selector
  (doc §10–§12).

## Verification
1. FK: `extract_arm_geometry.py` self-validation max link/EE error ≲1 cm vs Chrono, and FK-EE
   ≈ model `ee_base`.
2. Filter: flags dataset terminal-collision configs as unsafe, free configs as safe.
3. Env: random-policy smoke test — finite EE/reward, shield blocks unsafe Δq, clean resets;
   short PPO run (few iters) increases reward.
4. Training: success rate ↑, unsafe-action rate ↓→~0, final EE error < ε over a held-out goal set.
5. Chrono: deploy π_reach+shield → success rate reported and **0 Chrono collisions** across a
   goal battery (near / far / boundary / lower-workspace goals).
