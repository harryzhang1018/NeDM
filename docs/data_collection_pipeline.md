# HMMWV Data Collection Pipeline

## Objective

The first milestone is not a general vehicle model. It is a controlled overfit experiment on a single Chrono vehicle configuration:

- `veh.HMMWV_Full`
- flat rigid terrain
- fixed friction `mu = 0.9`
- `TMEASY` tires
- `SMC` contact

That keeps the first dataset narrow enough that a small NN can learn it, while still exposing roll, pitch, body slip, yaw dynamics, and tire saturation effects that a kinematic bicycle model would miss.

## Why This Shape

The attached survey at [deep-research-report-vehicle.md](/home/harry/NeDM/deep-research-report-vehicle.md) points in two directions that matter immediately:

- learned vehicle models should be validated on multi-step rollouts, not only one-step error
- hybrid and physics-guided models tend to generalize better than pure black-box regressors

That means the collector should preserve episode boundaries, action histories, and physically meaningful channels. It should not only dump shuffled scalar rows.

## Pipeline Stages

1. Define a scenario manifest in [configs/hmmwv_overfit_v1.json](/home/harry/NeDM/configs/hmmwv_overfit_v1.json).
2. Sample driver commands into a `ChDataDriver` input table.
3. Run a headless PyChrono HMMWV simulation.
4. Discard the initial settling transient with a warmup window.
5. Log per-episode CSV files plus an episode index.
6. Split by episode, not by row.
7. Build one-step or multi-step training windows later from the episode files.

## Maneuver Set

The default manifest uses deliberate excitation maneuvers instead of a single path-following controller:

- launch and brake
- left step steer
- right step steer
- low-speed sinusoidal steering
- medium-speed sinusoidal steering
- steering chirp

This is a better first dataset than free driving because it covers the local state-action space around straight-line, transient cornering, and combined longitudinal/lateral response.

## Logged Signals

The collector records a compact chassis core plus optional tire channels.

Core chassis state:

- position and quaternion
- roll, pitch, yaw
- world-frame and body-frame linear velocity
- world-frame and body-frame linear acceleration
- world-frame and body-frame angular velocity
- speed, body slip, roll rate, yaw rate
- steering, throttle, braking commands

Optional tire channels:

- longitudinal slip
- slip angle
- camber angle
- tire force in world frame
- tire moment in world frame

The PyChrono SWIG wrapper exposed the tire channels cleanly. Suspension force reporting exists in Chrono, but the Python binding currently returns a raw SWIG object that is not convenient to serialize, so it is intentionally left out of the first pass.

## Warmup Rule

The HMMWV starts above the terrain and needs a short drop-and-settle period before the tire forces become meaningful. The collector therefore uses a per-scenario `warmup_s` window and only records after that time.

## Recommended First Training Target

For the first overfit experiment, start with a simple discrete-time state transition at 100 Hz:

- input state: `vx_body, vy_body, yaw_rate, roll, pitch`
- control: `steering, throttle, braking`
- target: next-step `vx_body, vy_body, yaw_rate, roll, pitch`

Then compare against two extensions:

- residual prediction on top of a simple bicycle model
- richer targets that add `ax_body, ay_body`

## Split Strategy

Do not split randomly at the row level. Keep whole episodes together.

- `train`: most episodes
- `val`: a deterministic episode-level holdout

For this first narrow dataset, holding out entire maneuvers is optional. Once the collector is stable, the next real test should hold out scenario families or friction values.

## Known Limits In This First Version

- single vehicle only
- rigid flat terrain only
- fixed friction only
- no sensor noise or observation model
- no NN training code yet

That is deliberate. The immediate goal is a reliable data source, not a broad benchmark.
