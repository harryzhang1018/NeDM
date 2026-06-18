# HMMWV CRM Terrain Data Collection Pipeline Design

Date: 2026-06-18

## Goal

Collect HMMWV dynamics episodes on Chrono `CRMTerrain` deformable soil while keeping the raw dataset layout compatible with the existing rigid-terrain pipeline:

- one CSV per episode under `episodes/`
- one episode JSON sidecar per episode
- a shard-level `dataset_index.json`
- a resolved collector config
- the same 100 Hz logged state/action schema used by `scripts/build_hmmwv_training_dataset.py`
- a processed cache built with `--state-field-preset tire_force_omega`

The first production target should be a fixed-soil CRM dataset, not a broad soil benchmark. Terrain and soil variation can come after a fixed configuration passes rollout and validation checks.

## Production Launch Scripts

The implemented production collector is resumable and uses the same raw CSV,
episode sidecar, `dataset_index.json`, and `tire_force_omega` processed-cache
format as the rigid-terrain HMMWV datasets.

For a 2000-episode CRM dataset on a workstation:

```bash
cd /home/harry/NeDM
git pull origin main
bash scripts/launch_hmmwv_crm2000_collection.sh
```

Monitor progress:

```bash
tmux attach -t hmmwv_crm2000_collection
tail -f artifacts/datasets/hmmwv_crm_2000/logs/run.log
```

Default 2000-episode settings:

- raw output: `artifacts/datasets/hmmwv_crm_2000`
- processed output: `artifacts/training_datasets/hmmwv_crm_2000_force_omega_seq_v1`
- terrain: `150 m x 150 m x 0.25 m`
- CRM spacing: `0.08 m`
- boundary exit margin: `5 m`
- Chrono MBD threads: `12`
- episode duration range: `12 s` to `18 s`
- processed state preset: `tire_force_omega`

Useful overrides:

```bash
PYTHON_BIN=/path/to/nedm/bin/python CHRONO_THREADS=12 BUILD_PROCESSED=1 \
  bash scripts/launch_hmmwv_crm2000_collection.sh
```

Set `BUILD_PROCESSED=0` to collect raw episodes only. Re-running the launcher
or run script is safe by default because the collector uses episode sidecars as
completion markers and resumes completed episodes.

## Existing Constraints

The current collector in `src/nedm/hmmwv_data.py` only accepts `terrain.type` values `rigid` and `rigid_heightmap`, and every episode currently calls `create_rigid_terrain`. A CRM dataset therefore requires a new terrain backend, not just a new config file.

Chrono CRM has different semantics from the current TMEASY-on-rigid setup:

- `CRMTerrain` is an SPH-FSI problem builder, not just a contact terrain.
- `CRMTerrain.GetHeight`, `GetNormal`, and `GetCoefficientFriction` are placeholders, so terrain-query-based tire models are not the right source of CRM tire forces.
- Interacting wheels must be registered with the FSI terrain using `AddRigidBody(...)` or `AddFeaMesh(...)`.
- With `CRMTerrain.RegisterVehicle(...)`, the vehicle dynamics are advanced inside `terrain.Advance(step)`. The collector must not also call `hmmwv.Advance(step)`.

Useful Chrono references in this repo:

- `chrono/doxygen/documentation/manuals/vehicle/vehicle_terrain_crm.md`
- `chrono/doxygen/documentation/manuals/vehicle/vehicle_terrain_crm_performance.md`
- `chrono/src/demos/python/vehicle/demo_VEH_CRMTerrain_WheeledVehicle.py`
- `chrono/src/chrono_vehicle/terrain/CRMTerrain.h`
- `chrono/src/chrono_vehicle/terrain/CRMTerrain.cpp`
- `test/test_log_crm_tire_force.py`
- `test/test_log_rigid_tire_force.py`

## Vehicle And Tire Choice

Use `veh.HMMWV_Full` as the primary vehicle to preserve continuity with the existing datasets:

- contact method: `SMC`
- engine: `SHAFTS`
- transmission: `AUTOMATIC_SHAFTS`
- drive: `AWD`
- steering: `PITMAN_ARM`
- tire model: `RIGID_MESH` or `RIGID`

For CRM, the force source should be FSI force on the wheel spindle, not `tire.ReportTireForce(terrain)`.

Recommended first tire choice:

- tire type: `veh.TireModelType_RIGID_MESH`
- FSI collision mesh: `chrono/data/vehicle/hmmwv/hmmwv_tire_coarse_closed.obj`
- source JSON for dimensions/mass: `chrono/data/vehicle/hmmwv/tire/HMMWV_RigidMeshTire_CoarseClosed.json`

Fallback if `HMMWV_Full` does not expose enough Python-side tire control:

- use `veh.WheeledVehicle(veh.GetVehicleDataFile("hmmwv/vehicle/HMMWV_Vehicle.json"), chrono.ChContactMethod_SMC)`
- initialize `HMMWV_EngineShafts.json`, `HMMWV_AutomaticTransmissionShafts.json`, and `HMMWV_RigidMeshTire_CoarseClosed.json`
- keep a wrapper that exposes the same state channels as the current `HMMWV_Full` collector.

## Raw CSV Schema

The CRM collector should emit the same `BASE_FIELDS` and `tire_field_names()` columns as the rigid tire-force collector.

Fields used by the primary training preset must have the same semantics:

- `tire_*_force_wheel_fx_n`
- `tire_*_force_wheel_fy_n`
- `tire_*_force_wheel_fz_n`
- `tire_*_spindle_omega_radps`

For CRM, compute these from spindle state and FSI body force:

1. `force_world = terrain.GetFsiBodyForce(spindle)`
2. `torque_world = terrain.GetFsiBodyTorque(spindle)`
3. `spin_axis = spindle.GetRot().GetAxisY()`
4. `heading = spin_axis.Cross(world_up).GetNormalized()`
5. `lateral = world_up.Cross(heading)`
6. `force_wheel_fx = force_world.Dot(heading)`
7. `force_wheel_fy = force_world.Dot(lateral)`
8. `force_wheel_fz = force_world.Dot(world_up)`
9. `spindle_omega = spindle.GetAngVelParent().Dot(spin_axis)`
10. `wheel_vx = spindle.GetPosDt().Dot(heading)`
11. `slip_ratio = (spindle_omega * nominal_radius - wheel_vx) / max(abs(wheel_vx), 0.1)`

Fields that are native to analytical tire models should remain finite in CRM logs:

- `longitudinal_slip`: use the same derived slip ratio convention as `slip_ratio`, or log `0.0` until a validated derived definition is added.
- `slip_angle_rad`: derive from spindle velocity in the wheel frame when practical.
- `camber_angle_rad`: derive against world-up when practical.
- `deflection_m`: use `0.0` for rigid CRM tires.
- tire moments: use `terrain.GetFsiBodyTorque(spindle)`.

The processed training cache can ignore these auxiliary fields by selecting `tire_force_omega`.

## Collector Refactor

Add a small terrain-runtime abstraction instead of threading CRM conditionals through the whole episode loop.

Proposed shape:

```python
@dataclass
class TerrainRuntime:
    terrain: Any
    metadata: dict[str, Any]
    force_source: str
    advance_vehicle_separately: bool
```

Backends:

- `create_rigid_terrain_runtime(...)`
  - existing `RigidTerrain` and `rigid_heightmap`
  - `force_source = "tire_report"`
  - `advance_vehicle_separately = True`

- `create_crm_terrain_runtime(...)`
  - new `CRMTerrain`
  - `force_source = "crm_fsi"`
  - `advance_vehicle_separately = False` when `terrain.RegisterVehicle(...)` is used

Episode loop changes:

- keep `ChDataDriver` and generated driver profiles
- call `driver.Synchronize(time_s)`, `terrain.Synchronize(time_s)`, and `hmmwv.Synchronize(time_s, driver_inputs, terrain)`
- capture rows after synchronization and before advancing
- call `driver.Advance(step_size_s)`
- call `terrain.Advance(step_size_s)`
- call `hmmwv.Advance(step_size_s)` only if `advance_vehicle_separately` is true

Tire logging should move into a helper:

- `capture_tire_channels_from_report_tire_force(...)`
- `capture_tire_channels_from_crm_fsi(...)`

Both helpers should populate the same CSV column names.

## CRM Config Schema

Add `terrain.type = "crm"`:

```json
{
  "terrain": {
    "type": "crm",
    "mode": "box",
    "length_m": 30.0,
    "width_m": 10.0,
    "depth_m": 0.35,
    "initial_spacing_m": 0.05,
    "active_domain_m": [1.2, 1.2, 0.9],
    "active_domain_delay_s": 1.0,
    "soil": {
      "density": 1700.0,
      "cohesion": 5000.0,
      "friction": 0.8,
      "young_modulus_pa": 1000000.0,
      "poisson_ratio": 0.3,
      "mu_I0": 0.04,
      "average_diam_m": 0.005
    },
    "sph": {
      "integration_scheme": "RK2",
      "d0_multiplier": 1.2,
      "free_surface_threshold": 0.8,
      "artificial_viscosity": 0.5,
      "shifting_method": "PPST",
      "shifting_ppst_push": 3.0,
      "shifting_ppst_pull": 1.0,
      "viscosity_method": "ARTIFICIAL_BILATERAL",
      "boundary_method": "ADAMI",
      "num_proximity_search_steps": 1
    },
    "moving_patch": {
      "enabled": false,
      "buffer_distance_m": 8.0,
      "shift_distance_m": 4.0
    }
  },
  "logging": {
    "include_tire_channels": true,
    "tire_force_source": "crm_fsi"
  }
}
```

Use `mode = "box"` for the first turning dataset. `ConstructMovingPatch(...)` is valuable for mostly straight or shallow-turn episodes, but Chrono's CRM moving patch shifts in positive X and is not a general 2D moving terrain. The existing 300G turn distribution will need shorter durations and lower steering amplitudes for CRM unless a very large finite box is affordable.

Recommended initial simulation settings:

- `simulation.step_size_s = 0.0005`
- `simulation.record_step_s = 0.01`
- `simulation.driver_sample_step_s = 0.01`
- `scenario_generator.warmup_s = 1.0` to `2.5`
- `--jobs 1` for initial CRM smoke tests

Only raise `num_proximity_search_steps` above `1` after comparing force, sinkage, and trajectory metrics against the baseline.

## Proposed Files

Design target files:

- `test/test_log_hmmwv_crm_tire_force.py`
  - single HMMWV CRM smoke test
  - validates FSI force logging and vehicle motion

- `scripts/prepare_hmmwv_crm10g_generation.py`
  - writes shard configs matching the 300G family mix where feasible
  - uses shorter durations and CRM-safe speed/steering ranges

- `scripts/smoke_test_hmmwv_crm10g.sh`
  - writes a smoke config
  - collects 4 to 12 episodes with `--jobs 1`
  - validates schema and force sanity

- `scripts/cluster/collect_hmmwv_crm10g.sh`
  - Slurm array collection
  - one shard per task
  - conservative default `JOBS=1`

- `scripts/validate_hmmwv_crm_dataset.py`
  - validates raw shards with CRM-specific force/sinkage tolerances

Optional later:

- `scripts/prepare_hmmwv_crm300g_generation.py`
- `scripts/cluster/collect_hmmwv_crm300g.sh`

Do not add the 300G CRM scripts until the 10G CRM dataset has passed dynamics-training smoke tests. CRM cost and terrain finite-domain effects make direct 300G scaling risky.

## Shard Design

Start with a fixed-soil 10G-equivalent dataset:

- raw root: `artifacts/datasets/hmmwv_crm_fixedsoil_10g_shards`
- plan root: `artifacts/datasets/hmmwv_crm_fixedsoil_10g_plan`
- processed cache: `artifacts/training_datasets/hmmwv_crm_fixedsoil_10g_force_omega_seq_v1`

Use the same family labels as the rigid 300G tire-force dataset:

- `multi_steer`
- `sustained_turn`
- `sine_steer`
- `chirp_steer`
- `doublet_steer`
- `steer_brake`

Initial CRM-safe differences:

- shorter episodes, roughly 8 to 24 s after warmup
- lower top speed than the rigid 300G fast band
- smaller sustained-turn steering amplitudes
- termination if chassis approaches 80 to 90 percent of terrain half-width or half-length
- per-episode sidecar records `terminated_out_of_bounds`, `sinkage_limit_hit`, and `patch_moved_count`

Suggested first shard size should be empirical. Run one 12-episode smoke shard and use its raw bytes per second of simulated time to choose an episodes-per-shard value. Do not assume the rigid 300G ratio of 256 episodes to 2.4 GB carries over.

## Validation Gates

Gate 1: API smoke

- HMMWV creates and settles on CRM terrain.
- Four wheel spindles are registered as FSI rigid bodies.
- `GetFsiBodyForce` and `GetFsiBodyTorque` are finite for every wheel.
- settled-window total vertical force is within a broad band around vehicle weight, initially `[0.6, 1.4]`.
- throttle produces positive net wheel-frame `Fx` and measurable forward motion.

Gate 2: collector smoke shard

- `dataset_index.json` exists.
- every episode CSV has all base and tire columns.
- all logged values are finite.
- episode JSON records terrain type, soil parameters, SPH spacing, particle count, BCE count, and force source.
- generated rows can be read by `scripts/build_hmmwv_training_dataset.py`.

Gate 3: processed cache smoke

```bash
python scripts/build_hmmwv_training_dataset.py \
  --dataset-root artifacts/datasets/hmmwv_crm_fixedsoil_10g_shards/smoke \
  --output-dir artifacts/training_datasets/hmmwv_crm_fixedsoil_smoke_force_omega_seq_v1 \
  --state-field-preset tire_force_omega \
  --disk-backed-arrays
```

Expected metadata:

- `state_field_preset = "tire_force_omega"`
- `len(state_fields) = 23`
- actions are the standard 3 driver controls
- no NaN or inf in state/action/target arrays

Gate 4: model-training smoke

- one epoch or a few optimizer steps on the CRM processed cache
- no normalization pathologies from CRM force spikes
- compare force/channel percentiles against the rigid 300G force/omega cache

## Scale-Up Path

1. Implement and pass `test/test_log_hmmwv_crm_tire_force.py`.
2. Add `terrain.type = "crm"` to the collector with `force_source = "crm_fsi"`.
3. Collect a 4 to 12 episode smoke shard.
4. Build a smoke processed cache with `tire_force_omega`.
5. Collect a calibration shard across spacing and soil options:
   - spacing: `0.06`, `0.05`, `0.04`
   - proximity search steps: `1`, then `4`, then `10`
   - fixed soil first; soil variation later
6. Pick a production CRM setting based on runtime, particle count, force sanity, and open-loop rollout quality.
7. Collect fixed-soil 10G.
8. Fine-tune from the 300G rigid force/omega model only after validating normalization and force-scale differences.
9. Consider 300G CRM only if 10G CRM improves Chrono transfer enough to justify cost.

## Training Integration

The existing preprocessing path can be reused as long as the raw CRM shard has the expected CSV/index layout:

```bash
python scripts/build_hmmwv_training_dataset.py \
  --dataset-root artifacts/datasets/hmmwv_crm_fixedsoil_10g_shards/shard_* \
  --output-dir artifacts/training_datasets/hmmwv_crm_fixedsoil_10g_force_omega_seq_v1 \
  --state-field-preset tire_force_omega \
  --disk-backed-arrays
```

For model work, start with a CRM-only cache. A mixed flat+CRM cache should use explicit metadata documenting source domains and should be evaluated on both domains. If force magnitudes differ sharply from rigid terrain, use separate CRM normalization for the first training pass before trying combined normalization.

## Main Risks

- `HMMWV_Full` plus CRM rigid-mesh tires may expose fewer Python-side controls than the Polaris JSON demo. The fallback is a JSON-based HMMWV vehicle with explicit tire initialization.
- CRM force spikes can dominate normalization and target loss. Use percentile diagnostics before training long runs.
- A finite CRM box cannot support the same long turning episodes as the 900 m rigid terrain. Moving patch helps only for mostly forward trajectories.
- CRM cost can make row-count or raw-GB targets misleading. Scale by useful simulated seconds and transition counts, not only raw bytes.
- The CRM tire model is not the same physical setup as TMEASY on rigid terrain. The dataset is schema-compatible with rigid 300G, but not physics-identical.
