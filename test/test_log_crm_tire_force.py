"""Smoke test: per-wheel tire force logging on CRM terrain.

Barebone version of chrono/src/demos/python/vehicle/demo_VEH_CRMTerrain_WheeledVehicle.py
(Polaris RZR on CRM/SPH deformable terrain), stripped of visualization and the
path-follower driver. Runs 2 s of simulation (0.5 s free-roll settling, then a
gentle throttle ramp) and logs, per wheel:

- world-frame FSI force/torque on the spindle (CRMTerrain.GetFsiBodyForce/Torque)
- derived wheel-frame forces (x = heading, z = world up) per the unified schema
- spindle omega and a derived slip ratio (nominal tire radius convention)

Pass criteria printed at the end:
1. no NaN/inf in any logged channel
2. all four wheels carry positive vertical load after settling
3. summed vertical load ~ vehicle weight during the settled braking window
4. positive net longitudinal force and forward motion once throttle is applied
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pychrono.core as chrono
import pychrono.fsi as fsi
import pychrono.vehicle as veh

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "tests"
OUT_CSV = OUT_DIR / "crm_tire_force_log.csv"

# Simulation settings (mirroring the demo, shrunk for a short smoke test)
STEP_SIZE = 5e-4
T_END = 2.0
RECORD_EVERY = 20  # 100 Hz logging
SETTLE_UNTIL = 0.5  # free-roll settling: locked brakes while landing dig in the nose
THROTTLE_MAX = 0.35  # gentle ramp; hard throttle trenches the wheels into soft soil
SPACING = 0.04
ACTIVE_BOX_DIM = 0.8
TERRAIN_LENGTH = 12.0
TERRAIN_WIDTH = 3.0
TERRAIN_DEPTH = 0.25
VEHICLE_INIT_X = 4.0
VEHICLE_INIT_HEIGHT = 0.25

# CRM soil material (same as demo)
DENSITY = 1700
COHESION = 5e3
FRICTION = 0.8
YOUNGS_MODULUS = 1e6
POISSON_RATIO = 0.3

WORLD_UP = chrono.ChVector3d(0, 0, 1)
GRAVITY = 9.81


def wheel_frame_axes(spindle):
    """Wheel-aligned frame: x = heading, z = world up (the cross-terrain convention)."""
    spin_axis = spindle.GetRot().GetAxisY()
    x_axis = spin_axis.Cross(WORLD_UP)
    x_axis = x_axis.GetNormalized()
    y_axis = WORLD_UP.Cross(x_axis)
    return x_axis, y_axis


def create_vehicle():
    vehicle = veh.WheeledVehicle(
        veh.GetVehicleDataFile("Polaris/Polaris.json"), chrono.ChContactMethod_SMC
    )
    vehicle.Initialize(
        chrono.ChCoordsysd(chrono.ChVector3d(VEHICLE_INIT_X, 0, VEHICLE_INIT_HEIGHT), chrono.QUNIT)
    )
    vehicle.GetChassis().SetFixed(False)
    vehicle.SetChassisVisualizationType(chrono.VisualizationType_NONE)
    vehicle.SetSuspensionVisualizationType(chrono.VisualizationType_NONE)
    vehicle.SetSteeringVisualizationType(chrono.VisualizationType_NONE)
    vehicle.SetWheelVisualizationType(chrono.VisualizationType_NONE)
    vehicle.SetTireVisualizationType(chrono.VisualizationType_NONE)

    engine = veh.ReadEngineJSON(veh.GetVehicleDataFile("Polaris/Polaris_EngineSimpleMap.json"))
    transmission = veh.ReadTransmissionJSON(
        veh.GetVehicleDataFile("Polaris/Polaris_AutomaticTransmissionSimpleMap.json")
    )
    vehicle.InitializePowertrain(veh.ChPowertrainAssembly(engine, transmission))

    for axle in vehicle.GetAxles():
        for wheel in axle.GetWheels():
            tire = veh.ReadTireJSON(veh.GetVehicleDataFile("Polaris/Polaris_RigidTire.json"))
            vehicle.InitializeTire(tire, wheel, chrono.VisualizationType_NONE)
    return vehicle


def create_terrain(vehicle):
    sys_mbs = vehicle.GetSystem()
    sys_mbs.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    sys_mbs.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    sys_mbs.SetNumThreads(4, 1, 1)
    sys_mbs.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    terrain = veh.CRMTerrain(sys_mbs, SPACING)
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    terrain.SetStepSizeCFD(STEP_SIZE)
    terrain.RegisterVehicle(vehicle)

    mat_props = fsi.ElasticMaterialProperties()
    mat_props.density = DENSITY
    mat_props.Young_modulus = YOUNGS_MODULUS
    mat_props.Poisson_ratio = POISSON_RATIO
    mat_props.mu_I0 = 0.04
    mat_props.mu_fric_s = FRICTION
    mat_props.mu_fric_2 = FRICTION
    mat_props.average_diam = 0.005
    mat_props.cohesion_coeff = COHESION
    terrain.SetElasticSPH(mat_props)

    sph_params = fsi.SPHParameters()
    sph_params.integration_scheme = fsi.IntegrationScheme_RK2
    sph_params.initial_spacing = SPACING
    sph_params.d0_multiplier = 1.2
    sph_params.free_surface_threshold = 0.8
    sph_params.artificial_viscosity = 0.5
    sph_params.shifting_method = fsi.ShiftingMethod_PPST
    sph_params.shifting_ppst_push = 3.0
    sph_params.shifting_ppst_pull = 1.0
    sph_params.use_consistent_gradient_discretization = False
    sph_params.use_consistent_laplacian_discretization = False
    sph_params.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph_params.boundary_method = fsi.BoundaryMethod_ADAMI
    terrain.SetSPHParameters(sph_params)

    # Register the wheel spindles as FSI rigid bodies (rigid tire collision mesh)
    mesh_filename = veh.GetVehicleDataFile("Polaris/meshes/Polaris_tire_collision.obj")
    geometry = chrono.ChBodyGeometry()
    geometry.coll_meshes.append(
        chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh_filename, chrono.VNULL)
    )
    for axle in vehicle.GetAxles():
        for wheel in axle.GetWheels():
            terrain.AddRigidBody(wheel.GetSpindle(), geometry, False)

    terrain.SetActiveDomain(chrono.ChVector3d(ACTIVE_BOX_DIM))
    terrain.SetActiveDomainDelay(0)

    terrain.Construct(
        chrono.ChVector3d(TERRAIN_LENGTH, TERRAIN_WIDTH, TERRAIN_DEPTH),
        chrono.ChVector3d(TERRAIN_LENGTH / 2, 0, 0),
        fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,
    )
    terrain.Initialize()
    return terrain


def main() -> int:
    chrono.SetChronoDataPath(str(REPO_ROOT / "chrono" / "data") + "/")
    veh.SetVehicleDataPath(str(REPO_ROOT / "chrono" / "data" / "vehicle") + "/")

    print("Creating vehicle...")
    vehicle = create_vehicle()
    print("Creating CRM terrain...")
    terrain = create_terrain(vehicle)
    aabb = terrain.GetSPHBoundingBox()
    print(f"  SPH particles: {terrain.GetNumSPHParticles()}")
    print(f"  SPH AABB z: [{aabb.min.z:.3f}, {aabb.max.z:.3f}]")
    for axle_index, axle in enumerate(vehicle.GetAxles()):
        for wheel in axle.GetWheels():
            pos = wheel.GetSpindle().GetPos()
            print(f"  spindle a{axle_index} at x={pos.x:.2f} z={pos.z:.3f} "
                  f"(bottom z={pos.z - wheel.GetTire().GetRadius():.3f})")

    wheels = []
    for axle_index, axle in enumerate(vehicle.GetAxles()):
        for wheel in axle.GetWheels():
            spindle = wheel.GetSpindle()
            side = "L" if spindle.GetPos().y > 0 else "R"
            wheels.append((f"a{axle_index}{side}", wheel, spindle, wheel.GetTire().GetRadius()))
    print("  wheels:", [(name, f"R={radius:.3f}") for name, wheel, spindle, radius in wheels])

    mass = vehicle.GetMass()
    weight = mass * GRAVITY
    print(f"  vehicle mass: {mass:.1f} kg (weight {weight:.0f} N)")

    fieldnames = ["time_s", "speed_mps", "throttle", "braking", "pos_x_m", "pos_z_m", "pitch_rad"]
    for name, *_ in wheels:
        fieldnames += [
            f"{name}_force_world_x_n", f"{name}_force_world_y_n", f"{name}_force_world_z_n",
            f"{name}_torque_world_x_nm", f"{name}_torque_world_y_nm", f"{name}_torque_world_z_nm",
            f"{name}_force_wheel_fx_n", f"{name}_force_wheel_fy_n", f"{name}_force_wheel_fz_n",
            f"{name}_spindle_omega_radps", f"{name}_slip_ratio",
        ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    time = 0.0
    step_count = 0
    print(f"Running {T_END} s of simulation...")
    while time < T_END:
        inputs = veh.DriverInputs()
        if time < SETTLE_UNTIL:
            inputs.m_throttle = 0.0
        else:
            inputs.m_throttle = min(THROTTLE_MAX, (time - SETTLE_UNTIL) / 0.5 * THROTTLE_MAX)
        inputs.m_braking = 0.0
        inputs.m_steering = 0.0

        terrain.Synchronize(time)
        vehicle.Synchronize(time, inputs, terrain)
        # terrain.Advance steps the coupled FSI problem (CFD + the vehicle's MBS)
        terrain.Advance(STEP_SIZE)

        if step_count % RECORD_EVERY == 0:
            chassis_pos = vehicle.GetPos()
            rot = vehicle.GetRot()
            row = {
                "time_s": round(time, 6),
                "speed_mps": vehicle.GetSpeed(),
                "throttle": inputs.m_throttle,
                "braking": inputs.m_braking,
                "pos_x_m": chassis_pos.x,
                "pos_z_m": chassis_pos.z,
                "pitch_rad": rot.GetCardanAnglesZYX().y,
            }
            for name, wheel, spindle, radius in wheels:
                force = terrain.GetFsiBodyForce(spindle)
                torque = terrain.GetFsiBodyTorque(spindle)
                x_axis, y_axis = wheel_frame_axes(spindle)
                omega = spindle.GetAngVelParent().Dot(spindle.GetRot().GetAxisY())
                v_wheel = spindle.GetPosDt()
                vx = v_wheel.Dot(x_axis)
                slip = (omega * radius - vx) / max(abs(vx), 0.1)
                row[f"{name}_force_world_x_n"] = force.x
                row[f"{name}_force_world_y_n"] = force.y
                row[f"{name}_force_world_z_n"] = force.z
                row[f"{name}_torque_world_x_nm"] = torque.x
                row[f"{name}_torque_world_y_nm"] = torque.y
                row[f"{name}_torque_world_z_nm"] = torque.z
                row[f"{name}_force_wheel_fx_n"] = force.Dot(x_axis)
                row[f"{name}_force_wheel_fy_n"] = force.Dot(y_axis)
                row[f"{name}_force_wheel_fz_n"] = force.Dot(WORLD_UP)
                row[f"{name}_spindle_omega_radps"] = omega
                row[f"{name}_slip_ratio"] = slip
            rows.append(row)

        time += STEP_SIZE
        step_count += 1
        if step_count % 400 == 0:
            print(f"  t={time:.2f}s speed={vehicle.GetSpeed():.2f} m/s")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")

    # ---- sanity checks ----
    failures = []

    all_values = [v for row in rows for v in row.values()]
    if any(isinstance(v, float) and not math.isfinite(v) for v in all_values):
        failures.append("found NaN/inf in logged channels")

    def mean(values):
        return sum(values) / len(values)

    # settled window: vehicle resting in soil, before throttle bites
    settled = [r for r in rows if 0.35 <= r["time_s"] <= SETTLE_UNTIL]
    wheel_names = [name for name, *_ in wheels]
    fz_means = {n: mean([r[f"{n}_force_wheel_fz_n"] for r in settled]) for n in wheel_names}
    total_fz = sum(fz_means.values())
    print(f"\nsettled window (0.35-{SETTLE_UNTIL} s):")
    for n in wheel_names:
        print(f"  {n}: mean Fz = {fz_means[n]:8.1f} N")
    print(f"  sum Fz = {total_fz:.0f} N vs weight = {weight:.0f} N "
          f"(ratio {total_fz / weight:.2f})")
    for n in wheel_names:
        if fz_means[n] <= 0:
            failures.append(f"wheel {n} carries no vertical load in settled window")
    if not 0.6 <= total_fz / weight <= 1.4:
        failures.append(f"summed Fz/weight ratio {total_fz / weight:.2f} outside [0.6, 1.4]")

    # throttle window: expect net positive traction and forward motion
    powered = [r for r in rows if r["time_s"] >= 1.5]
    total_fx = mean([sum(r[f"{n}_force_wheel_fx_n"] for n in wheel_names) for r in powered])
    end_speed = rows[-1]["speed_mps"]
    slip_end = {n: mean([r[f"{n}_slip_ratio"] for r in powered]) for n in wheel_names}
    print(f"\npowered window (>= 1.5 s):")
    print(f"  mean total wheel-frame Fx = {total_fx:.1f} N")
    print(f"  end speed = {end_speed:.2f} m/s")
    print(f"  mean slip ratios = " + ", ".join(f"{n}:{slip_end[n]:.3f}" for n in wheel_names))
    if total_fx <= 0:
        failures.append(f"mean total Fx {total_fx:.1f} N is not propulsive under throttle")
    if end_speed < 0.05:
        failures.append(f"vehicle did not move under throttle (end speed {end_speed:.3f} m/s)")

    print("\n" + ("FAIL:\n  " + "\n  ".join(failures) if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
