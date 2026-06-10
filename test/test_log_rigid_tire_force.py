"""Validation test: extended tire-force channels on rigid terrain (TMEASY).

Runs one HMMWV episode (launch, then steady cruise, then a steering step) and
logs rows through the production ``capture_row`` from ``nedm.hmmwv_data`` with
``include_tires=True``. Checks:

1. every logged channel is finite
2. straight-cruise window: summed wheel-frame Fz ~ vehicle weight
3. the derived wheel-frame forces (x = heading, z = world up) match the native
   TMEASY contact-frame forces from ReportTireForceLocal on flat terrain
4. the derived slip ratio tracks TMEASY's kinematic longitudinal slip during
   steady cruise

Check 3 is the load-bearing one for the cross-terrain schema: it shows the
spindle-state-derived frame (the only one available on SCM/CRM) reproduces the
tire-model frame in the regime where both exist.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pychrono as chrono
import pychrono.vehicle as veh

from nedm.hmmwv_data import (
    WHEEL_SPECS,
    capture_row,
    configure_chrono_data_paths,
    create_hmmwv,
    create_rigid_terrain,
    tire_field_names,
)

CONFIG = {
    "chrono_data_root": "chrono/data",
    "vehicle_data_root": "chrono/data/vehicle",
    "simulation": {"step_size_s": 0.002, "tire_step_size_s": 0.001},
    "vehicle": {
        "model": "HMMWV_Full",
        "contact_method": "SMC",
        "chassis_fixed": False,
        "init": {"x_m": 0.0, "y_m": 0.0, "z_m": 1.6},
        "engine_model": "SHAFTS",
        "transmission_model": "AUTOMATIC_SHAFTS",
        "drive_type": "AWD",
        "steering_type": "PITMAN_ARM",
        "tire_model": "TMEASY",
    },
    "terrain": {
        "type": "rigid",
        "length_m": 600.0,
        "width_m": 600.0,
        "friction": 0.9,
        "restitution": 0.01,
        "young_modulus_pa": 2e7,
    },
}

STEP_SIZE = 0.002
T_END = 8.0
RECORD_EVERY = 5  # 100 Hz
THROTTLE_FROM = 0.5
STEER_FROM = 6.0


def driver_inputs_at(time_s: float):
    inputs = veh.DriverInputs()
    inputs.m_throttle = 0.3 if time_s >= THROTTLE_FROM else 0.0
    inputs.m_steering = 0.25 if time_s >= STEER_FROM else 0.0
    inputs.m_braking = 0.0
    return inputs


def main() -> int:
    configure_chrono_data_paths(REPO_ROOT, CONFIG)
    hmmwv = create_hmmwv(CONFIG)
    terrain = create_rigid_terrain(hmmwv.GetSystem(), CONFIG)
    vehicle = hmmwv.GetVehicle()

    tire_radii = {
        name: float(vehicle.GetTire(axle, side).GetRadius())
        for name, axle, side in WHEEL_SPECS
    }
    mass = vehicle.GetMass()
    weight = mass * 9.81
    print(f"vehicle mass {mass:.1f} kg, weight {weight:.0f} N, radii {tire_radii}")

    rows = []
    local_force_log = []  # native TMEASY contact-frame forces, for check 3
    time_s = 0.0
    step_count = 0
    while time_s < T_END:
        inputs = driver_inputs_at(time_s)
        terrain.Synchronize(time_s)
        hmmwv.Synchronize(time_s, inputs, terrain)
        step_count += 1

        # capture between Synchronize and Advance: tire reports and spindle
        # state then refer to the same instant
        if step_count % RECORD_EVERY == 0:
            rows.append(
                capture_row(
                    hmmwv=hmmwv,
                    terrain=terrain,
                    scenario_name="tire_force_check",
                    scenario_family="test",
                    episode_id="tire_force_check",
                    split="train",
                    sample_index=len(rows),
                    time_s=time_s,
                    driver_inputs=inputs,
                    include_tires=True,
                    tire_radii=tire_radii,
                )
            )
            native = {"time_s": time_s}
            for name, axle, side in WHEEL_SPECS:
                frame = chrono.ChCoordsysd()
                tf_local = vehicle.GetTire(axle, side).ReportTireForceLocal(terrain, frame)
                native[name] = (tf_local.force.x, tf_local.force.y, tf_local.force.z)
            local_force_log.append(native)

        terrain.Advance(STEP_SIZE)
        hmmwv.Advance(STEP_SIZE)
        time_s = float(hmmwv.GetSystem().GetChTime())

    print(f"captured {len(rows)} rows; end speed {rows[-1]['speed_mps']:.2f} m/s")

    failures = []
    wheel_names = [name for name, *_ in WHEEL_SPECS]

    # 1. all channels finite, all tire fields present
    expected_tire_fields = set(tire_field_names())
    missing = expected_tire_fields - set(rows[0].keys())
    if missing:
        failures.append(f"missing tire fields: {sorted(missing)}")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                failures.append(f"non-finite {key} at t={row['time_s']:.2f}")
                break

    def mean(values):
        return sum(values) / len(values)

    # 2. straight steady cruise: load on tires ~ weight
    cruise = [r for r in rows if 4.0 <= r["time_s"] <= 5.5]
    total_fz = mean(
        [sum(r[f"{n}_force_wheel_fz_n"] for n in wheel_names) for r in cruise]
    )
    print(f"cruise window: sum Fz = {total_fz:.0f} N vs weight {weight:.0f} N "
          f"(ratio {total_fz / weight:.3f})")
    if not 0.9 <= total_fz / weight <= 1.1:
        failures.append(f"cruise Fz/weight ratio {total_fz / weight:.3f} outside [0.9, 1.1]")

    # 3. derived wheel-frame forces vs native TMEASY contact-frame forces,
    # including the steering window where the wheel frame rotates
    max_err = 0.0
    for row, native in zip(rows, local_force_log):
        if row["time_s"] < 1.0:
            continue  # skip drop-and-settle transient
        for n in wheel_names:
            derived = (
                row[f"{n}_force_wheel_fx_n"],
                row[f"{n}_force_wheel_fy_n"],
                row[f"{n}_force_wheel_fz_n"],
            )
            err = max(abs(d - nat) for d, nat in zip(derived, native[n]))
            max_err = max(max_err, err)
    print(f"max |derived - native local| force error: {max_err:.1f} N")
    if max_err > 0.05 * weight:  # 5% of weight ~ 1.2 kN headroom across saturation events
        failures.append(f"derived wheel-frame force deviates from native by {max_err:.0f} N")

    # 4. derived slip ratio (nominal radius) vs TMEASY kinematic slip (live
    # R_eff = R0 - deflection/3): the gap must equal the radius-convention
    # difference omega*(deflection/3)/|vx|, i.e. both track the same omega/vx.
    raw_gap = 0.0
    residual = 0.0
    for r in cruise:
        for n in wheel_names:
            omega = r[f"{n}_spindle_omega_radps"]
            vx = r[f"{n}_wheel_vx_mps"]
            expected_gap = omega * (r[f"{n}_deflection_m"] / 3.0) / max(abs(vx), 0.1)
            gap = r[f"{n}_slip_ratio"] - r[f"{n}_longitudinal_slip"]
            raw_gap = max(raw_gap, abs(gap))
            residual = max(residual, abs(gap - expected_gap))
    sample = cruise[len(cruise) // 2]
    print("cruise slip ratios:",
          {n: round(sample[f"{n}_slip_ratio"], 4) for n in wheel_names})
    print(f"max |derived - kinematic| slip gap: {raw_gap:.4f}; "
          f"after radius-convention correction: {residual:.4f}")
    if residual > 0.03:
        failures.append(
            f"slip gap unexplained by radius convention: residual {residual:.3f}"
        )

    print("\n" + ("FAIL:\n  " + "\n  ".join(failures) if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
