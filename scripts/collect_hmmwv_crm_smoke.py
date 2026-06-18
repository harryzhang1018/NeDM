from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pychrono as chrono
import pychrono.fsi as fsi
import pychrono.vehicle as veh

from nedm.hmmwv_data import (
    BASE_FIELDS,
    WHEEL_SPECS,
    assign_split,
    build_driver_entries,
    build_time_grid,
    configure_chrono_data_paths,
    create_hmmwv,
    csv_field_names,
)


WORLD_UP = chrono.ChVector3d(0, 0, 1)
GRAVITY = 9.81
DEFAULT_OUTPUT_DIR = Path("artifacts/datasets/hmmwv_crm_smoke")


@dataclass(frozen=True)
class WheelRuntime:
    name: str
    axle_index: int
    side: Any
    spindle: Any
    radius_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one small HMMWV-on-CRM smoke episode while writing the same "
            "episode CSV/index schema used by the rigid terrain datasets."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--warmup-s", type=float, default=0.2)
    parser.add_argument("--step-size-s", type=float, default=5e-4)
    parser.add_argument("--record-step-s", type=float, default=0.01)
    parser.add_argument(
        "--chrono-threads",
        type=int,
        default=12,
        help="Number of Chrono MBD threads to use for CRM runs.",
    )
    parser.add_argument(
        "--crm-spacing-m",
        type=float,
        default=0.08,
        help="CRM SPH spacing. Chrono's C++ CRM wheeled demo uses 0.04 m.",
    )
    parser.add_argument("--terrain-length-m", type=float, default=150.0)
    parser.add_argument("--terrain-width-m", type=float, default=150.0)
    parser.add_argument("--terrain-depth-m", type=float, default=0.25)
    parser.add_argument("--boundary-margin-m", type=float, default=5.0)
    parser.add_argument("--vehicle-x-m", type=float, default=8.0)
    parser.add_argument("--vehicle-z-m", type=float, default=0.7)
    parser.add_argument("--active-domain-m", type=float, nargs=3, default=[2.0, 2.0, 1.0])
    parser.add_argument("--active-domain-delay-s", type=float, default=0.1)
    parser.add_argument("--render", action="store_true", help="Open a VSG visualization while logging.")
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=800)
    parser.add_argument("--camera-track-z-m", type=float, default=1.2)
    parser.add_argument("--camera-x-m", type=float, default=7.0)
    parser.add_argument("--camera-y-m", type=float, default=-9.0)
    parser.add_argument("--camera-z-m", type=float, default=5.0)
    parser.add_argument("--camera-distance-m", type=float, default=11.0)
    parser.add_argument("--camera-height-m", type=float, default=4.0)
    parser.add_argument(
        "--camera-angle-deg",
        type=float,
        default=150.0,
        help="Chase camera angle: 0 is behind the vehicle, 180 is in front.",
    )
    parser.add_argument(
        "--camera-state",
        choices=("chase", "follow", "track", "free", "fixed"),
        default="chase",
        help="Initial VSG camera mode. Chase allows keyboard rotation/zoom; track/fixed use world camera positions.",
    )
    parser.add_argument("--camera-target-x-m", type=float, default=None)
    parser.add_argument("--camera-target-y-m", type=float, default=0.0)
    parser.add_argument("--camera-target-z-m", type=float, default=1.2)
    parser.add_argument("--driver-profile", choices=("launch_brake", "straight", "sine_steer"), default="launch_brake")
    parser.add_argument("--throttle-peak", type=float, default=0.55)
    parser.add_argument("--throttle-delay-s", type=float, default=0.2)
    parser.add_argument("--throttle-rise-s", type=float, default=1.0)
    parser.add_argument("--throttle-hold-s", type=float, default=5.0)
    parser.add_argument("--throttle-release-s", type=float, default=1.0)
    parser.add_argument("--brake-peak", type=float, default=0.35)
    parser.add_argument("--brake-delay-s", type=float, default=0.8)
    parser.add_argument("--brake-rise-s", type=float, default=0.5)
    parser.add_argument("--brake-hold-s", type=float, default=1.5)
    parser.add_argument("--brake-release-s", type=float, default=1.0)
    parser.add_argument("--steering-amplitude", type=float, default=0.0)
    parser.add_argument("--steering-frequency-hz", type=float, default=0.08)
    parser.add_argument("--progress-interval-s", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--chrono-data-root",
        type=str,
        default=None,
        help="Override Chrono data root, e.g. $CONDA_PREFIX/share/chrono/data.",
    )
    return parser.parse_args()


def build_collector_config(args: argparse.Namespace) -> dict[str, Any]:
    chrono_data_root = args.chrono_data_root or "chrono/data"
    return {
        "dataset_name": "hmmwv_crm_smoke",
        "chrono_data_root": chrono_data_root,
        "vehicle_data_root": str(Path(chrono_data_root) / "vehicle")
        if args.chrono_data_root
        else "chrono/data/vehicle",
        "output_subdir": str(args.output_dir),
        "simulation": {
            "step_size_s": float(args.step_size_s),
            "tire_step_size_s": float(args.step_size_s),
            "record_step_s": float(args.record_step_s),
            "driver_sample_step_s": float(args.record_step_s),
            "validation_ratio": 0.0,
            "chrono_threads": int(args.chrono_threads),
            "collision_threads": 1,
            "eigen_threads": 1,
        },
        "vehicle": {
            "model": "HMMWV_Full",
            "contact_method": "SMC",
            "chassis_fixed": False,
            "init": {
                "x_m": float(args.vehicle_x_m),
                "y_m": 0.0,
                "z_m": float(args.vehicle_z_m),
            },
            "engine_model": "SHAFTS",
            "transmission_model": "AUTOMATIC_SHAFTS",
            "drive_type": "AWD",
            "steering_type": "PITMAN_ARM",
            "tire_model": "RIGID_MESH",
        },
        "terrain": {
            "type": "crm",
            "length_m": float(args.terrain_length_m),
            "width_m": float(args.terrain_width_m),
            "depth_m": float(args.terrain_depth_m),
            "center_m": [0.0, 0.0, 0.0],
            "boundary_margin_m": float(args.boundary_margin_m),
            "initial_spacing_m": float(args.crm_spacing_m),
            "active_domain_m": [float(v) for v in args.active_domain_m],
            "active_domain_delay_s": float(args.active_domain_delay_s),
            "soil": {
                "density": 1700.0,
                "cohesion": 5000.0,
                "friction": 0.8,
                "young_modulus_pa": 1_000_000.0,
                "poisson_ratio": 0.3,
                "mu_I0": 0.04,
                "average_diam_m": 0.005,
            },
            "sph": {
                "integration_scheme": "RK2",
                "d0_multiplier": 1.0,
                "free_surface_threshold": 2.0,
                "artificial_viscosity": 0.5,
                "shifting_method": "NONE",
                "shifting_ppst_push": 1.0,
                "shifting_ppst_pull": 1.0,
                "viscosity_method": "ARTIFICIAL_BILATERAL",
                "boundary_method": "ADAMI",
                "num_proximity_search_steps": 4,
            },
        },
        "logging": {"include_tire_channels": True, "tire_force_source": "crm_fsi"},
        "scenario_generator": {
            "seed": 2026061801,
            "shuffle_seed": 2026061802,
            "warmup_s": float(args.warmup_s),
            "driver": {
                "mechanism": "ChDataDriver",
                "profile": str(args.driver_profile),
                "throttle_peak": float(args.throttle_peak),
                "throttle_delay_s": float(args.throttle_delay_s),
                "throttle_rise_s": float(args.throttle_rise_s),
                "throttle_hold_s": float(args.throttle_hold_s),
                "throttle_release_s": float(args.throttle_release_s),
                "brake_peak": float(args.brake_peak),
                "brake_delay_s": float(args.brake_delay_s),
                "brake_rise_s": float(args.brake_rise_s),
                "brake_hold_s": float(args.brake_hold_s),
                "brake_release_s": float(args.brake_release_s),
                "steering_amplitude": float(args.steering_amplitude),
                "steering_frequency_hz": float(args.steering_frequency_hz),
            },
            "families": [],
        },
    }


def configure_crm_terrain(hmmwv: Any, config: dict[str, Any]) -> tuple[Any, list[WheelRuntime]]:
    terrain_cfg = config["terrain"]
    soil_cfg = terrain_cfg["soil"]
    sph_cfg = terrain_cfg["sph"]
    simulation_cfg = config["simulation"]
    step_size_s = float(simulation_cfg["step_size_s"])
    chrono_threads = max(int(simulation_cfg.get("chrono_threads", 12)), 1)
    collision_threads = max(int(simulation_cfg.get("collision_threads", 1)), 1)
    eigen_threads = max(int(simulation_cfg.get("eigen_threads", 1)), 1)

    vehicle = hmmwv.GetVehicle()
    system = hmmwv.GetSystem()
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetNumThreads(chrono_threads, collision_threads, eigen_threads)
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    terrain = veh.CRMTerrain(system, float(terrain_cfg["initial_spacing_m"]))
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    terrain.SetStepSizeCFD(step_size_s)
    terrain.RegisterVehicle(vehicle)

    mat_props = fsi.ElasticMaterialProperties()
    mat_props.density = float(soil_cfg["density"])
    mat_props.Young_modulus = float(soil_cfg["young_modulus_pa"])
    mat_props.Poisson_ratio = float(soil_cfg["poisson_ratio"])
    mat_props.mu_I0 = float(soil_cfg["mu_I0"])
    mat_props.mu_fric_s = float(soil_cfg["friction"])
    mat_props.mu_fric_2 = float(soil_cfg["friction"])
    mat_props.average_diam = float(soil_cfg["average_diam_m"])
    mat_props.cohesion_coeff = float(soil_cfg["cohesion"])
    terrain.SetElasticSPH(mat_props)

    sph_params = fsi.SPHParameters()
    sph_params.integration_scheme = fsi.IntegrationScheme_RK2
    sph_params.initial_spacing = float(terrain_cfg["initial_spacing_m"])
    sph_params.d0_multiplier = float(sph_cfg["d0_multiplier"])
    sph_params.free_surface_threshold = float(sph_cfg["free_surface_threshold"])
    sph_params.artificial_viscosity = float(sph_cfg["artificial_viscosity"])
    shifting_methods = {
        "NONE": fsi.ShiftingMethod_NONE,
        "PPST": fsi.ShiftingMethod_PPST,
        "XSPH": fsi.ShiftingMethod_XSPH,
        "DIFFUSION": fsi.ShiftingMethod_DIFFUSION,
        "DIFFUSION_XSPH": fsi.ShiftingMethod_DIFFUSION_XSPH,
        "PPST_XSPH": fsi.ShiftingMethod_PPST_XSPH,
    }
    shifting_method_name = str(sph_cfg["shifting_method"]).upper()
    sph_params.shifting_method = shifting_methods[shifting_method_name]
    sph_params.shifting_ppst_push = float(sph_cfg["shifting_ppst_push"])
    sph_params.shifting_ppst_pull = float(sph_cfg["shifting_ppst_pull"])
    sph_params.use_consistent_gradient_discretization = False
    sph_params.use_consistent_laplacian_discretization = False
    sph_params.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph_params.boundary_method = fsi.BoundaryMethod_ADAMI
    if hasattr(sph_params, "num_proximity_search_steps"):
        sph_params.num_proximity_search_steps = int(sph_cfg["num_proximity_search_steps"])
    terrain.SetSPHParameters(sph_params)

    wheels = collect_wheel_runtime(vehicle)
    mesh_filename = veh.GetVehicleDataFile("hmmwv/hmmwv_tire_coarse_closed.obj")
    geometry = chrono.ChBodyGeometry()
    geometry.coll_meshes.append(
        chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh_filename, chrono.VNULL)
    )
    for wheel in wheels:
        terrain.AddRigidBody(wheel.spindle, geometry, False)

    terrain.SetActiveDomain(chrono.ChVector3d(*terrain_cfg["active_domain_m"]))
    terrain.SetActiveDomainDelay(float(terrain_cfg["active_domain_delay_s"]))
    terrain.Construct(
        chrono.ChVector3d(
            float(terrain_cfg["length_m"]),
            float(terrain_cfg["width_m"]),
            float(terrain_cfg["depth_m"]),
        ),
        chrono.ChVector3d(*terrain_cfg["center_m"]),
        fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,
    )
    terrain.Initialize()
    return terrain, wheels


def collect_wheel_runtime(vehicle: Any) -> list[WheelRuntime]:
    by_spec: dict[tuple[int, Any], Any] = {}
    for axle_index, axle in enumerate(vehicle.GetAxles()):
        for wheel in axle.GetWheels():
            side = veh.LEFT if wheel.GetSpindle().GetPos().y > 0 else veh.RIGHT
            by_spec[(axle_index, side)] = wheel

    wheels: list[WheelRuntime] = []
    for name, axle_index, side in WHEEL_SPECS:
        wheel = by_spec[(axle_index, side)]
        tire = vehicle.GetTire(axle_index, side)
        wheels.append(
            WheelRuntime(
                name=name,
                axle_index=axle_index,
                side=side,
                spindle=wheel.GetSpindle(),
                radius_m=float(tire.GetRadius()),
            )
        )
    return wheels


def constant_profile(value: float) -> dict[str, Any]:
    return {"kind": "constant", "value": float(value)}


def piecewise_profile(points: list[tuple[float, float]]) -> dict[str, Any]:
    return {"kind": "piecewise_linear", "points": [[float(t), float(v)] for t, v in points]}


def sine_profile(
    amplitude: float,
    frequency_hz: float,
    start_s: float,
    end_s: float,
    offset: float = 0.0,
) -> dict[str, Any]:
    return {
        "kind": "sine",
        "amplitude": float(amplitude),
        "offset": float(offset),
        "frequency_hz": float(frequency_hz),
        "phase_rad": 0.0,
        "start_s": float(start_s),
        "end_s": float(end_s),
    }


def bounded_next(current_s: float, delta_s: float, duration_s: float) -> float:
    return min(float(current_s) + max(float(delta_s), 0.0), float(duration_s))


def build_crm_smoke_scenario(args: argparse.Namespace) -> dict[str, Any]:
    duration_s = float(args.duration_s)
    warmup_s = float(args.warmup_s)
    throttle_peak = max(min(float(args.throttle_peak), 1.0), 0.0)
    brake_peak = max(min(float(args.brake_peak), 1.0), 0.0)

    throttle_start_s = bounded_next(warmup_s, float(args.throttle_delay_s), duration_s)
    throttle_peak_s = bounded_next(throttle_start_s, float(args.throttle_rise_s), duration_s)
    throttle_hold_end_s = bounded_next(throttle_peak_s, float(args.throttle_hold_s), duration_s)
    throttle_release_end_s = bounded_next(throttle_hold_end_s, float(args.throttle_release_s), duration_s)
    brake_start_s = bounded_next(throttle_release_end_s, float(args.brake_delay_s), duration_s)
    brake_peak_s = bounded_next(brake_start_s, float(args.brake_rise_s), duration_s)
    brake_hold_end_s = bounded_next(brake_peak_s, float(args.brake_hold_s), duration_s)
    brake_release_end_s = bounded_next(brake_hold_end_s, float(args.brake_release_s), duration_s)

    if args.driver_profile == "launch_brake":
        throttle = piecewise_profile(
            [
                (0.0, 0.0),
                (throttle_start_s, 0.0),
                (throttle_peak_s, throttle_peak),
                (throttle_hold_end_s, throttle_peak),
                (throttle_release_end_s, 0.0),
                (duration_s, 0.0),
            ]
        )
    else:
        throttle = piecewise_profile(
            [
                (0.0, 0.0),
                (throttle_start_s, 0.0),
                (throttle_peak_s, throttle_peak),
                (duration_s, throttle_peak),
            ]
        )
    braking = constant_profile(0.0)
    if args.driver_profile == "launch_brake":
        braking = piecewise_profile(
            [
                (0.0, 0.0),
                (brake_start_s, 0.0),
                (brake_peak_s, brake_peak),
                (brake_hold_end_s, brake_peak),
                (brake_release_end_s, 0.0),
                (duration_s, 0.0),
            ]
        )

    steering = constant_profile(0.0)
    if args.driver_profile == "sine_steer":
        start_s = bounded_next(warmup_s, 0.5, duration_s)
        steering = sine_profile(
            amplitude=float(args.steering_amplitude),
            frequency_hz=float(args.steering_frequency_hz),
            start_s=start_s,
            end_s=duration_s,
        )

    return {
        "name": "hmmwv_crm_smoke_000",
        "family": f"crm_smoke_{args.driver_profile}",
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "driver": {
            "steering": steering,
            "throttle": throttle,
            "braking": braking,
        },
    }


def capture_base_fields(
    hmmwv: Any,
    scenario_name: str,
    scenario_family: str,
    episode_id: str,
    split: str,
    sample_index: int,
    time_s: float,
    driver_inputs: Any,
) -> dict[str, Any]:
    vehicle = hmmwv.GetVehicle()
    body = hmmwv.GetChassis().GetBody()
    ref = body.GetFrameRefToAbs()

    pos = ref.GetPos()
    quat = ref.GetRot()
    euler_zyx = quat.GetCardanAnglesZYX()
    vel_world = ref.GetPosDt()
    vel_body = ref.TransformDirectionParentToLocal(vel_world)
    acc_world = body.GetPosDt2()
    acc_body = ref.TransformDirectionParentToLocal(acc_world)
    ang_world = ref.GetAngVelParent()
    ang_body = ref.GetAngVelLocal()

    return {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "sample_index": sample_index,
        "time_s": time_s,
        "driver_steering": float(driver_inputs.m_steering),
        "driver_throttle": float(driver_inputs.m_throttle),
        "driver_braking": float(driver_inputs.m_braking),
        "pos_x_m": float(pos.x),
        "pos_y_m": float(pos.y),
        "pos_z_m": float(pos.z),
        "quat_e0": float(quat.e0),
        "quat_e1": float(quat.e1),
        "quat_e2": float(quat.e2),
        "quat_e3": float(quat.e3),
        "roll_rad": float(vehicle.GetRoll()),
        "pitch_rad": float(vehicle.GetPitch()),
        "yaw_rad": float(euler_zyx.z),
        "vel_world_x_mps": float(vel_world.x),
        "vel_world_y_mps": float(vel_world.y),
        "vel_world_z_mps": float(vel_world.z),
        "vel_body_x_mps": float(vel_body.x),
        "vel_body_y_mps": float(vel_body.y),
        "vel_body_z_mps": float(vel_body.z),
        "acc_world_x_mps2": float(acc_world.x),
        "acc_world_y_mps2": float(acc_world.y),
        "acc_world_z_mps2": float(acc_world.z),
        "acc_body_x_mps2": float(acc_body.x),
        "acc_body_y_mps2": float(acc_body.y),
        "acc_body_z_mps2": float(acc_body.z),
        "ang_vel_world_x_radps": float(ang_world.x),
        "ang_vel_world_y_radps": float(ang_world.y),
        "ang_vel_world_z_radps": float(ang_world.z),
        "ang_vel_body_x_radps": float(ang_body.x),
        "ang_vel_body_y_radps": float(ang_body.y),
        "ang_vel_body_z_radps": float(ang_body.z),
        "speed_mps": float(vehicle.GetSpeed()),
        "body_slip_rad": float(vehicle.GetSlipAngle()),
        "roll_rate_radps": float(vehicle.GetRollRate()),
        "yaw_rate_radps": float(vehicle.GetYawRate()),
    }


def capture_crm_row(
    hmmwv: Any,
    terrain: Any,
    wheels: list[WheelRuntime],
    scenario_name: str,
    scenario_family: str,
    episode_id: str,
    split: str,
    sample_index: int,
    time_s: float,
    driver_inputs: Any,
) -> dict[str, Any]:
    row = capture_base_fields(
        hmmwv=hmmwv,
        scenario_name=scenario_name,
        scenario_family=scenario_family,
        episode_id=episode_id,
        split=split,
        sample_index=sample_index,
        time_s=time_s,
        driver_inputs=driver_inputs,
    )

    for wheel in wheels:
        spindle = wheel.spindle
        force = terrain.GetFsiBodyForce(spindle)
        torque = terrain.GetFsiBodyTorque(spindle)
        spin_axis = spindle.GetRot().GetAxisY()
        heading = spin_axis.Cross(WORLD_UP).GetNormalized()
        lateral = WORLD_UP.Cross(heading)
        v_wheel = spindle.GetPosDt()
        wheel_vx = float(v_wheel.Dot(heading))
        wheel_vy = float(v_wheel.Dot(lateral))
        omega = float(spindle.GetAngVelParent().Dot(spin_axis))
        slip_ratio = (omega * wheel.radius_m - wheel_vx) / max(abs(wheel_vx), 0.1)
        slip_angle = math.atan2(wheel_vy, max(abs(wheel_vx), 1e-6))

        prefix = wheel.name
        row[f"{prefix}_longitudinal_slip"] = float(slip_ratio)
        row[f"{prefix}_slip_angle_rad"] = float(slip_angle)
        row[f"{prefix}_camber_angle_rad"] = 0.0
        row[f"{prefix}_force_world_x_n"] = float(force.x)
        row[f"{prefix}_force_world_y_n"] = float(force.y)
        row[f"{prefix}_force_world_z_n"] = float(force.z)
        row[f"{prefix}_moment_world_x_nm"] = float(torque.x)
        row[f"{prefix}_moment_world_y_nm"] = float(torque.y)
        row[f"{prefix}_moment_world_z_nm"] = float(torque.z)
        row[f"{prefix}_force_wheel_fx_n"] = float(force.Dot(heading))
        row[f"{prefix}_force_wheel_fy_n"] = float(force.Dot(lateral))
        row[f"{prefix}_force_wheel_fz_n"] = float(force.Dot(WORLD_UP))
        row[f"{prefix}_spindle_omega_radps"] = omega
        row[f"{prefix}_wheel_vx_mps"] = wheel_vx
        row[f"{prefix}_slip_ratio"] = float(slip_ratio)
        row[f"{prefix}_deflection_m"] = 0.0

    return row


def enable_vehicle_visuals(hmmwv: Any) -> None:
    for setter in (
        hmmwv.SetChassisVisualizationType,
        hmmwv.SetSuspensionVisualizationType,
        hmmwv.SetSteeringVisualizationType,
        hmmwv.SetWheelVisualizationType,
        hmmwv.SetTireVisualizationType,
    ):
        setter(chrono.VisualizationType_PRIMITIVES)


def create_visualization(hmmwv: Any, terrain: Any, args: argparse.Namespace) -> Any:
    enable_vehicle_visuals(hmmwv)
    aabb = terrain.GetSPHBoundingBox()
    vis_fsi = fsi.ChSphVisualizationVSG(terrain.GetFsiSystemSPH())
    vis_fsi.EnableFluidMarkers(True)
    vis_fsi.EnableBoundaryMarkers(False)
    vis_fsi.EnableRigidBodyMarkers(False)
    vis_fsi.SetSPHColorCallback(
        fsi.ParticleHeightColorCallback(aabb.min.z, aabb.max.z),
        chrono.ChColormap.Type_BROWN,
    )

    vis = veh.ChWheeledVehicleVisualSystemVSG()
    vis.AttachVehicle(hmmwv.GetVehicle())
    vis.AttachPlugin(vis_fsi)
    vis.SetWindowTitle("HMMWV CRM smoke collection")
    vis.SetWindowSize(int(args.render_width), int(args.render_height))
    vis.SetWindowPosition(100, 100)
    vis.EnableSkyTexture()
    vis.SetLightIntensity(1.0)
    vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4)
    vis.SetCameraAngleDeg(40)
    vis.SetChaseCamera(
        chrono.ChVector3d(0.0, 0.0, float(args.camera_track_z_m)),
        float(args.camera_distance_m),
        float(args.camera_height_m),
    )
    vis.SetChaseCameraAngle(math.radians(float(args.camera_angle_deg)))

    camera_state = str(args.camera_state).lower()
    camera_pos = chrono.ChVector3d(float(args.camera_x_m), float(args.camera_y_m), float(args.camera_z_m))
    if camera_state == "follow":
        vis.SetChaseCameraState(veh.ChChaseCamera.Follow)
    elif camera_state == "track":
        vis.SetChaseCameraPosition(camera_pos)
    elif camera_state == "free":
        vis.SetChaseCameraState(veh.ChChaseCamera.Free)
    elif camera_state == "fixed":
        target_x = float(args.camera_target_x_m) if args.camera_target_x_m is not None else float(args.vehicle_x_m)
        camera_target = chrono.ChVector3d(
            target_x,
            float(args.camera_target_y_m),
            float(args.camera_target_z_m),
        )
        vis.SetChaseCameraPosition(camera_pos, camera_target)
    vis.Initialize()
    return vis


def assert_finite_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value in {key} at sample {row['sample_index']}")


def summarize_force(rows: list[dict[str, Any]], vehicle_mass_kg: float) -> dict[str, float]:
    wheel_names = [name for name, _, _ in WHEEL_SPECS]
    settled = [row for row in rows if 0.2 <= row["time_s"] <= 0.6]
    powered = [row for row in rows if row["time_s"] >= 1.2]
    target_rows = settled or rows
    fz_values = [sum(float(row[f"{name}_force_wheel_fz_n"]) for name in wheel_names) for row in target_rows]
    fx_values = [sum(float(row[f"{name}_force_wheel_fx_n"]) for name in wheel_names) for row in (powered or rows)]
    return {
        "mean_sum_fz_n": float(sum(fz_values) / max(len(fz_values), 1)),
        "mean_powered_fx_n": float(sum(fx_values) / max(len(fx_values), 1)),
        "weight_n": float(vehicle_mass_kg * GRAVITY),
    }


def summarize_pose(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "min_pos_z_m": min(float(row["pos_z_m"]) for row in rows),
        "min_pos_x_m": min(float(row["pos_x_m"]) for row in rows),
        "max_pos_x_m": max(float(row["pos_x_m"]) for row in rows),
        "max_abs_y_m": max(abs(float(row["pos_y_m"])) for row in rows),
        "min_pos_y_m": min(float(row["pos_y_m"]) for row in rows),
        "max_pos_y_m": max(float(row["pos_y_m"]) for row in rows),
        "final_pos_x_m": float(rows[-1]["pos_x_m"]),
        "final_pos_y_m": float(rows[-1]["pos_y_m"]),
        "final_speed_mps": float(rows[-1]["speed_mps"]),
    }


def terrain_bounds(config: dict[str, Any]) -> dict[str, float]:
    terrain_cfg = config["terrain"]
    center_x, center_y, _ = terrain_cfg["center_m"]
    half_length = 0.5 * float(terrain_cfg["length_m"])
    half_width = 0.5 * float(terrain_cfg["width_m"])
    return {
        "min_x_m": float(center_x) - half_length,
        "max_x_m": float(center_x) + half_length,
        "min_y_m": float(center_y) - half_width,
        "max_y_m": float(center_y) + half_width,
    }


def boundary_distance_m(pos: Any, bounds: dict[str, float]) -> float:
    return min(
        float(pos.x) - bounds["min_x_m"],
        bounds["max_x_m"] - float(pos.x),
        float(pos.y) - bounds["min_y_m"],
        bounds["max_y_m"] - float(pos.y),
    )


def run_episode(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = build_collector_config(args)
    scenario = build_crm_smoke_scenario(args)
    config["scenarios"] = [scenario]
    configure_chrono_data_paths(REPO_ROOT, config)

    output_root = args.output_dir.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    hmmwv = create_hmmwv(config)
    terrain, wheels = configure_crm_terrain(hmmwv, config)
    vehicle = hmmwv.GetVehicle()
    vis = create_visualization(hmmwv, terrain, args) if args.render else None
    driver_step_s = float(config["simulation"]["driver_sample_step_s"])
    driver_entries = build_driver_entries(scenario, driver_step_s)
    driver = veh.ChDataDriver(vehicle, driver_entries)
    driver.Initialize()
    bounds = terrain_bounds(config)
    boundary_margin_m = max(float(args.boundary_margin_m), 0.0)
    boundary_exit = False
    boundary_exit_time_s: float | None = None
    boundary_exit_pos: dict[str, float] | None = None

    aabb = terrain.GetSPHBoundingBox()
    print(f"CRM particles: {terrain.GetNumSPHParticles()}")
    print(f"CRM boundary BCE markers: {terrain.GetNumBoundaryBCEMarkers()}")
    print(f"CRM AABB z: [{aabb.min.z:.3f}, {aabb.max.z:.3f}]")
    if vis is not None:
        print(
            "VSG camera controls: Left/Right rotate, Up/Down zoom in Chase mode; "
            "press 1 Chase, 2 Follow, 3 Track, 4 Inside, 5 Free."
        )

    episode_id = scenario["name"]
    scenario_name = scenario["name"]
    scenario_family = scenario.get("family", scenario_name)
    split = assign_split(episode_id, 0.0)
    csv_path = episodes_dir / f"{episode_id}.csv"
    rows: list[dict[str, Any]] = []

    duration_s = float(scenario["duration_s"])
    step_size_s = float(args.step_size_s)
    record_step_s = float(args.record_step_s)
    next_record_time_s = float(scenario["warmup_s"])
    next_render_time_s = 0.0
    render_step_s = 1.0 / max(float(args.render_fps), 1e-6)
    next_progress_time_s = 0.0
    progress_interval_s = max(float(args.progress_interval_s), 0.0)
    sample_index = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_field_names(include_tires=True))
        writer.writeheader()

        while True:
            time_s = float(hmmwv.GetSystem().GetChTime())
            if time_s > duration_s + 1e-12:
                break

            pos = vehicle.GetPos()
            dist_to_boundary_m = boundary_distance_m(pos, bounds)
            if dist_to_boundary_m <= boundary_margin_m:
                boundary_exit = True
                boundary_exit_time_s = time_s
                boundary_exit_pos = {
                    "x_m": float(pos.x),
                    "y_m": float(pos.y),
                    "z_m": float(pos.z),
                    "distance_to_boundary_m": float(dist_to_boundary_m),
                }
                print(
                    "boundary exit at "
                    f"t={time_s:.2f}s pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) "
                    f"distance_to_boundary={dist_to_boundary_m:.2f}m"
                )
                break

            driver.Synchronize(time_s)
            driver_inputs = driver.GetInputs()
            terrain.Synchronize(time_s)
            hmmwv.Synchronize(time_s, driver_inputs, terrain)

            if progress_interval_s > 0 and time_s + 1e-12 >= next_progress_time_s:
                pos = vehicle.GetPos()
                print(
                    f"t={time_s:5.2f}s speed={vehicle.GetSpeed():5.2f} m/s "
                    f"pos=({pos.x:5.1f}, {pos.y:5.1f}, {pos.z:4.2f})"
                )
                next_progress_time_s += progress_interval_s

            if vis is not None and time_s + 1e-12 >= next_render_time_s:
                if not vis.Run():
                    break
                vis.Render()
                next_render_time_s += render_step_s

            if time_s + 1e-12 >= next_record_time_s:
                row = capture_crm_row(
                    hmmwv=hmmwv,
                    terrain=terrain,
                    wheels=wheels,
                    scenario_name=scenario_name,
                    scenario_family=scenario_family,
                    episode_id=episode_id,
                    split=split,
                    sample_index=sample_index,
                    time_s=time_s,
                    driver_inputs=driver_inputs,
                )
                writer.writerow(row)
                rows.append(row)
                sample_index += 1
                next_record_time_s += record_step_s

            if time_s >= duration_s:
                break

            if vis is not None:
                vis.Synchronize(time_s, driver_inputs)
                vis.Advance(step_size_s)
            driver.Advance(step_size_s)
            terrain.Advance(step_size_s)

    assert_finite_rows(rows)
    summary = summarize_force(rows, vehicle.GetMass())
    pose_summary = summarize_pose(rows)
    if summary["mean_sum_fz_n"] <= 0:
        raise ValueError("CRM smoke produced non-positive mean vertical tire load")
    if pose_summary["min_pos_z_m"] < -0.5:
        raise ValueError(
            f"CRM smoke vehicle left the terrain or fell through it: min pos_z={pose_summary['min_pos_z_m']:.2f} m"
        )

    episode_meta = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "csv_path": str(csv_path.relative_to(output_root)),
        "rows": len(rows),
        "duration_s": duration_s,
        "warmup_s": float(scenario["warmup_s"]),
        "driver_entry_count": len(build_time_grid(duration_s, driver_step_s)),
        "tire_nominal_radius_m": {wheel.name: wheel.radius_m for wheel in wheels},
        "terrain_type": "crm",
        "tire_force_source": "crm_fsi",
        "crm_particles": int(terrain.GetNumSPHParticles()),
        "crm_boundary_bce_markers": int(terrain.GetNumBoundaryBCEMarkers()),
        "crm_force_summary": summary,
        "crm_pose_summary": pose_summary,
        "terrain_bounds_m": bounds,
        "boundary_margin_m": boundary_margin_m,
        "terminated_near_boundary": boundary_exit,
        "boundary_exit_time_s": boundary_exit_time_s,
        "boundary_exit_pos": boundary_exit_pos,
    }
    (episodes_dir / f"{episode_id}.json").write_text(json.dumps(episode_meta, indent=2) + "\n")

    dataset_index = {
        "dataset_name": config["dataset_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": 1,
        "episodes": [
            {
                "episode_id": episode_id,
                "scenario_name": scenario_name,
                "scenario_family": scenario_family,
                "split": split,
                "csv_path": str(csv_path.relative_to(output_root)),
                "rows": len(rows),
                "duration_s": duration_s,
                "warmup_s": float(scenario["warmup_s"]),
            }
        ],
    }
    (output_root / "dataset_index.json").write_text(json.dumps(dataset_index, indent=2) + "\n")
    (output_root / "collector_config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    return dataset_index, episode_meta


def main() -> int:
    args = parse_args()
    dataset_index, episode_meta = run_episode(args)
    force_summary = episode_meta["crm_force_summary"]
    ratio = force_summary["mean_sum_fz_n"] / max(force_summary["weight_n"], 1.0)
    print(
        f"wrote {dataset_index['episode_count']} CRM smoke episode with "
        f"{episode_meta['rows']} rows to {args.output_dir}"
    )
    print(
        f"mean settled sum Fz = {force_summary['mean_sum_fz_n']:.0f} N "
        f"vs weight {force_summary['weight_n']:.0f} N (ratio {ratio:.2f})"
    )
    print(f"mean powered wheel-frame Fx = {force_summary['mean_powered_fx_n']:.0f} N")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
