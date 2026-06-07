from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pychrono as chrono
import pychrono.vehicle as veh

from nedm.generated_scenarios import expand_scenarios, validate_generator_config


CONTACT_METHODS = {
    "SMC": chrono.ChContactMethod_SMC,
    "NSC": chrono.ChContactMethod_NSC,
}

ENGINE_MODELS = {
    "SHAFTS": veh.EngineModelType_SHAFTS,
    "SIMPLE": veh.EngineModelType_SIMPLE,
}

TRANSMISSION_MODELS = {
    "AUTOMATIC_SHAFTS": veh.TransmissionModelType_AUTOMATIC_SHAFTS,
    "AUTOMATIC_SIMPLE_MAP": veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP,
}

DRIVE_TYPES = {
    "AWD": veh.DrivelineTypeWV_AWD,
    "RWD": veh.DrivelineTypeWV_RWD,
    "FWD": veh.DrivelineTypeWV_FWD,
}

STEERING_TYPES = {
    "PITMAN_ARM": veh.SteeringTypeWV_PITMAN_ARM,
    "PITMAN_ARM_SHAFTS": veh.SteeringTypeWV_PITMAN_ARM_SHAFTS,
}

TIRE_MODELS = {
    "TMEASY": veh.TireModelType_TMEASY,
    "PAC89": veh.TireModelType_PAC89,
    "FIALA": veh.TireModelType_FIALA,
    "RIGID": veh.TireModelType_RIGID,
    "RIGID_MESH": veh.TireModelType_RIGID_MESH,
}

WHEEL_SPECS = (
    ("tire_fl", 0, veh.LEFT),
    ("tire_fr", 0, veh.RIGHT),
    ("tire_rl", 1, veh.LEFT),
    ("tire_rr", 1, veh.RIGHT),
)

BASE_FIELDS = [
    "episode_id",
    "scenario_name",
    "scenario_family",
    "split",
    "sample_index",
    "time_s",
    "driver_steering",
    "driver_throttle",
    "driver_braking",
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "quat_e0",
    "quat_e1",
    "quat_e2",
    "quat_e3",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "vel_world_x_mps",
    "vel_world_y_mps",
    "vel_world_z_mps",
    "vel_body_x_mps",
    "vel_body_y_mps",
    "vel_body_z_mps",
    "acc_world_x_mps2",
    "acc_world_y_mps2",
    "acc_world_z_mps2",
    "acc_body_x_mps2",
    "acc_body_y_mps2",
    "acc_body_z_mps2",
    "ang_vel_world_x_radps",
    "ang_vel_world_y_radps",
    "ang_vel_world_z_radps",
    "ang_vel_body_x_radps",
    "ang_vel_body_y_radps",
    "ang_vel_body_z_radps",
    "speed_mps",
    "body_slip_rad",
    "roll_rate_radps",
    "yaw_rate_radps",
]


@dataclass
class EpisodeResult:
    episode_id: str
    scenario_name: str
    scenario_family: str
    split: str
    csv_path: Path
    rows: int
    duration_s: float
    warmup_s: float


def repo_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["scenarios"] = expand_scenarios(config)
    validate_materialized_scenarios(config["scenarios"])
    return config


def validate_config(config: dict[str, Any]) -> None:
    simulation = config["simulation"]
    if simulation["step_size_s"] <= 0:
        raise ValueError("simulation.step_size_s must be positive")
    if simulation["tire_step_size_s"] <= 0:
        raise ValueError("simulation.tire_step_size_s must be positive")
    if simulation["record_step_s"] <= 0:
        raise ValueError("simulation.record_step_s must be positive")
    if simulation["driver_sample_step_s"] <= 0:
        raise ValueError("simulation.driver_sample_step_s must be positive")
    if not 0.0 <= simulation["validation_ratio"] < 1.0:
        raise ValueError("simulation.validation_ratio must be in [0, 1)")
    if config["terrain"]["type"] != "rigid":
        raise ValueError("Only rigid terrain is supported in the first pipeline")
    if not config.get("scenarios") and "scenario_generator" not in config:
        raise ValueError("Config must define either scenarios or scenario_generator")
    if "scenario_generator" in config:
        validate_generator_config(config["scenario_generator"])
    for scenario in config.get("scenarios", []):
        validate_scenario(scenario)


def validate_scenario(scenario: dict[str, Any]) -> None:
    if scenario["duration_s"] <= 0:
        raise ValueError(f"Scenario {scenario['name']} has non-positive duration")
    if scenario["warmup_s"] < 0 or scenario["warmup_s"] >= scenario["duration_s"]:
        raise ValueError(f"Scenario {scenario['name']} warmup must be in [0, duration)")


def validate_materialized_scenarios(scenarios: list[dict[str, Any]]) -> None:
    if not scenarios:
        raise ValueError("No scenarios were materialized from the config")
    seen_names: set[str] = set()
    for scenario in scenarios:
        validate_scenario(scenario)
        if scenario["name"] in seen_names:
            raise ValueError(f"Duplicate scenario name: {scenario['name']}")
        seen_names.add(scenario["name"])


def resolve_project_path(repo_root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return repo_root / path


def configure_chrono_data_paths(repo_root: Path, config: dict[str, Any]) -> None:
    chrono_data_root = resolve_project_path(repo_root, config["chrono_data_root"])
    vehicle_data_root = resolve_project_path(repo_root, config["vehicle_data_root"])
    chrono.SetChronoDataPath(str(chrono_data_root) + "/")
    veh.SetDataPath(str(vehicle_data_root) + "/")


def build_output_root(repo_root: Path, config: dict[str, Any], override: str | None) -> Path:
    if override:
        return resolve_project_path(repo_root, override)
    return resolve_project_path(repo_root, config["output_subdir"])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def evaluate_piecewise_linear(points: list[list[float]], time_s: float) -> float:
    if time_s <= points[0][0]:
        return points[0][1]
    for start, end in zip(points, points[1:]):
        if time_s <= end[0]:
            alpha = 0.0 if end[0] == start[0] else (time_s - start[0]) / (end[0] - start[0])
            return lerp(start[1], end[1], alpha)
    return points[-1][1]


def evaluate_profile(profile: dict[str, Any], time_s: float) -> float:
    kind = profile["kind"]
    if kind == "constant":
        return float(profile["value"])
    if kind == "piecewise_linear":
        return float(evaluate_piecewise_linear(profile["points"], time_s))
    if kind == "sine":
        start_s = float(profile.get("start_s", 0.0))
        end_s = float(profile.get("end_s", math.inf))
        offset = float(profile.get("offset", 0.0))
        if time_s < start_s or time_s > end_s:
            return offset
        amplitude = float(profile["amplitude"])
        frequency_hz = float(profile["frequency_hz"])
        phase_rad = float(profile.get("phase_rad", 0.0))
        dt = time_s - start_s
        return offset + amplitude * math.sin(phase_rad + 2.0 * math.pi * frequency_hz * dt)
    if kind == "chirp":
        start_s = float(profile.get("start_s", 0.0))
        end_s = float(profile["end_s"])
        offset = float(profile.get("offset", 0.0))
        if time_s < start_s or time_s > end_s:
            return offset
        amplitude = float(profile["amplitude"])
        phase_rad = float(profile.get("phase_rad", 0.0))
        f0 = float(profile["start_frequency_hz"])
        f1 = float(profile["end_frequency_hz"])
        dt = time_s - start_s
        horizon = max(end_s - start_s, 1e-9)
        slope = (f1 - f0) / horizon
        phase = phase_rad + 2.0 * math.pi * (f0 * dt + 0.5 * slope * dt * dt)
        return offset + amplitude * math.sin(phase)
    raise ValueError(f"Unsupported driver profile kind: {kind}")


def sample_channel(profile: dict[str, Any], time_s: float, channel_name: str) -> float:
    value = evaluate_profile(profile, time_s)
    if channel_name == "steering":
        return clamp(value, -1.0, 1.0)
    return clamp(value, 0.0, 1.0)


def build_time_grid(duration_s: float, step_s: float) -> list[float]:
    times: list[float] = []
    current = 0.0
    while current < duration_s:
        times.append(round(current, 10))
        current += step_s
    if not times or abs(times[-1] - duration_s) > 1e-9:
        times.append(duration_s)
    return times


def build_driver_entries(scenario: dict[str, Any], driver_step_s: float) -> Any:
    entries = []
    for time_s in build_time_grid(float(scenario["duration_s"]), driver_step_s):
        steering = sample_channel(scenario["driver"]["steering"], time_s, "steering")
        throttle = sample_channel(scenario["driver"]["throttle"], time_s, "throttle")
        braking = sample_channel(scenario["driver"]["braking"], time_s, "braking")
        entries.append(veh.DataDriverEntry(time_s, steering, throttle, braking, 0.0))
    return veh.vector_Entry(entries)


def scenario_matches(scenario_name: str, scenario_filter: str | None) -> bool:
    if not scenario_filter:
        return True
    return scenario_filter.lower() in scenario_name.lower()


def assign_split(episode_id: str, validation_ratio: float) -> str:
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    scaled = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if scaled < validation_ratio else "train"


def create_hmmwv(config: dict[str, Any]) -> Any:
    vehicle_cfg = config["vehicle"]
    init_cfg = vehicle_cfg["init"]

    if vehicle_cfg["model"] != "HMMWV_Full":
        raise ValueError("Only HMMWV_Full is implemented in the first pipeline")

    hmmwv = veh.HMMWV_Full()
    hmmwv.SetContactMethod(CONTACT_METHODS[vehicle_cfg["contact_method"]])
    hmmwv.SetChassisFixed(bool(vehicle_cfg["chassis_fixed"]))
    hmmwv.SetInitPosition(
        chrono.ChCoordsysd(
            chrono.ChVector3d(init_cfg["x_m"], init_cfg["y_m"], init_cfg["z_m"]),
            chrono.ChQuaterniond(1, 0, 0, 0),
        )
    )
    hmmwv.SetEngineType(ENGINE_MODELS[vehicle_cfg["engine_model"]])
    hmmwv.SetTransmissionType(TRANSMISSION_MODELS[vehicle_cfg["transmission_model"]])
    hmmwv.SetDriveType(DRIVE_TYPES[vehicle_cfg["drive_type"]])
    hmmwv.SetSteeringType(STEERING_TYPES[vehicle_cfg["steering_type"]])
    hmmwv.SetTireType(TIRE_MODELS[vehicle_cfg["tire_model"]])
    hmmwv.SetTireStepSize(config["simulation"]["tire_step_size_s"])
    hmmwv.Initialize()

    hmmwv.SetChassisVisualizationType(chrono.VisualizationType_NONE)
    hmmwv.SetSuspensionVisualizationType(chrono.VisualizationType_NONE)
    hmmwv.SetSteeringVisualizationType(chrono.VisualizationType_NONE)
    hmmwv.SetWheelVisualizationType(chrono.VisualizationType_NONE)
    hmmwv.SetTireVisualizationType(chrono.VisualizationType_NONE)
    hmmwv.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    return hmmwv


def create_rigid_terrain(system: Any, config: dict[str, Any]) -> Any:
    terrain_cfg = config["terrain"]
    terrain = veh.RigidTerrain(system)

    if config["vehicle"]["contact_method"] == "NSC":
        patch_mat = chrono.ChContactMaterialNSC()
        patch_mat.SetFriction(terrain_cfg["friction"])
        patch_mat.SetRestitution(terrain_cfg["restitution"])
    else:
        patch_mat = chrono.ChContactMaterialSMC()
        patch_mat.SetFriction(terrain_cfg["friction"])
        patch_mat.SetRestitution(terrain_cfg["restitution"])
        patch_mat.SetYoungModulus(terrain_cfg["young_modulus_pa"])

    terrain.AddPatch(
        patch_mat,
        chrono.CSYSNORM,
        float(terrain_cfg["length_m"]),
        float(terrain_cfg["width_m"]),
    )
    terrain.Initialize()
    return terrain


def tire_field_names() -> list[str]:
    fields: list[str] = []
    for wheel_name, _, _ in WHEEL_SPECS:
        fields.extend(
            [
                f"{wheel_name}_longitudinal_slip",
                f"{wheel_name}_slip_angle_rad",
                f"{wheel_name}_camber_angle_rad",
                f"{wheel_name}_force_world_x_n",
                f"{wheel_name}_force_world_y_n",
                f"{wheel_name}_force_world_z_n",
                f"{wheel_name}_moment_world_x_nm",
                f"{wheel_name}_moment_world_y_nm",
                f"{wheel_name}_moment_world_z_nm",
            ]
        )
    return fields


def csv_field_names(include_tires: bool) -> list[str]:
    fields = list(BASE_FIELDS)
    if include_tires:
        fields.extend(tire_field_names())
    return fields


def capture_row(
    hmmwv: Any,
    terrain: Any,
    scenario_name: str,
    scenario_family: str,
    episode_id: str,
    split: str,
    sample_index: int,
    time_s: float,
    driver_inputs: Any,
    include_tires: bool,
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

    row: dict[str, Any] = {
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

    if include_tires:
        for wheel_name, axle_index, side in WHEEL_SPECS:
            tire = vehicle.GetTire(axle_index, side)
            tire_force = tire.ReportTireForce(terrain)
            row[f"{wheel_name}_longitudinal_slip"] = float(tire.GetLongitudinalSlip())
            row[f"{wheel_name}_slip_angle_rad"] = float(tire.GetSlipAngle())
            row[f"{wheel_name}_camber_angle_rad"] = float(tire.GetCamberAngle())
            row[f"{wheel_name}_force_world_x_n"] = float(tire_force.force.x)
            row[f"{wheel_name}_force_world_y_n"] = float(tire_force.force.y)
            row[f"{wheel_name}_force_world_z_n"] = float(tire_force.force.z)
            row[f"{wheel_name}_moment_world_x_nm"] = float(tire_force.moment.x)
            row[f"{wheel_name}_moment_world_y_nm"] = float(tire_force.moment.y)
            row[f"{wheel_name}_moment_world_z_nm"] = float(tire_force.moment.z)

    return row


def run_episode(
    config: dict[str, Any],
    scenario: dict[str, Any],
    output_root: Path,
    include_tires: bool,
) -> EpisodeResult:
    simulation_cfg = config["simulation"]
    step_size_s = float(simulation_cfg["step_size_s"])
    record_step_s = float(simulation_cfg["record_step_s"])
    driver_step_s = float(simulation_cfg["driver_sample_step_s"])

    scenario_name = scenario["name"]
    scenario_family = scenario.get("family", scenario_name)
    episode_id = scenario_name
    split = assign_split(episode_id, float(simulation_cfg["validation_ratio"]))
    episode_csv_path = output_root / "episodes" / f"{episode_id}.csv"
    episode_meta_path = output_root / "episodes" / f"{episode_id}.json"

    hmmwv = create_hmmwv(config)
    terrain = create_rigid_terrain(hmmwv.GetSystem(), config)
    driver_entries = build_driver_entries(scenario, driver_step_s)
    driver = veh.ChDataDriver(hmmwv.GetVehicle(), driver_entries)
    driver.Initialize()

    next_record_time_s = float(scenario["warmup_s"])
    sample_index = 0
    row_count = 0
    duration_s = float(scenario["duration_s"])

    with episode_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_field_names(include_tires))
        writer.writeheader()

        while True:
            time_s = float(hmmwv.GetSystem().GetChTime())
            if time_s > duration_s + 1e-9:
                break

            driver_inputs = driver.GetInputs()

            if time_s + 1e-9 >= next_record_time_s:
                writer.writerow(
                    capture_row(
                        hmmwv=hmmwv,
                        terrain=terrain,
                        scenario_name=scenario_name,
                        scenario_family=scenario_family,
                        episode_id=episode_id,
                        split=split,
                        sample_index=sample_index,
                        time_s=time_s,
                        driver_inputs=driver_inputs,
                        include_tires=include_tires,
                    )
                )
                row_count += 1
                sample_index += 1
                next_record_time_s += record_step_s

            if time_s >= duration_s:
                break

            driver.Synchronize(time_s)
            terrain.Synchronize(time_s)
            hmmwv.Synchronize(time_s, driver_inputs, terrain)

            driver.Advance(step_size_s)
            terrain.Advance(step_size_s)
            hmmwv.Advance(step_size_s)

    episode_meta = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "csv_path": str(episode_csv_path.relative_to(output_root)),
        "rows": row_count,
        "duration_s": duration_s,
        "warmup_s": float(scenario["warmup_s"]),
        "driver_entry_count": len(build_time_grid(duration_s, driver_step_s)),
    }
    with episode_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(episode_meta, handle, indent=2)

    return EpisodeResult(
        episode_id=episode_id,
        scenario_name=scenario_name,
        scenario_family=scenario_family,
        split=split,
        csv_path=episode_csv_path,
        rows=row_count,
        duration_s=duration_s,
        warmup_s=float(scenario["warmup_s"]),
    )


def write_dataset_summary(
    config: dict[str, Any],
    output_root: Path,
    episode_results: list[EpisodeResult],
) -> None:
    summary = {
        "dataset_name": config["dataset_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episode_results),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "scenario_name": episode.scenario_name,
                "scenario_family": episode.scenario_family,
                "split": episode.split,
                "csv_path": str(episode.csv_path.relative_to(output_root)),
                "rows": episode.rows,
                "duration_s": episode.duration_s,
                "warmup_s": episode.warmup_s,
            }
            for episode in episode_results
        ],
    }
    with (output_root / "dataset_index.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_root / "collector_config.resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def list_scenarios(config: dict[str, Any]) -> None:
    for scenario in config["scenarios"]:
        print(scenario["name"])


def build_worker_config(config: dict[str, Any]) -> dict[str, Any]:
    worker_config = dict(config)
    worker_config["scenarios"] = []
    if "scenario_generator" in worker_config:
        worker_config["scenario_generator"] = {}
    return worker_config


def execute_scenarios(
    worker_config: dict[str, Any],
    scenarios: list[dict[str, Any]],
    output_root: Path,
    include_tires: bool,
    jobs: int,
) -> list[EpisodeResult]:
    if jobs <= 1:
        return [
            run_episode(
                config=worker_config,
                scenario=scenario,
                output_root=output_root,
                include_tires=include_tires,
            )
            for scenario in scenarios
        ]

    results: list[EpisodeResult] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_name = {
            executor.submit(run_episode, worker_config, scenario, output_root, include_tires): scenario["name"]
            for scenario in scenarios
        }
        for future in as_completed(future_to_name):
            results.append(future.result())
    results.sort(key=lambda item: item.episode_id)
    return results


def collect_dataset(
    config_path: Path,
    output_override: str | None = None,
    scenario_filter: str | None = None,
    start_index: int = 0,
    max_scenarios: int | None = None,
    jobs: int = 1,
    dry_run: bool = False,
) -> list[EpisodeResult]:
    repo_root = repo_root_from_module()
    config = load_config(config_path)
    configure_chrono_data_paths(repo_root, config)
    output_root = build_output_root(repo_root, config, output_override)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "episodes").mkdir(parents=True, exist_ok=True)

    include_tires = bool(config["logging"]["include_tire_channels"])
    selected_scenarios = [
        scenario
        for scenario in config["scenarios"]
        if scenario_matches(scenario["name"], scenario_filter)
    ]
    if start_index:
        selected_scenarios = selected_scenarios[start_index:]
    if max_scenarios is not None:
        selected_scenarios = selected_scenarios[:max_scenarios]

    if dry_run:
        print(f"config: {config_path}")
        print(f"output: {output_root}")
        print(f"selected scenarios: {len(selected_scenarios)}")
        preview = selected_scenarios[: min(20, len(selected_scenarios))]
        for scenario in preview:
            print(f"- {scenario['name']} ({scenario['duration_s']} s)")
        if len(selected_scenarios) > len(preview):
            print(f"... {len(selected_scenarios) - len(preview)} more")
        return []

    worker_config = build_worker_config(config)
    results = execute_scenarios(
        worker_config=worker_config,
        scenarios=selected_scenarios,
        output_root=output_root,
        include_tires=include_tires,
        jobs=max(1, jobs),
    )
    write_dataset_summary(config, output_root, results)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a simple HMMWV dynamics dataset from PyChrono.")
    parser.add_argument(
        "--config",
        default="configs/hmmwv_overfit_v1.json",
        help="Path to the collector config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the output root.",
    )
    parser.add_argument(
        "--scenario-filter",
        default=None,
        help="Only run scenarios whose names contain this substring.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Only run the first N matching scenarios.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N matching scenarios before collection.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker processes for parallel episode generation.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenarios and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and selected scenarios without running Chrono.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config_path = resolve_project_path(repo_root_from_module(), args.config)
    config = load_config(config_path)

    if args.list_scenarios:
        list_scenarios(config)
        return 0

    results = collect_dataset(
        config_path=config_path,
        output_override=args.output_dir,
        scenario_filter=args.scenario_filter,
        start_index=args.start_index,
        max_scenarios=args.max_scenarios,
        jobs=args.jobs,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        total_rows = sum(result.rows for result in results)
        print(f"wrote {len(results)} episode files and {total_rows} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
