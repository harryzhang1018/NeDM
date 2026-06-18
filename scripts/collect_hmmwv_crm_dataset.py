from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pychrono.vehicle as veh

from nedm.generated_scenarios import expand_scenarios, validate_generator_config
from nedm.hmmwv_data import (
    WHEEL_SPECS,
    assign_split,
    build_driver_entries,
    build_time_grid,
    configure_chrono_data_paths,
    create_hmmwv,
    csv_field_names,
    resolve_project_path,
)

import collect_hmmwv_crm_smoke as crm


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    scenario_name: str
    scenario_family: str
    split: str
    csv_path: Path
    rows: int
    duration_s: float
    warmup_s: float
    terminated_near_boundary: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect generated HMMWV CRM terrain episodes using the rigid-terrain ChDataDriver mechanism."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenario-filter", type=str, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--chrono-data-root",
        type=str,
        default=None,
        help="Override Chrono data root, e.g. $CONDA_PREFIX/share/chrono/data.",
    )
    return parser.parse_args()


def load_config(config_path: Path, chrono_data_root: str | None = None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    if chrono_data_root:
        config["chrono_data_root"] = chrono_data_root
        config["vehicle_data_root"] = str(Path(chrono_data_root) / "vehicle")
    config["scenarios"] = expand_scenarios(config)
    if not config["scenarios"]:
        raise ValueError("No scenarios were materialized from the config")
    return config


def validate_config(config: dict[str, Any]) -> None:
    simulation = config["simulation"]
    for key in ("step_size_s", "tire_step_size_s", "record_step_s", "driver_sample_step_s"):
        if float(simulation[key]) <= 0:
            raise ValueError(f"simulation.{key} must be positive")
    for key in ("chrono_threads", "collision_threads", "eigen_threads"):
        if key in simulation and int(simulation[key]) <= 0:
            raise ValueError(f"simulation.{key} must be positive")
    if config["terrain"]["type"] != "crm":
        raise ValueError("CRM collector requires terrain.type = 'crm'")
    if not 0.0 <= float(simulation["validation_ratio"]) < 1.0:
        raise ValueError("simulation.validation_ratio must be in [0, 1)")
    if not config.get("scenarios") and "scenario_generator" not in config:
        raise ValueError("Config must define either scenarios or scenario_generator")
    if "scenario_generator" in config:
        validate_generator_config(config["scenario_generator"])


def build_output_root(config: dict[str, Any], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    return resolve_project_path(REPO_ROOT, config["output_subdir"])


def scenario_matches(scenario_name: str, scenario_filter: str | None) -> bool:
    return not scenario_filter or scenario_filter.lower() in scenario_name.lower()


def completed_result_from_sidecar(output_root: Path, sidecar_path: Path) -> EpisodeResult | None:
    try:
        meta = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    csv_path = output_root / meta.get("csv_path", "")
    if not csv_path.is_file() or int(meta.get("rows", 0)) <= 0:
        return None
    return EpisodeResult(
        episode_id=str(meta["episode_id"]),
        scenario_name=str(meta["scenario_name"]),
        scenario_family=str(meta["scenario_family"]),
        split=str(meta["split"]),
        csv_path=csv_path,
        rows=int(meta["rows"]),
        duration_s=float(meta["duration_s"]),
        warmup_s=float(meta["warmup_s"]),
        terminated_near_boundary=bool(meta.get("terminated_near_boundary", False)),
    )


def run_episode(
    config: dict[str, Any],
    scenario: dict[str, Any],
    output_root: Path,
    progress_interval_s: float,
) -> EpisodeResult:
    simulation_cfg = config["simulation"]
    step_size_s = float(simulation_cfg["step_size_s"])
    record_step_s = float(simulation_cfg["record_step_s"])
    driver_step_s = float(simulation_cfg["driver_sample_step_s"])

    scenario_name = scenario["name"]
    scenario_family = scenario.get("family", scenario_name)
    episode_id = scenario_name
    split = assign_split(episode_id, float(simulation_cfg["validation_ratio"]))
    csv_path = output_root / "episodes" / f"{episode_id}.csv"
    sidecar_path = output_root / "episodes" / f"{episode_id}.json"

    hmmwv = create_hmmwv(config)
    terrain, wheels = crm.configure_crm_terrain(hmmwv, config)
    vehicle = hmmwv.GetVehicle()
    driver_entries = build_driver_entries(scenario, driver_step_s)
    driver = veh.ChDataDriver(vehicle, driver_entries)
    driver.Initialize()

    bounds = crm.terrain_bounds(config)
    boundary_margin_m = max(float(config["terrain"].get("boundary_margin_m", 5.0)), 0.0)
    boundary_exit = False
    boundary_exit_time_s: float | None = None
    boundary_exit_pos: dict[str, float] | None = None

    rows: list[dict[str, Any]] = []
    duration_s = float(scenario["duration_s"])
    next_record_time_s = float(scenario["warmup_s"])
    next_progress_time_s = 0.0
    sample_index = 0

    print(
        f"START {episode_id} family={scenario_family} duration={duration_s:.2f}s "
        f"split={split} particles={terrain.GetNumSPHParticles()} bce={terrain.GetNumBoundaryBCEMarkers()}",
        flush=True,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_field_names(include_tires=True))
        writer.writeheader()

        while True:
            time_s = float(hmmwv.GetSystem().GetChTime())
            if time_s > duration_s + 1e-12:
                break

            pos = vehicle.GetPos()
            dist_to_boundary_m = crm.boundary_distance_m(pos, bounds)
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
                    f"BOUNDARY {episode_id} t={time_s:.2f}s "
                    f"pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) "
                    f"distance={dist_to_boundary_m:.2f}m",
                    flush=True,
                )
                break

            driver.Synchronize(time_s)
            driver_inputs = driver.GetInputs()
            terrain.Synchronize(time_s)
            hmmwv.Synchronize(time_s, driver_inputs, terrain)

            if progress_interval_s > 0 and time_s + 1e-12 >= next_progress_time_s:
                pos = vehicle.GetPos()
                print(
                    f"{episode_id} t={time_s:5.2f}s speed={vehicle.GetSpeed():5.2f} m/s "
                    f"pos=({pos.x:6.1f}, {pos.y:6.1f}, {pos.z:4.2f})",
                    flush=True,
                )
                next_progress_time_s += progress_interval_s

            if time_s + 1e-12 >= next_record_time_s:
                row = crm.capture_crm_row(
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

            driver.Advance(step_size_s)
            terrain.Advance(step_size_s)

    if not rows:
        raise ValueError(f"{episode_id} produced no rows")
    crm.assert_finite_rows(rows)
    force_summary = crm.summarize_force(rows, vehicle.GetMass())
    pose_summary = crm.summarize_pose(rows)
    if force_summary["mean_sum_fz_n"] <= 0:
        raise ValueError(f"{episode_id} produced non-positive mean vertical tire load")
    if pose_summary["min_pos_z_m"] < -0.5:
        raise ValueError(f"{episode_id} left terrain or fell through: min z={pose_summary['min_pos_z_m']:.2f} m")

    sidecar = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "csv_path": str(csv_path.relative_to(output_root)),
        "rows": len(rows),
        "duration_s": duration_s,
        "warmup_s": float(scenario["warmup_s"]),
        "driver_entry_count": len(build_time_grid(duration_s, driver_step_s)),
        "driver": scenario["driver"],
        "tire_nominal_radius_m": {wheel.name: wheel.radius_m for wheel in wheels},
        "terrain_type": "crm",
        "tire_force_source": "crm_fsi",
        "crm_particles": int(terrain.GetNumSPHParticles()),
        "crm_boundary_bce_markers": int(terrain.GetNumBoundaryBCEMarkers()),
        "crm_force_summary": force_summary,
        "crm_pose_summary": pose_summary,
        "terrain_bounds_m": bounds,
        "boundary_margin_m": boundary_margin_m,
        "terminated_near_boundary": boundary_exit,
        "boundary_exit_time_s": boundary_exit_time_s,
        "boundary_exit_pos": boundary_exit_pos,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"DONE {episode_id} rows={len(rows)} final=({pose_summary['final_pos_x_m']:.1f}, "
        f"{pose_summary['final_pos_y_m']:.1f}) boundary={boundary_exit}",
        flush=True,
    )
    return EpisodeResult(
        episode_id=episode_id,
        scenario_name=scenario_name,
        scenario_family=scenario_family,
        split=split,
        csv_path=csv_path,
        rows=len(rows),
        duration_s=duration_s,
        warmup_s=float(scenario["warmup_s"]),
        terminated_near_boundary=boundary_exit,
    )


def write_dataset_summary(config: dict[str, Any], output_root: Path, results: list[EpisodeResult]) -> None:
    results = sorted(results, key=lambda item: item.episode_id)
    summary = {
        "dataset_name": config["dataset_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(results),
        "episodes": [
            {
                "episode_id": result.episode_id,
                "scenario_name": result.scenario_name,
                "scenario_family": result.scenario_family,
                "split": result.split,
                "csv_path": str(result.csv_path.relative_to(output_root)),
                "rows": result.rows,
                "duration_s": result.duration_s,
                "warmup_s": result.warmup_s,
                "terminated_near_boundary": result.terminated_near_boundary,
            }
            for result in results
        ],
    }
    (output_root / "dataset_index.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "collector_config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")


def collect_dataset(args: argparse.Namespace) -> list[EpisodeResult]:
    config = load_config(args.config, chrono_data_root=args.chrono_data_root)
    configure_chrono_data_paths(REPO_ROOT, config)
    output_root = build_output_root(config, args.output_dir)
    if output_root.exists() and args.overwrite and not args.resume:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "episodes").mkdir(parents=True, exist_ok=True)

    scenarios = [s for s in config["scenarios"] if scenario_matches(s["name"], args.scenario_filter)]
    scenarios = scenarios[max(args.start_index, 0) :]
    if args.max_scenarios is not None:
        scenarios = scenarios[: max(args.max_scenarios, 0)]

    results: list[EpisodeResult] = []
    for scenario in scenarios:
        sidecar_path = output_root / "episodes" / f"{scenario['name']}.json"
        if args.resume and sidecar_path.is_file():
            completed = completed_result_from_sidecar(output_root, sidecar_path)
            if completed is not None:
                print(f"SKIP {completed.episode_id} rows={completed.rows}", flush=True)
                results.append(completed)
                write_dataset_summary(config, output_root, results)
                continue
        result = run_episode(config, scenario, output_root, args.progress_interval_s)
        results.append(result)
        write_dataset_summary(config, output_root, results)

    write_dataset_summary(config, output_root, results)
    return results


def main() -> int:
    args = parse_args()
    results = collect_dataset(args)
    boundary_count = sum(1 for result in results if result.terminated_near_boundary)
    print(f"wrote {len(results)} CRM episodes; boundary-terminated={boundary_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
