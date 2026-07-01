"""Tracked-vehicle drive-mode dynamics data collection.

Drive-mode data collector for the base NN-ROM described in
``docs/tracked_vehicle_nn_rom_rl_plan.md``: the tracked base moves under
throttle/steering/brake while the 4-DOF arm stays welded at its imported home
pose. This is "drive mode" from ``docs/arm-dyn-model.md`` S3.1 -- base moves,
arm does not -- the counterpart to the reach-mode arm-only collector in
``arm_data.py``.

The scene -- M113 tracked vehicle + front-welded ``LRV_Arm`` + flat rigid
terrain, tuned for the single-pin track's stability requirements (small
physics step, high-iteration BB solver) -- is exactly
``nedm.arm_data.build_scene()``: the real mounted-arm configuration the
vehicle actually carries, so the base ROM's mass/inertia/CG match deployment
instead of a bare M113, just called with a much larger ``terrain_size_m``
(the terrain patch is a genuine finite box, and the 100 m default that's
plenty for reach-mode's stationary base is far too small once the base
actually drives -- see ``TERRAIN_SIZE_M`` below). Unlike ``arm_data.py``, the arm's four
``ChLinkMotorRotationAngle`` joint motors are left completely alone: with no
angle target ever set they hold ``q = 0`` (home) as a hard constraint, so the
arm rides as fixed dead weight and this module needs neither the PD actuator
nor the arm collision setup that ``arm_data.py`` builds for reach-mode data.

The maneuver library (straight launches, coast-down, steering arcs, S-turns,
pivot-like turns, brake-while-steering, broad random commands, stop-and-go --
plan S4.3) reuses the scenario_generator/driver-profile machinery already
built for the HMMWV collector (``nedm.generated_scenarios``,
``nedm.hmmwv_data.sample_channel``): the tracked-vehicle-specific variants
(sharp/pivot-like arcs, straight stop-and-go) are expressed as config-level
overrides of the existing ``step_steer``/``multi_steer`` templates -- see
``configs/tracked_vehicle_drive_v1.json`` -- not new generator code. Driver
commands are evaluated directly from the scenario's continuous profile at
each recorded step (no ``ChDataDriver`` table/interpolation layer), which is
simpler and needs no ``driver_sample_step_s`` knob.

Run in the NeDM conda env:

    conda run -n nedm python -m nedm.tracked_vehicle_data --config configs/tracked_vehicle_drive_v1.json

Add ``--render`` to watch one run in the Irrlicht viewer, ``--dry-run`` to
resolve the config without running Chrono, ``--list-scenarios`` to print the
materialized scenario names.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pychrono.vehicle as veh

from nedm.arm_data import SETTLE_TIME, STEP_SIZE, build_scene, make_vis
from nedm.generated_scenarios import expand_scenarios, validate_generator_config
from nedm.hmmwv_data import (
    assign_split,
    resolve_project_path,
    sample_channel,
    scenario_matches,
    validate_materialized_scenarios,
    validate_scenario,
)

# build_scene()'s terrain is a genuine finite rigid box, not an infinite
# plane -- driving past its edge means falling off it. The default 100 m used
# for reach-mode (stationary-base) collection is far too small once the base
# actually drives: a 45 s multi_steer episode at even a few m/s can cover
# tens of meters. Use a patch generous enough that no realistic episode in
# this collector's maneuver library gets truncated by hitting the edge (a
# bigger flat box patch costs nothing extra -- collision against it is O(1)).
TERRAIN_SIZE_M = 600.0

# Divergence guards for data-quality filtering (not RL termination logic --
# just keeps a rare solver hiccup from silently polluting the dataset).
MAX_ABS_ROLL_PITCH_RAD = math.radians(45.0)
WORKSPACE_BOUND_M = 0.9 * TERRAIN_SIZE_M / 2.0  # margin inside the terrain patch edge

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
    "roll_rate_radps",
    "yaw_rate_radps",
    "left_sprocket_speed_radps",
    "right_sprocket_speed_radps",
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
    diverged: bool
    diverge_reason: str | None


def repo_root_from_module() -> Path:
    """NeDM repo root (src/nedm/tracked_vehicle_data.py -> parents[2])."""
    return Path(__file__).resolve().parents[2]


def validate_config(config: dict[str, Any]) -> None:
    simulation = config["simulation"]
    if simulation["record_step_s"] <= 0:
        raise ValueError("simulation.record_step_s must be positive")
    if not 0.0 <= simulation["validation_ratio"] < 1.0:
        raise ValueError("simulation.validation_ratio must be in [0, 1)")
    if not config.get("scenarios") and "scenario_generator" not in config:
        raise ValueError("Config must define either scenarios or scenario_generator")
    if "scenario_generator" in config:
        validate_generator_config(config["scenario_generator"])
    for scenario in config.get("scenarios", []):
        validate_scenario(scenario)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["scenarios"] = expand_scenarios(config)
    validate_materialized_scenarios(config["scenarios"])
    return config


def build_output_root(repo_root: Path, config: dict[str, Any], override: str | None) -> Path:
    if override:
        return resolve_project_path(repo_root, override)
    return resolve_project_path(repo_root, config["output_subdir"])


def list_scenarios(config: dict[str, Any]) -> None:
    for scenario in config["scenarios"]:
        print(scenario["name"])


# ---------------------------------------------------------------------------
# Physics stepping (mirrors arm_data.py's tick handling: render at most once
# per recorded/control step, Synchronize+Advance every physics tick).
# ---------------------------------------------------------------------------
def _sync(m113: Any, terrain: Any, driver_inputs: Any, t: float, vis: Any, do_render: bool) -> bool:
    keep_going = True
    if vis is not None and do_render:
        keep_going = vis.Run()
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
    terrain.Synchronize(t)
    m113.Synchronize(t, driver_inputs)
    if vis is not None:
        vis.Synchronize(t, driver_inputs)
    return keep_going


def _advance(m113: Any, terrain: Any, vis: Any) -> None:
    terrain.Advance(STEP_SIZE)
    m113.Advance(STEP_SIZE)
    if vis is not None:
        vis.Advance(STEP_SIZE)


def _state_is_valid(row: dict[str, Any]) -> tuple[bool, str | None]:
    numeric_checks = (
        row["pos_x_m"], row["pos_y_m"], row["pos_z_m"],
        row["vel_body_x_mps"], row["vel_body_y_mps"], row["yaw_rate_radps"],
    )
    if not all(math.isfinite(v) for v in numeric_checks):
        return False, "non_finite_state"
    if abs(row["roll_rad"]) > MAX_ABS_ROLL_PITCH_RAD or abs(row["pitch_rad"]) > MAX_ABS_ROLL_PITCH_RAD:
        return False, "rollover"
    if abs(row["pos_x_m"]) > WORKSPACE_BOUND_M or abs(row["pos_y_m"]) > WORKSPACE_BOUND_M:
        return False, "out_of_bounds"
    return True, None


def capture_row(
    vehicle: Any,
    scenario_name: str,
    scenario_family: str,
    episode_id: str,
    split: str,
    sample_index: int,
    time_s: float,
    driver_inputs: Any,
) -> dict[str, Any]:
    body = vehicle.GetChassisBody()
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
    driveline = vehicle.GetDriveline()

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
        "roll_rate_radps": float(vehicle.GetRollRate()),
        "yaw_rate_radps": float(vehicle.GetYawRate()),
        "left_sprocket_speed_radps": float(driveline.GetSprocketSpeed(veh.LEFT)),
        "right_sprocket_speed_radps": float(driveline.GetSprocketSpeed(veh.RIGHT)),
    }


def run_episode(
    scenario: dict[str, Any],
    output_root: Path,
    record_step_s: float,
    validation_ratio: float,
    render: bool = False,
) -> EpisodeResult:
    """Run one drive-mode episode (fresh scene) and write its transition CSV."""
    m113, vehicle, terrain, _gripper = build_scene(terrain_size_m=TERRAIN_SIZE_M)
    vis = make_vis(vehicle, "Tracked-vehicle drive-mode data collection") if render else None
    if vis is not None:
        vehicle.EnableRealtime(True)

    system = m113.GetSystem()
    n_sub = max(1, int(round(record_step_s / STEP_SIZE)))

    # Physically settle onto the tracks before driving (braked, matching the
    # reach-mode collector's settle handling for this same fragile single-pin
    # track model); the scenario clock starts fresh at 0 after this.
    settle_inputs = veh.DriverInputs()
    settle_inputs.m_throttle = 0.0
    settle_inputs.m_steering = 0.0
    settle_inputs.m_braking = 1.0
    settle_tick = 0
    while system.GetChTime() < SETTLE_TIME - 1e-9:
        _sync(m113, terrain, settle_inputs, system.GetChTime(), vis, settle_tick % n_sub == 0)
        _advance(m113, terrain, vis)
        settle_tick += 1
    t0 = system.GetChTime()

    scenario_name = scenario["name"]
    scenario_family = scenario.get("family", scenario_name)
    episode_id = scenario_name
    split = assign_split(episode_id, validation_ratio)
    csv_path = output_root / "episodes" / f"{episode_id}.csv"
    meta_path = output_root / "episodes" / f"{episode_id}.json"

    duration_s = float(scenario["duration_s"])
    warmup_s = float(scenario["warmup_s"])
    driver_profile = scenario["driver"]

    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_steering = 0.0
    driver_inputs.m_braking = 0.0

    rows = 0
    sample_index = 0
    diverged = False
    diverge_reason: str | None = None
    tick = 0

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS)
        writer.writeheader()
        while True:
            t_local = system.GetChTime() - t0
            if t_local > duration_s + 1e-9:
                break

            is_boundary = tick % n_sub == 0
            if is_boundary:
                driver_inputs.m_steering = sample_channel(driver_profile["steering"], t_local, "steering")
                driver_inputs.m_throttle = sample_channel(driver_profile["throttle"], t_local, "throttle")
                driver_inputs.m_braking = sample_channel(driver_profile["braking"], t_local, "braking")

            keep_going = _sync(m113, terrain, driver_inputs, system.GetChTime(), vis, is_boundary)
            if not keep_going:
                break

            if is_boundary and t_local + 1e-9 >= warmup_s:
                row = capture_row(
                    vehicle, scenario_name, scenario_family, episode_id, split,
                    sample_index, t_local, driver_inputs,
                )
                writer.writerow(row)
                rows += 1
                sample_index += 1
                ok, reason = _state_is_valid(row)
                if not ok:
                    diverged = True
                    diverge_reason = reason
                    break

            if t_local >= duration_s:
                break

            _advance(m113, terrain, vis)
            tick += 1

    meta = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "csv_path": str(csv_path.relative_to(output_root)),
        "rows": rows,
        "duration_s": duration_s,
        "warmup_s": warmup_s,
        "record_step_s": record_step_s,
        "step_size_s": STEP_SIZE,
        "settle_time_s": SETTLE_TIME,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    return EpisodeResult(
        episode_id=episode_id, scenario_name=scenario_name, scenario_family=scenario_family,
        split=split, csv_path=csv_path, rows=rows, duration_s=duration_s, warmup_s=warmup_s,
        diverged=diverged, diverge_reason=diverge_reason,
    )


def write_dataset_summary(
    config: dict[str, Any], output_root: Path, episode_results: list[EpisodeResult]
) -> None:
    summary = {
        "dataset_name": config["dataset_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(episode_results),
        "episodes": [
            {
                "episode_id": r.episode_id,
                "scenario_name": r.scenario_name,
                "scenario_family": r.scenario_family,
                "split": r.split,
                "csv_path": str(r.csv_path.relative_to(output_root)),
                "rows": r.rows,
                "duration_s": r.duration_s,
                "warmup_s": r.warmup_s,
                "diverged": r.diverged,
                "diverge_reason": r.diverge_reason,
            }
            for r in episode_results
        ],
    }
    with (output_root / "dataset_index.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_root / "collector_config.resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def collect_dataset(
    config_path: Path,
    output_override: str | None = None,
    scenario_filter: str | None = None,
    start_index: int = 0,
    max_scenarios: int | None = None,
    dry_run: bool = False,
    render: bool = False,
) -> list[EpisodeResult]:
    repo_root = repo_root_from_module()
    config = load_config(config_path)
    output_root = build_output_root(repo_root, config, output_override)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "episodes").mkdir(parents=True, exist_ok=True)

    simulation_cfg = config["simulation"]
    record_step_s = float(simulation_cfg["record_step_s"])
    validation_ratio = float(simulation_cfg["validation_ratio"])

    selected_scenarios = [
        scenario for scenario in config["scenarios"]
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
            print(f"- {scenario['name']} ({scenario['duration_s']:.1f} s, family={scenario.get('family')})")
        if len(selected_scenarios) > len(preview):
            print(f"... {len(selected_scenarios) - len(preview)} more")
        return []

    # Sequential, one Chrono scene per episode (fresh build_scene() call) with
    # an explicit gc.collect() afterward -- matches arm_data.py's collection
    # loop for this same M113+arm scene rather than hmmwv_data.py's
    # multi-process --jobs option, since this scene is heavier and the known
    # failure mode here is memory/stability, not throughput.
    results: list[EpisodeResult] = []
    for scenario in selected_scenarios:
        result = run_episode(scenario, output_root, record_step_s, validation_ratio, render=render)
        flag = f"DIVERGED({result.diverge_reason})" if result.diverged else "ok"
        print(f"  {result.episode_id}: {result.rows} rows, {flag}, split={result.split}")
        results.append(result)
        gc.collect()

    write_dataset_summary(config, output_root, results)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect tracked-vehicle (M113 + arm-at-home) drive-mode dynamics data."
    )
    parser.add_argument("--config", default="configs/tracked_vehicle_drive_v1.json",
                        help="Path to the collector config JSON.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the output root.")
    parser.add_argument("--scenario-filter", default=None,
                        help="Only run scenarios whose names contain this substring.")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Only run the first N matching scenarios.")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Skip the first N matching scenarios before collection.")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="Print available scenarios and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve config and selected scenarios without running Chrono.")
    parser.add_argument("--render", action="store_true",
                        help="Open the Irrlicht viewer (debug; one window per episode).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
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
        dry_run=args.dry_run,
        render=args.render,
    )
    if not args.dry_run:
        total_rows = sum(result.rows for result in results)
        n_diverged = sum(1 for result in results if result.diverged)
        print(f"wrote {len(results)} episode files and {total_rows} samples ({n_diverged} diverged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
