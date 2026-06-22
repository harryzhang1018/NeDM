"""CRM (deformable-soil SPH) terrain construction and tire-force capture for HMMWV.

This is the single source of truth for the Chrono CRMTerrain setup and the
FSI-based tire-channel capture. Both the data collector
(``scripts/collect_hmmwv_crm_smoke.py`` / ``collect_hmmwv_crm_dataset.py``) and
the RL CRM evaluation env (``nedm.rl.hmmwv_chrono_crm_tracking_env``) import from
here, so the eval terrain physics stays identical to the physics the dynamics
model was trained on.

CRM differs from the rigid/SCM path in three ways the callers must respect:

1. The terrain owns the coupled FSI + multibody advance. Step it with
   ``terrain.Advance(dt)`` only -- do NOT also call ``hmmwv.Advance(dt)``.
2. Tire forces are read back from the FSI solver per spindle body
   (``GetFsiBodyForce`` / ``GetFsiBodyTorque``), not from ``ReportTireForce``.
3. ``configure_crm_terrain`` reconfigures the system solver/timestepper/threads,
   so it must run after ``create_hmmwv`` and before any stepping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pychrono as chrono
import pychrono.fsi as fsi
import pychrono.vehicle as veh

from nedm.hmmwv_data import WHEEL_SPECS, capture_row


WORLD_UP = chrono.ChVector3d(0, 0, 1)
GRAVITY = 9.81

SHIFTING_METHODS = {
    "NONE": fsi.ShiftingMethod_NONE,
    "PPST": fsi.ShiftingMethod_PPST,
    "XSPH": fsi.ShiftingMethod_XSPH,
    "DIFFUSION": fsi.ShiftingMethod_DIFFUSION,
    "DIFFUSION_XSPH": fsi.ShiftingMethod_DIFFUSION_XSPH,
    "PPST_XSPH": fsi.ShiftingMethod_PPST_XSPH,
}


@dataclass(frozen=True)
class WheelRuntime:
    """Per-wheel handles needed to read FSI forces back from the SPH terrain."""

    name: str
    axle_index: int
    side: Any
    spindle: Any
    radius_m: float


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


def configure_crm_terrain(hmmwv: Any, config: dict[str, Any]) -> tuple[Any, list[WheelRuntime]]:
    """Build a CRMTerrain for ``hmmwv`` from ``config["terrain"]`` and register its wheels.

    Mirrors ``scripts/collect_hmmwv_crm_smoke.configure_crm_terrain``. Reconfigures
    the multibody system solver/timestepper/threads (CRM requires it), so call this
    immediately after ``create_hmmwv`` and before stepping.
    """
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
    sph_params.shifting_method = SHIFTING_METHODS[str(sph_cfg["shifting_method"]).upper()]
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


def capture_crm_tire_fields(terrain: Any, wheels: list[WheelRuntime]) -> dict[str, float]:
    """Per-wheel tire channels read from the FSI solver (CRM force source).

    Produces the same keys as ``nedm.hmmwv_data.tire_field_names()`` but sources
    the contact force/torque from ``GetFsiBodyForce``/``GetFsiBodyTorque`` on each
    spindle body. Channels with no SPH analogue (camber, deflection) are zero.
    """
    fields: dict[str, float] = {}
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
        fields[f"{prefix}_longitudinal_slip"] = float(slip_ratio)
        fields[f"{prefix}_slip_angle_rad"] = float(slip_angle)
        fields[f"{prefix}_camber_angle_rad"] = 0.0
        fields[f"{prefix}_force_world_x_n"] = float(force.x)
        fields[f"{prefix}_force_world_y_n"] = float(force.y)
        fields[f"{prefix}_force_world_z_n"] = float(force.z)
        fields[f"{prefix}_moment_world_x_nm"] = float(torque.x)
        fields[f"{prefix}_moment_world_y_nm"] = float(torque.y)
        fields[f"{prefix}_moment_world_z_nm"] = float(torque.z)
        fields[f"{prefix}_force_wheel_fx_n"] = float(force.Dot(heading))
        fields[f"{prefix}_force_wheel_fy_n"] = float(force.Dot(lateral))
        fields[f"{prefix}_force_wheel_fz_n"] = float(force.Dot(WORLD_UP))
        fields[f"{prefix}_spindle_omega_radps"] = omega
        fields[f"{prefix}_wheel_vx_mps"] = wheel_vx
        fields[f"{prefix}_slip_ratio"] = float(slip_ratio)
        fields[f"{prefix}_deflection_m"] = 0.0
    return fields


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
    """Full CSV row for a CRM sample: shared body state + FSI-sourced tire channels."""
    row = capture_row(
        hmmwv=hmmwv,
        terrain=terrain,
        scenario_name=scenario_name,
        scenario_family=scenario_family,
        episode_id=episode_id,
        split=split,
        sample_index=sample_index,
        time_s=time_s,
        driver_inputs=driver_inputs,
        include_tires=False,
    )
    row.update(capture_crm_tire_fields(terrain, wheels))
    return row


def validate_crm_config(config: dict[str, Any]) -> None:
    simulation = config["simulation"]
    for key in ("step_size_s", "tire_step_size_s"):
        if float(simulation[key]) <= 0:
            raise ValueError(f"simulation.{key} must be positive")
    if config["terrain"].get("type") != "crm":
        raise ValueError("CRM config requires terrain.type == 'crm'")


def load_crm_config(config_path: str | Path) -> dict[str, Any]:
    """Load a CRM collector-style config for evaluation use.

    Unlike ``nedm.hmmwv_data.load_config`` this accepts ``terrain.type == 'crm'``
    and does not materialize scenarios -- the RL eval env drives the vehicle from
    a reference set, not from generated driver profiles.
    """
    config = json.loads(Path(config_path).read_text())
    validate_crm_config(config)
    config.setdefault("scenarios", [])
    return config
