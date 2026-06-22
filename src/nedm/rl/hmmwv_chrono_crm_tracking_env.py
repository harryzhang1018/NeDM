from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import pychrono.vehicle as veh

from nedm.hmmwv_data import capture_row
from nedm.hmmwv_crm import capture_crm_row, configure_crm_terrain, load_crm_config
from nedm.rl.hmmwv_chrono_tracking_env import ChronoHMMWVSim, HMMWVChronoTrackingEnv


class HMMWVChronoCRMTrackingEnv(HMMWVChronoTrackingEnv):
    """Chrono HMMWV tracking eval on CRM (deformable-soil SPH) terrain.

    Same observation/reward/reference interface as ``HMMWVChronoTrackingEnv`` --
    only the terrain and the way the vehicle is stepped differ. The vehicle,
    references, warm-start, observations, reward, and termination are all reused
    from the base class. CRM-specific overrides:

    * ``_create_terrain`` builds a ``CRMTerrain`` and returns the per-wheel FSI
      handles (instead of flat/heightmap ``RigidTerrain``).
    * ``_advance_sim_steps`` advances the terrain only -- the CRM/FSI terrain
      co-advances the registered multibody system, so calling ``hmmwv.Advance``
      as well would double-integrate the vehicle.
    * ``_capture_state_pose_np`` reads tire channels from the FSI solver.

    CRM is far more expensive than rigid terrain (each substep is a full SPH
    solve over millions of particles). Keep the eval terrain only as large as the
    references need, and expect minutes-per-reference, not seconds.
    """

    def _load_chrono_config(self, config_path: Path) -> dict[str, Any]:
        # The rigid-terrain loader rejects terrain.type == 'crm' and tries to
        # materialize scenarios; the CRM loader accepts the collector config the
        # CRM dataset was generated with and skips scenario expansion.
        return load_crm_config(config_path)

    def _create_terrain(self, hmmwv: Any, config: dict[str, Any], reference_id: int) -> tuple[Any, Any]:
        # CRM terrain is homogeneous (no per-episode heightmap), so reference_id
        # is unused here -- every reference drives over the same soil bed.
        terrain, wheels = configure_crm_terrain(hmmwv, config)
        return terrain, wheels

    def _advance_sim_steps(self, sim: ChronoHMMWVSim, num_chrono_steps: int) -> None:
        if self.vis is not None and sim is self._render_sim:
            raise NotImplementedError("CRM eval rendering is not supported; see start_render().")
        for _ in range(num_chrono_steps):
            time_s = float(sim.hmmwv.GetSystem().GetChTime())
            sim.terrain.Synchronize(time_s)
            sim.hmmwv.Synchronize(time_s, sim.driver_inputs, sim.terrain)
            # CRMTerrain.Advance co-steps the coupled FSI + multibody system. Do
            # NOT also call sim.hmmwv.Advance() (the rigid path does) -- that would
            # integrate the vehicle twice per substep.
            sim.terrain.Advance(self.chrono_step_size_s)

    def _capture_state_pose_np(self, sim: ChronoHMMWVSim | None) -> tuple[np.ndarray, np.ndarray]:
        if sim is None:
            raise RuntimeError("Chrono simulation is not initialized")
        time_s = float(sim.hmmwv.GetSystem().GetChTime())
        if self.include_tires:
            row = capture_crm_row(
                hmmwv=sim.hmmwv,
                terrain=sim.terrain,
                wheels=sim.wheels,
                scenario_name="rl_tracking",
                scenario_family="rl_tracking",
                episode_id="rl_tracking",
                split="eval",
                sample_index=0,
                time_s=time_s,
                driver_inputs=sim.driver_inputs,
            )
        else:
            row = capture_row(
                hmmwv=sim.hmmwv,
                terrain=sim.terrain,
                scenario_name="rl_tracking",
                scenario_family="rl_tracking",
                episode_id="rl_tracking",
                split="eval",
                sample_index=0,
                time_s=time_s,
                driver_inputs=sim.driver_inputs,
                include_tires=False,
            )
        state = np.asarray([float(row[field]) for field in self.state_fields], dtype=np.float32)
        pose = np.asarray([float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["yaw_rad"])], dtype=np.float32)
        return state, pose

    def start_render(self, reference_id: int | None = None, output_dir: str | Path | None = None) -> Path:
        # The rigid env renders with Irrlicht and draws the reference line on the
        # RigidTerrain ground body; CRM terrain has no such patch/ground body and
        # the collector renders the SPH field with a separate VSG plugin. Porting
        # that to the offscreen eval path is future work.
        raise NotImplementedError(
            "CRM tracking eval does not support rendering yet. Run without --render "
            "(see scripts/collect_hmmwv_crm_smoke.py --render for CRM VSG visualization)."
        )

    def _add_reference_line(self, sim: ChronoHMMWVSim, reference_id: int) -> None:  # pragma: no cover
        raise NotImplementedError("CRM tracking eval does not support rendering yet.")
