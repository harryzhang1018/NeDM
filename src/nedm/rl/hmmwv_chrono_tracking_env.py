from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

import pychrono as chrono
import pychrono.vehicle as veh

from nedm.hmmwv_data import (
    capture_row,
    configure_chrono_data_paths,
    create_hmmwv,
    create_rigid_terrain,
    load_config,
    repo_root_from_module,
    resolve_height_map,
    resolve_project_path,
)
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.rl.hmmwv_tracking_env import default_env_cfg, merge_env_cfg, wrap_angle
from nedm.rl.references import ReferenceSet, load_reference_set
from nedm.training.dataset import load_metadata


@dataclass
class ChronoHMMWVSim:
    hmmwv: Any
    terrain: Any
    driver_inputs: Any
    # CRM-only: per-wheel handles used to read FSI tire forces. None for rigid terrain.
    wheels: Any = None


def default_chrono_env_cfg() -> dict[str, Any]:
    cfg = default_env_cfg()
    cfg.update(
        {
            "num_envs": 1,
            "device": "cuda",
            "auto_reset": False,
            "chrono_config": "configs/hmmwv_overfit_v1.json",
            "chrono_step_size_s": None,
            "warm_start_context": True,
            # Run Chrono against reference actions before policy hand-off so the
            # vehicle/terrain state settles before eval metrics start.
            "pre_roll_time_s": 6.0,
            # Max steering-command change per policy step; None disables the filter.
            "steering_rate_limit": None,
            # Offscreen Irrlicht rendering (saves one PNG per render frame; no interactive window required,
            # but still needs an X/GL context, e.g. a desktop session or `xvfb-run`).
            "render": False,
            "render_fps": 50.0,
            "render_width": 1280,
            "render_height": 720,
            "render_output_dir": None,
            "render_camera_distance": 6.0,
            "render_camera_height": 0.5,
            "render_line_z_m": 0.3,
            # True: detailed HMMWV mesh (demo_VEH_HMMWV.py look). False: robust all-PRIMITIVES.
            "render_vehicle_mesh": True,
        }
    )
    return cfg


def load_checkpoint_metadata(
    checkpoint_path: str | Path,
    processed_dataset_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int, float]:
    checkpoint = torch.load(resolve_dynamics_checkpoint_path(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if processed_dataset_dir is not None:
        config = dict(config)
        config["processed_dataset_dir"] = str(Path(processed_dataset_dir).expanduser().resolve())
    metadata = checkpoint.get("metadata")
    if metadata is None:
        metadata = load_metadata(Path(config["processed_dataset_dir"]).expanduser().resolve())
    return metadata, config, int(config["model"]["block_size"]), float(metadata["dt_s"])


def state_fields_require_tire_capture(state_fields: list[str]) -> bool:
    return any(field.startswith("tire_") for field in state_fields)


class HMMWVChronoTrackingEnv(VecEnv):
    """Chrono HMMWV trajectory tracking env for policy evaluation.

    The observation/reward interface mirrors ``HMMWVNeuralTrackingEnv`` so a PPO
    policy trained against the NN dynamics can be evaluated against Chrono.
    Chrono stepping is CPU-bound and intentionally serial.
    """

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = default_chrono_env_cfg()
        self.cfg = merge_env_cfg({**self.cfg, **(cfg or {})})
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 3
        self.action_repeat = int(self.cfg["action_repeat"])
        self.obs_history_steps = int(self.cfg["obs_history_steps"])
        self.reference_preview_steps = int(self.cfg["reference_preview_steps"])
        self.max_episode_length = int(self.cfg["max_episode_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", False))
        self.warm_start_context = bool(self.cfg.get("warm_start_context", True))
        self.pre_roll_time_s = float(self.cfg.get("pre_roll_time_s", 6.0))

        self.metadata, self.dynamics_config, self.context_steps, self.dt_s = load_checkpoint_metadata(
            checkpoint_path=self.cfg["dynamics_checkpoint"],
            processed_dataset_dir=self.cfg.get("processed_dataset_dir"),
        )
        self.step_dt = self.dt_s * self.action_repeat
        self.pre_roll_steps = max(0, int(round(self.pre_roll_time_s / self.dt_s)))
        self.policy_start_ref_step = self.pre_roll_steps + self.context_steps - 1
        if self.obs_history_steps > self.context_steps:
            raise ValueError(
                f"obs_history_steps={self.obs_history_steps} exceeds dynamics context {self.context_steps}"
            )

        self.state_fields = list(self.metadata["state_fields"])
        self.action_fields = list(self.metadata["action_fields"])
        self.state_index = {field_name: index for index, field_name in enumerate(self.state_fields)}
        state_error_fields = self.cfg["reward"].get("state_error_fields")
        if state_error_fields is None:
            self.reward_state_indices = torch.arange(len(self.state_fields), dtype=torch.long, device=self.device)
        else:
            missing = [field_name for field_name in state_error_fields if field_name not in self.state_index]
            if missing:
                raise ValueError(f"Unknown reward.state_error_fields: {missing}")
            self.reward_state_indices = torch.tensor(
                [self.state_index[field_name] for field_name in state_error_fields],
                dtype=torch.long,
                device=self.device,
            )
            if self.reward_state_indices.numel() == 0:
                raise ValueError("reward.state_error_fields must not be empty")
        self.include_tires = state_fields_require_tire_capture(self.state_fields)
        self.reference_set = load_reference_set(self.cfg["reference_path"])
        self._validate_reference_set(self.reference_set)

        self.reference_states = torch.as_tensor(self.reference_set.states, dtype=torch.float32, device=self.device)
        self.reference_actions = torch.as_tensor(self.reference_set.actions, dtype=torch.float32, device=self.device)
        self.reference_poses = torch.as_tensor(self.reference_set.poses, dtype=torch.float32, device=self.device)
        self.num_references = int(self.reference_states.shape[0])
        self.reference_length = int(self.reference_states.shape[1])
        min_required_length = self.pre_roll_steps + self.context_steps + self.action_repeat + 1
        if self.reference_length < min_required_length:
            raise ValueError(
                f"Reference length {self.reference_length} is too short for pre_roll_steps={self.pre_roll_steps}, "
                f"context={self.context_steps}, and action_repeat={self.action_repeat}"
            )
        max_policy_steps_from_refs = max(
            1,
            (self.reference_length - self.pre_roll_steps - self.context_steps - 1) // self.action_repeat,
        )
        if self.max_episode_length > max_policy_steps_from_refs:
            self.max_episode_length = max_policy_steps_from_refs

        normalization = self.metadata["normalization"]
        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32, device=self.device)
        self.state_std = torch.clamp(
            torch.tensor(normalization["state_std"], dtype=torch.float32, device=self.device),
            min=1.0e-6,
        )
        self.action_mean = torch.tensor(normalization["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.clamp(
            torch.tensor(normalization["action_std"], dtype=torch.float32, device=self.device),
            min=1.0e-6,
        )
        self.action_low = torch.tensor(self.cfg["action_low"], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.cfg["action_high"], dtype=torch.float32, device=self.device)
        action_center_cfg = self.cfg.get("action_center", "dataset_mean")
        if action_center_cfg == "dataset_mean":
            self.action_center = self.action_mean.clone()
        else:
            self.action_center = torch.tensor(action_center_cfg, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(self.cfg["action_scale"], dtype=torch.float32, device=self.device)

        repo_root = repo_root_from_module()
        chrono_config_path = resolve_project_path(repo_root, self.cfg["chrono_config"])
        self.chrono_config = self._load_chrono_config(chrono_config_path)
        configure_chrono_data_paths(repo_root, self.chrono_config)
        simulation_cfg = self.chrono_config["simulation"]
        self.chrono_step_size_s = (
            float(self.cfg["chrono_step_size_s"])
            if self.cfg.get("chrono_step_size_s") is not None
            else float(simulation_cfg["step_size_s"])
        )
        self.chrono_steps_per_nn_step = max(1, int(round(self.dt_s / self.chrono_step_size_s)))
        self.chrono_step_size_s = self.dt_s / self.chrono_steps_per_nn_step
        self.chrono_steps_per_policy_step = self.action_repeat * self.chrono_steps_per_nn_step

        self.state_hist = torch.zeros(
            self.num_envs,
            self.context_steps,
            len(self.state_fields),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_hist = torch.zeros(
            self.num_envs,
            self.context_steps,
            len(self.action_fields),
            dtype=torch.float32,
            device=self.device,
        )
        self.pose = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ref_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ref_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_pos_error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_track_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sims: list[ChronoHMMWVSim | None] = [None for _ in range(self.num_envs)]
        self.extras: dict[str, Any] = {}

        # Offscreen rendering state (only used when cfg["render"] is enabled and start_render() is called).
        self.render_enabled = bool(self.cfg.get("render", False))
        self.render_fps = float(self.cfg.get("render_fps", 50.0))
        self._render_every = max(1, int(round((1.0 / self.render_fps) / self.chrono_step_size_s)))
        self.vis: Any = None
        self._render_sim: ChronoHMMWVSim | None = None
        self._render_dir: Path | None = None
        self._render_substep = 0
        self._render_frame = 0

        self.num_obs = self._observation_dim()
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.reset()

    @property
    def unwrapped(self) -> "HMMWVChronoTrackingEnv":
        return self

    def _load_chrono_config(self, config_path: Path) -> dict[str, Any]:
        """Load and validate the Chrono collector config. Subclasses override for
        terrain types (e.g. CRM) that the rigid-terrain loader rejects."""
        return load_config(config_path)

    def _validate_reference_set(self, reference_set: ReferenceSet) -> None:
        if abs(reference_set.dt_s - self.dt_s) > 1.0e-8:
            raise ValueError(f"Reference dt_s={reference_set.dt_s} does not match checkpoint dt_s={self.dt_s}")
        if reference_set.state_fields != self.state_fields:
            raise ValueError(
                "Reference state fields do not match dynamics checkpoint metadata "
                f"({len(reference_set.state_fields)} reference fields vs "
                f"{len(self.state_fields)} checkpoint fields). Rebuild the RL reference set "
                "from the processed dataset used by the dynamics checkpoint."
            )
        if reference_set.action_fields != self.action_fields:
            raise ValueError(
                "Reference action fields do not match dynamics checkpoint metadata "
                f"({len(reference_set.action_fields)} reference fields vs "
                f"{len(self.action_fields)} checkpoint fields)."
            )

    def _observation_dim(self) -> int:
        state_dim = len(self.state_fields)
        action_dim = len(self.action_fields)
        return (
            self.obs_history_steps * state_dim
            + self.obs_history_steps * action_dim
            + state_dim
            + 3
            + self.reference_preview_steps * 3
            + action_dim
        )

    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, device=self.device)
        initial_reference_ids = self.cfg.get("initial_reference_ids")
        if initial_reference_ids is not None:
            reference_ids = torch.tensor(initial_reference_ids, dtype=torch.long, device=self.device)
            if reference_ids.numel() != env_ids.numel():
                raise ValueError("initial_reference_ids must match num_envs")
            self.reset_idx(env_ids, reference_ids=reference_ids)
        else:
            self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: torch.Tensor, reference_ids: torch.Tensor | None = None) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_cpu = env_ids.detach().cpu().long().tolist()
        if reference_ids is None:
            reference_ids_cpu = torch.randint(
                low=0,
                high=self.num_references,
                size=(len(env_ids_cpu),),
            ).long().tolist()
        else:
            reference_ids_cpu = reference_ids.detach().cpu().long().tolist()
            if len(reference_ids_cpu) != len(env_ids_cpu):
                raise ValueError("reference_ids must have the same length as env_ids")

        for env_index, reference_id in zip(env_ids_cpu, reference_ids_cpu, strict=True):
            ref_id_tensor = torch.tensor(reference_id, dtype=torch.long, device=self.device)
            self.ref_ids[env_index] = ref_id_tensor
            self.ref_step_buf[env_index] = self.policy_start_ref_step

            if self.warm_start_context:
                # Reset Chrono at reference index 0, pre-roll for stabilization, then keep the last
                # context window before policy hand-off.
                self.sims[env_index] = self._create_sim(reference_id, ref_step=0)
                state_history_np, pose_np = self._warm_start_sim_context(self.sims[env_index], reference_id)
                self.state_hist[env_index] = torch.as_tensor(
                    state_history_np, dtype=torch.float32, device=self.device
                )
                context_start = self.pre_roll_steps
                context_stop = context_start + self.context_steps
                self.action_hist[env_index] = self.reference_actions[reference_id, context_start:context_stop]
                self.pose[env_index] = torch.as_tensor(pose_np, dtype=torch.float32, device=self.device)
            else:
                context_start = self.pre_roll_steps
                context_stop = context_start + self.context_steps
                self.state_hist[env_index] = self.reference_states[reference_id, context_start:context_stop]
                self.action_hist[env_index] = self.reference_actions[reference_id, context_start:context_stop]
                self.pose[env_index] = self.reference_poses[reference_id, self.policy_start_ref_step]
                self.sims[env_index] = self._create_sim(reference_id, ref_step=0)
                reference_actions_np = self.reference_set.actions[reference_id]
                for ref_step in range(self.policy_start_ref_step):
                    self._set_driver_action_np(self.sims[env_index], reference_actions_np[ref_step])
                    self._advance_sim_steps(self.sims[env_index], self.chrono_steps_per_nn_step)
                state_np, pose_np = self._capture_state_pose_np(self.sims[env_index])
                self.state_hist[env_index, -1] = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
                self.pose[env_index] = torch.as_tensor(pose_np, dtype=torch.float32, device=self.device)

            self.actions[env_index] = self.reference_actions[reference_id, self.policy_start_ref_step]
            self.last_actions[env_index] = self.actions[env_index]

            self.episode_length_buf[env_index] = 0
            self.rew_buf[env_index] = 0.0
            self.reset_buf[env_index] = 0
            self.time_out_buf[env_index] = False
            self.episode_reward_sum[env_index] = 0.0
            self.episode_pos_error_sum[env_index] = 0.0
            self.episode_track_reward_sum[env_index] = 0.0

    def _create_sim(self, reference_id: int, ref_step: int) -> ChronoHMMWVSim:
        ref_pose = self.reference_set.poses[reference_id, ref_step]
        ref_state = self.reference_set.states[reference_id, ref_step]
        ref_action = self.reference_set.actions[reference_id, ref_step]

        config = copy.deepcopy(self.chrono_config)
        init_cfg = config["vehicle"]["init"]
        init_cfg["x_m"] = float(ref_pose[0])
        init_cfg["y_m"] = float(ref_pose[1])
        init_cfg["yaw_rad"] = float(ref_pose[2])
        init_cfg["fwd_vel_mps"] = max(0.0, float(ref_state[self.state_index["vel_body_x_mps"]]))

        hmmwv = create_hmmwv(config)
        terrain, wheels = self._create_terrain(hmmwv, config, reference_id)
        driver_inputs = veh.DriverInputs()
        driver_inputs.m_steering = float(ref_action[0])
        driver_inputs.m_throttle = float(ref_action[1])
        driver_inputs.m_braking = float(ref_action[2])
        driver_inputs.m_clutch = 0.0
        sim = ChronoHMMWVSim(hmmwv=hmmwv, terrain=terrain, driver_inputs=driver_inputs, wheels=wheels)
        if self.render_enabled:
            # Enable visual assets HERE, before any warm-start stepping. Chrono's suspension/steering
            # primitive visuals freeze their hardpoints in absolute coords at Initialize() and
            # re-express them in each body's frame when the asset is added; adding them after the
            # bodies have moved (warm-start) yields stale offsets that "hover" off the wheels. The
            # chassis/wheel/tire meshes use fixed body-local offsets, so they're unaffected.
            self._enable_vehicle_visuals(sim)
        return sim

    def _create_terrain(self, hmmwv: Any, config: dict[str, Any], reference_id: int) -> tuple[Any, Any]:
        """Build the terrain for one sim. Returns (terrain, wheels); wheels is None
        for rigid terrain (only CRM needs per-wheel FSI handles)."""
        # rigid_heightmap terrain is per-episode: each reference was collected on the
        # bumpy field deterministically assigned to its episode_id. Reproduce that exact
        # terrain so the eval drives over the same bumps. resolve_height_map returns None
        # for flat 'rigid' terrain, which leaves the previous flat behaviour unchanged.
        episode_id = self.reference_set.episode_ids[reference_id]
        height_map = resolve_height_map(config, episode_id)
        terrain = create_rigid_terrain(
            hmmwv.GetSystem(),
            config,
            height_map_path=height_map[1] if height_map is not None else None,
        )
        return terrain, None

    def _set_driver_action_np(self, sim: ChronoHMMWVSim, action: np.ndarray) -> None:
        sim.driver_inputs.m_steering = float(action[0])
        sim.driver_inputs.m_throttle = float(action[1])
        sim.driver_inputs.m_braking = float(action[2])
        sim.driver_inputs.m_clutch = 0.0

    def _advance_sim_steps(self, sim: ChronoHMMWVSim, num_chrono_steps: int) -> None:
        rendering = self.vis is not None and sim is self._render_sim
        for _ in range(num_chrono_steps):
            time_s = float(sim.hmmwv.GetSystem().GetChTime())
            sim.terrain.Synchronize(time_s)
            sim.hmmwv.Synchronize(time_s, sim.driver_inputs, sim.terrain)
            if rendering:
                self.vis.Synchronize(time_s, sim.driver_inputs)
            sim.terrain.Advance(self.chrono_step_size_s)
            sim.hmmwv.Advance(self.chrono_step_size_s)
            if rendering:
                self.vis.Advance(self.chrono_step_size_s)
                self._render_substep += 1
                if self._render_substep % self._render_every == 0 and self.vis.Run():
                    self.vis.BeginScene()
                    self.vis.Render()
                    self.vis.EndScene()
                    self.vis.WriteImageToFile(str(self._render_dir / f"frame_{self._render_frame:05d}.png"))
                    self._render_frame += 1

    def start_render(self, reference_id: int | None = None, output_dir: str | Path | None = None) -> Path:
        """Open an offscreen Chrono Irrlicht renderer attached to env 0's vehicle.

        Draws the reference trajectory (from the policy hand-off index onward) as a line in the scene
        and saves one PNG per render frame into ``output_dir``. Requires an X/GL context (desktop
        session or ``xvfb-run``); there is no interactive window dependency, but Irrlicht cannot create
        its video driver fully headless.
        """
        if not self.render_enabled:
            raise RuntimeError("Enable cfg['render'] before calling start_render().")
        import pychrono.irrlicht as chronoirr  # noqa: F401  # ensures the Irrlicht module is importable

        sim = self.sims[0]
        if sim is None:
            raise RuntimeError("Reset the env (reset()/reset_idx) before start_render().")
        ref_id = int(self.ref_ids[0].item()) if reference_id is None else int(reference_id)

        out = Path(output_dir or self.cfg.get("render_output_dir") or "artifacts/rl_runs/_render_frames")
        out = out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        self._render_dir = out

        # Vehicle visuals were already enabled at sim creation, before warm-start stepping
        # (see _create_sim) — enabling them here, after the bodies have moved, mislocates the
        # suspension/steering primitive cylinders. Just draw the reference line.
        # Reference line must live on a body already in the system before the vis binds assets.
        self._add_reference_line(sim, ref_id)

        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle(f"HMMWV RL tracking - ref {ref_id}: {self.reference_names()[ref_id]}")
        vis.SetWindowSize(int(self.cfg.get("render_width", 1280)), int(self.cfg.get("render_height", 720)))
        vis.SetChaseCamera(
            chrono.ChVector3d(0.0, 0.0, 1.75),
            float(self.cfg.get("render_camera_distance", 8.0)),
            float(self.cfg.get("render_camera_height", 1.0)),
        )
        vis.Initialize()
        for optional_setup in (vis.AddLogo, vis.AddSkyBox):
            try:
                optional_setup()
            except Exception:
                pass
        vis.AddLightDirectional()
        vis.AttachVehicle(sim.hmmwv.GetVehicle())
        try:
            vis.BindAll()
        except Exception:
            pass

        self.vis = vis
        self._render_sim = sim
        self._render_substep = 0
        self._render_frame = 0
        return out

    def _enable_vehicle_visuals(self, sim: ChronoHMMWVSim) -> None:
        # create_hmmwv() builds the vehicle with VisualizationType_NONE, so re-enable assets here.
        # With cfg["render_vehicle_mesh"] (default), match demo_VEH_HMMWV.py: detailed MESH for the
        # chassis/wheels/tires and PRIMITIVES for the suspension/steering linkages. Set it False to
        # fall back to all-PRIMITIVES (no external mesh assets needed, robust if meshes are missing).
        hmmwv = sim.hmmwv
        if bool(self.cfg.get("render_vehicle_mesh", True)):
            vis_types = {
                hmmwv.SetChassisVisualizationType: chrono.VisualizationType_MESH,
                hmmwv.SetSuspensionVisualizationType: chrono.VisualizationType_MESH,
                hmmwv.SetSteeringVisualizationType: chrono.VisualizationType_MESH,
                hmmwv.SetWheelVisualizationType: chrono.VisualizationType_MESH,
                hmmwv.SetTireVisualizationType: chrono.VisualizationType_MESH,
            }
        else:
            vis_types = {
                setter: chrono.VisualizationType_PRIMITIVES
                for setter in (
                    hmmwv.SetChassisVisualizationType,
                    hmmwv.SetSuspensionVisualizationType,
                    hmmwv.SetSteeringVisualizationType,
                    hmmwv.SetWheelVisualizationType,
                    hmmwv.SetTireVisualizationType,
                )
            }
        for setter, vis_type in vis_types.items():
            try:
                setter(vis_type)
            except Exception:
                pass

    def _add_reference_line(self, sim: ChronoHMMWVSim, reference_id: int) -> None:
        poses = self.reference_set.poses[reference_id]
        start = max(0, self.policy_start_ref_step)
        z = float(self.cfg.get("render_line_z_m", 0.3))
        points = chrono.vector_ChVector3d()
        for step in range(start, poses.shape[0]):
            points.append(chrono.ChVector3d(float(poses[step, 0]), float(poses[step, 1]), z))
        line = chrono.ChVisualShapeLine()
        line.SetLineGeometry(chrono.ChLineBezier(chrono.ChBezierCurve(points, False)))
        line.SetNumRenderPoints(max(2 * len(points), 400))
        try:
            line.SetColor(chrono.ChColor(0.05, 0.85, 0.25))
        except Exception:
            pass
        # Attach the line to the EXISTING terrain ground body. Adding a new ChBody to the vehicle's
        # ChSystem reorders the solver's body/variable layout and perturbs its floating-point
        # evaluation; the tracking policy then amplifies that into large trajectory divergence
        # (ref 9: 0.30 m -> 13.4 m). Reusing a body already in the system leaves the physics untouched.
        ground = sim.terrain.GetPatches()[0].GetGroundBody()
        ground.AddVisualShape(line)

    def close_render(self) -> None:
        if self.vis is not None:
            try:
                self.vis.GetDevice().closeDevice()
            except Exception:
                pass
        self.vis = None
        self._render_sim = None

    def _warm_start_sim_context(self, sim: ChronoHMMWVSim, reference_id: int) -> tuple[np.ndarray, np.ndarray]:
        state_history = np.empty(
            (self.context_steps, len(self.state_fields)),
            dtype=np.float32,
        )
        reference_actions = self.reference_set.actions[reference_id]
        for ref_step in range(self.pre_roll_steps):
            self._set_driver_action_np(sim, reference_actions[ref_step])
            self._advance_sim_steps(sim, self.chrono_steps_per_nn_step)

        state_np, pose_np = self._capture_state_pose_np(sim)
        state_history[0] = state_np

        for ref_step in range(1, self.context_steps):
            action_index = self.pre_roll_steps + ref_step - 1
            self._set_driver_action_np(sim, reference_actions[action_index])
            self._advance_sim_steps(sim, self.chrono_steps_per_nn_step)
            state_np, pose_np = self._capture_state_pose_np(sim)
            state_history[ref_step] = state_np

        self._set_driver_action_np(sim, reference_actions[self.policy_start_ref_step])
        return state_history, pose_np

    def _capture_state_pose_np(self, sim: ChronoHMMWVSim | None) -> tuple[np.ndarray, np.ndarray]:
        if sim is None:
            raise RuntimeError("Chrono simulation is not initialized")
        time_s = float(sim.hmmwv.GetSystem().GetChTime())
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
            include_tires=self.include_tires,
        )
        state = np.asarray([float(row[field]) for field in self.state_fields], dtype=np.float32)
        pose = np.asarray([float(row["pos_x_m"]), float(row["pos_y_m"]), float(row["yaw_rad"])], dtype=np.float32)
        return state, pose

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        driver_actions = self._scale_policy_actions(actions)
        steering_rate_limit = self.cfg.get("steering_rate_limit")
        if steering_rate_limit is not None:
            limit = float(steering_rate_limit)
            driver_actions[:, 0] = torch.clamp(
                driver_actions[:, 0],
                self.last_actions[:, 0] - limit,
                self.last_actions[:, 0] + limit,
            )
        self.actions = driver_actions
        driver_actions_cpu = driver_actions.detach().cpu().numpy()

        for env_index in range(self.num_envs):
            sim = self.sims[env_index]
            if sim is None:
                raise RuntimeError(f"Chrono simulation {env_index} is not initialized")
            self._set_driver_action_np(sim, driver_actions_cpu[env_index])
            for _ in range(self.action_repeat):
                self._advance_sim_steps(sim, self.chrono_steps_per_nn_step)
                state_np, pose_np = self._capture_state_pose_np(sim)
                self.state_hist[env_index] = torch.roll(self.state_hist[env_index], shifts=-1, dims=0)
                self.action_hist[env_index] = torch.roll(self.action_hist[env_index], shifts=-1, dims=0)
                self.state_hist[env_index, -1] = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
                self.action_hist[env_index, -1] = driver_actions[env_index]
                self.pose[env_index] = torch.as_tensor(pose_np, dtype=torch.float32, device=self.device)

        self.ref_step_buf = torch.clamp(self.ref_step_buf + self.action_repeat, max=self.reference_length - 1)
        self.episode_length_buf += 1
        rewards, reward_terms = self._compute_reward(driver_actions)
        dones, time_outs = self._compute_dones(reward_terms["position_error_m"])
        self.rew_buf = rewards
        self.reset_buf = dones.long()
        self.time_out_buf = time_outs
        self.episode_reward_sum += rewards
        self.episode_pos_error_sum += reward_terms["position_error_m"]
        self.episode_track_reward_sum += reward_terms["track_reward"]

        extras = self._make_extras(reward_terms, dones, time_outs)
        self.last_actions = driver_actions.clone()
        if self.auto_reset:
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            if done_env_ids.numel() > 0:
                self.reset_idx(done_env_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, self.rew_buf, dones.long(), self.extras

    def _scale_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(policy_actions)
        return torch.clamp(self.action_center + self.action_scale * bounded, self.action_low, self.action_high)

    def _reference_state_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.reference_states[self.ref_ids, self.ref_step_buf],
            self.reference_poses[self.ref_ids, self.ref_step_buf],
        )

    def _pose_error_local(self, ref_pose: torch.Tensor) -> torch.Tensor:
        dx_world = ref_pose[:, 0] - self.pose[:, 0]
        dy_world = ref_pose[:, 1] - self.pose[:, 1]
        cos_yaw = torch.cos(self.pose[:, 2])
        sin_yaw = torch.sin(self.pose[:, 2])
        dx_local = cos_yaw * dx_world + sin_yaw * dy_world
        dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
        yaw_error = wrap_angle(ref_pose[:, 2] - self.pose[:, 2])
        return torch.stack([dx_local, dy_local, yaw_error], dim=-1)

    def _compute_reward(self, driver_actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_cfg = self.cfg["reward"]
        ref_state, ref_pose = self._reference_state_pose()
        current_state = self.state_hist[:, -1, :]
        pose_error = self._pose_error_local(ref_pose)
        position_error = torch.linalg.norm(pose_error[:, :2], dim=-1)
        yaw_error = pose_error[:, 2]
        state_error_norm = (current_state - ref_state) / self.state_std

        position_loss = torch.square(position_error / float(reward_cfg["position_sigma_m"]))
        yaw_loss = torch.square(yaw_error / float(reward_cfg["yaw_sigma_rad"]))
        reward_state_error_norm = state_error_norm[:, self.reward_state_indices]
        state_loss = torch.mean(torch.square(reward_state_error_norm / float(reward_cfg["state_sigma"])), dim=-1)
        tracking_loss = (
            float(reward_cfg["position_weight"]) * position_loss
            + float(reward_cfg["yaw_weight"]) * yaw_loss
            + float(reward_cfg["state_weight"]) * state_loss
        )
        track_reward = torch.exp(-tracking_loss)

        action_rate = torch.sum(torch.square(driver_actions - self.last_actions), dim=-1)
        throttle_brake = driver_actions[:, 1] * driver_actions[:, 2]
        reward = (
            track_reward
            - float(reward_cfg["action_rate_weight"]) * action_rate
            - float(reward_cfg["throttle_brake_weight"]) * throttle_brake
        )
        return reward, {
            "track_reward": track_reward,
            "tracking_loss": tracking_loss,
            "position_error_m": position_error,
            "yaw_error_abs_rad": torch.abs(yaw_error),
            "state_error_norm": torch.sqrt(torch.mean(torch.square(reward_state_error_norm), dim=-1)),
            "action_rate": action_rate,
            "throttle_brake": throttle_brake,
        }

    def _compute_dones(self, position_error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        termination_cfg = self.cfg["termination"]
        current_state = self.state_hist[:, -1, :]
        roll = current_state[:, self.state_index["roll_rad"]]
        pitch = current_state[:, self.state_index["pitch_rad"]]
        failed = (
            (position_error > float(termination_cfg["max_position_error_m"]))
            | (torch.abs(roll) > float(termination_cfg["max_abs_roll_rad"]))
            | (torch.abs(pitch) > float(termination_cfg["max_abs_pitch_rad"]))
            | (~torch.isfinite(current_state).all(dim=-1))
            | (~torch.isfinite(self.pose).all(dim=-1))
        )
        reference_ended = self.ref_step_buf >= (self.reference_length - 1)
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = failed | reference_ended | time_outs
        return dones, time_outs | reference_ended

    def _make_extras(
        self,
        reward_terms: dict[str, torch.Tensor],
        dones: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "observations": {"critic": self.obs_buf},
            "time_outs": time_outs,
            "log": {
                "/tracking/track_reward": reward_terms["track_reward"].mean(),
                "/tracking/position_error_m": reward_terms["position_error_m"].mean(),
                "/tracking/yaw_error_abs_rad": reward_terms["yaw_error_abs_rad"].mean(),
                "/tracking/state_error_norm": reward_terms["state_error_norm"].mean(),
                "/tracking/action_rate": reward_terms["action_rate"].mean(),
                "/tracking/throttle_brake": reward_terms["throttle_brake"].mean(),
            },
        }
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            lengths = torch.clamp(self.episode_length_buf[done_env_ids].float(), min=1.0)
            extras["episode"] = {
                "/episode/reward": self.episode_reward_sum[done_env_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/mean_pos_error_m": (self.episode_pos_error_sum[done_env_ids] / lengths).mean(),
                "/episode/mean_track_reward": (self.episode_track_reward_sum[done_env_ids] / lengths).mean(),
            }
        return extras

    def _compute_observations(self) -> None:
        ref_state, ref_pose = self._reference_state_pose()
        current_state = self.state_hist[:, -1, :]
        history_states = self.state_hist[:, -self.obs_history_steps :, :]
        history_actions = self.action_hist[:, -self.obs_history_steps :, :]
        history_states_norm = (history_states - self.state_mean) / self.state_std
        history_actions_norm = (history_actions - self.action_mean) / self.action_std
        state_error_norm = (current_state - ref_state) / self.state_std
        pose_error = self._pose_error_local(ref_pose)
        pose_error_scaled = torch.stack(
            [
                pose_error[:, 0] / 10.0,
                pose_error[:, 1] / 10.0,
                pose_error[:, 2] / math.pi,
            ],
            dim=-1,
        )
        preview = self._reference_preview_local()
        last_actions_norm = (self.actions - self.action_mean) / self.action_std
        self.obs_buf = torch.cat(
            [
                history_states_norm.flatten(start_dim=1),
                history_actions_norm.flatten(start_dim=1),
                state_error_norm,
                pose_error_scaled,
                preview.flatten(start_dim=1),
                last_actions_norm,
            ],
            dim=-1,
        )
        self.extras = {"observations": {"critic": self.obs_buf}}

    def _reference_preview_local(self) -> torch.Tensor:
        offsets = (
            torch.arange(self.reference_preview_steps, device=self.device, dtype=torch.long)
            * self.action_repeat
        )
        preview_indices = torch.clamp(
            self.ref_step_buf[:, None] + offsets[None, :],
            max=self.reference_length - 1,
        )
        ref_ids = self.ref_ids[:, None].expand(-1, self.reference_preview_steps)
        preview_pose = self.reference_poses[ref_ids, preview_indices]
        dx_world = preview_pose[:, :, 0] - self.pose[:, None, 0]
        dy_world = preview_pose[:, :, 1] - self.pose[:, None, 1]
        cos_yaw = torch.cos(self.pose[:, None, 2])
        sin_yaw = torch.sin(self.pose[:, None, 2])
        dx_local = cos_yaw * dx_world + sin_yaw * dy_world
        dy_local = -sin_yaw * dx_world + cos_yaw * dy_world
        yaw_error = wrap_angle(preview_pose[:, :, 2] - self.pose[:, None, 2])
        return torch.stack([dx_local / 20.0, dy_local / 20.0, yaw_error / math.pi], dim=-1)

    def current_reference_state_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reference_state_pose()

    def current_pose(self) -> torch.Tensor:
        return self.pose

    def current_state(self) -> torch.Tensor:
        return self.state_hist[:, -1, :]

    def reference_names(self) -> list[str]:
        return [
            f"{family}/{episode_id}"
            for family, episode_id in zip(
                self.reference_set.scenario_families,
                self.reference_set.episode_ids,
                strict=True,
            )
        ]
