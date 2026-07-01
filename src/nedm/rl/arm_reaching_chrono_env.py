"""Chrono-backed arm reaching VecEnv for policy evaluation.

This environment mirrors :class:`nedm.rl.arm_reaching_env.ArmReachingEnv` at the
policy boundary, but replaces the learned transition model with the full M113 +
LRV arm Chrono scene used by ``nedm.arm_data``. Chrono itself is not vectorized;
``num_envs`` creates independent serial simulations and is intended to stay
small for evaluation.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pychrono as chrono
import pychrono.vehicle as veh
import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.arm_data import (
    CONTROL_DT,
    STEP_SIZE,
    arm_contact,
    build_and_prepare,
    gripper_center,
    joint_limit_violation,
    _substep,
)
from nedm.rl.arm_kinematics import ArmKinematics
from nedm.rl.arm_reaching_env import default_env_cfg, merge_env_cfg
from nedm.rl.arm_safety import ArmSafetyFilter
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.training.dataset import load_metadata


@dataclass
class ChronoArmSim:
    m113: Any
    vehicle: Any
    terrain: Any
    gripper: Any
    actuator: Any
    collision_links: Any
    driver_inputs: Any
    vis: Any = None
    ee_marker: Any = None
    goal_marker: Any = None


def default_chrono_env_cfg() -> dict[str, Any]:
    cfg = default_env_cfg()
    cfg.update(
        {
            "num_envs": 1,
            "device": "cpu",
            "auto_reset": False,
            "render": False,
            "warm_start_context": True,
            "defer_reset": False,
            "pre_roll_time_s": 6.0,
            "visual_markers": {
                "enabled": True,
                "radius_m": 0.08,
            },
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


class ArmReachingChronoEnv(VecEnv):
    """Chrono M113+arm reach evaluation env with the NN env observation contract."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None) -> None:
        self.cfg = default_chrono_env_cfg()
        self.cfg = merge_env_cfg({**self.cfg, **(cfg or {})})
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 4
        self.action_repeat = int(self.cfg["action_repeat"])
        self.max_episode_length = int(self.cfg["max_episode_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", False))
        self.warm_start_context = bool(self.cfg.get("warm_start_context", True))
        self.render = bool(self.cfg.get("render", False))
        self.pre_roll_time_s = float(self.cfg.get("pre_roll_time_s", 6.0))
        marker_cfg = self.cfg.get("visual_markers", {})
        self.visual_markers_enabled = bool(marker_cfg.get("enabled", True))
        self.marker_radius_m = float(marker_cfg.get("radius_m", 0.08))

        self.metadata, self.dynamics_config, self.context_steps, self.dt_s = load_checkpoint_metadata(
            checkpoint_path=self.cfg["dynamics_checkpoint"],
            processed_dataset_dir=self.cfg.get("processed_dataset_dir"),
        )
        if abs(self.dt_s - CONTROL_DT) > 1.0e-8:
            raise ValueError(
                f"Arm Chrono env expects checkpoint dt_s={CONTROL_DT}, got {self.dt_s}. "
                "The arm_data collector currently fixes the Chrono control period."
            )
        self.chrono_steps_per_nn_step = max(1, int(round(CONTROL_DT / STEP_SIZE)))

        dynamics_context_cfg = self.cfg.get("dynamics_context_steps")
        if dynamics_context_cfg is None:
            self.dynamics_context_steps = self.context_steps
        else:
            self.dynamics_context_steps = int(dynamics_context_cfg)
            if not 1 <= self.dynamics_context_steps <= self.context_steps:
                raise ValueError(
                    f"dynamics_context_steps={self.dynamics_context_steps} must be in [1, {self.context_steps}]"
                )

        self.state_fields = list(self.metadata["state_fields"])
        self.action_fields = list(self.metadata["action_fields"])
        self.state_index = {field_name: index for index, field_name in enumerate(self.state_fields)}
        self._validate_arm_fields()
        self.q_indices = torch.tensor([self.state_index[f"q_{i}"] for i in range(4)], device=self.device)
        self.qd_indices = torch.tensor([self.state_index[f"qd_{i}"] for i in range(4)], device=self.device)
        self.qcmd_indices = torch.tensor([self.state_index[f"qcmd_{i}"] for i in range(4)], device=self.device)
        self.ee_indices = torch.tensor(
            [self.state_index["ee_base_x"], self.state_index["ee_base_y"], self.state_index["ee_base_z"]],
            device=self.device,
        )
        self.q_indices_np = np.asarray([self.state_index[f"q_{i}"] for i in range(4)], dtype=np.int64)

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

        self.kin = ArmKinematics.from_json(self.cfg["geometry_path"], device=self.device, dtype=torch.float32)
        safety_cfg = self.cfg["safety"]
        self.safety = ArmSafetyFilter(
            self.kin,
            margins=safety_cfg.get("margins"),
            interpolation_alphas=safety_cfg.get("interpolation_alphas", [0.0, 0.25, 0.5, 0.75, 1.0]),
            vehicle_link_names=safety_cfg.get("vehicle_link_names"),
        )
        self.dq_max = self.kin.dq_max.to(self.device)
        self.action_scale = float(self.cfg.get("action_scale", 1.0))
        self.goal_q_lo = torch.tensor(self.cfg["goal"]["q_lo"], dtype=torch.float32, device=self.device)
        self.goal_q_hi = torch.tensor(self.cfg["goal"]["q_hi"], dtype=torch.float32, device=self.device)
        self.max_goal_sample_attempts = int(self.cfg["goal"].get("max_sample_attempts", 16))
        close_thresholds = self.cfg.get("logging", {}).get("close_thresholds_m", [0.04, 0.1, 0.2, 0.5])
        self.close_thresholds_m = [float(value) for value in close_thresholds]

        state_dim = len(self.state_fields)
        action_dim = len(self.action_fields)
        self.state_hist = torch.zeros(self.num_envs, self.context_steps, state_dim, dtype=torch.float32, device=self.device)
        self.action_hist = torch.zeros(
            self.num_envs, self.context_steps, action_dim, dtype=torch.float32, device=self.device
        )
        self.goal_base = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.policy_actions = torch.zeros_like(self.actions)
        self.last_policy_actions = torch.zeros_like(self.actions)
        self.unsafe_action_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.clearance_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ee_error_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.success_count_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_force_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.joint_limit_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nonfinite_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_kinds: list[str | None] = [None for _ in range(self.num_envs)]
        self.contact_links: list[list[str]] = [[] for _ in range(self.num_envs)]
        self.joint_limit_labels: list[list[str]] = [[] for _ in range(self.num_envs)]
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_reach_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_action_rate_penalty_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_success_bonus_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_ee_error_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_min_ee_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_unsafe_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.sims: list[ChronoArmSim | None] = [None for _ in range(self.num_envs)]
        self.extras: dict[str, Any] = {}

        self.num_obs = self._observation_dim()
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        if not bool(self.cfg.get("defer_reset", False)):
            self.reset()

    @property
    def unwrapped(self) -> "ArmReachingChronoEnv":
        return self

    def _validate_arm_fields(self) -> None:
        required_state = (
            [f"q_{i}" for i in range(4)]
            + [f"qd_{i}" for i in range(4)]
            + [f"qcmd_{i}" for i in range(4)]
            + ["ee_base_x", "ee_base_y", "ee_base_z"]
        )
        required_action = [f"act_{i}" for i in range(4)]
        missing_state = [field for field in required_state if field not in self.state_index]
        missing_action = [field for field in required_action if field not in self.action_fields]
        if missing_state or missing_action:
            raise ValueError(
                "ArmReachingChronoEnv requires the 15-D arm state/action layout. "
                f"missing_state={missing_state}, missing_action={missing_action}"
            )
        if len(self.action_fields) != 4:
            raise ValueError(f"ArmReachingChronoEnv expects 4 action fields, got {self.action_fields}")

    def _observation_dim(self) -> int:
        return 4 + 4 + 4 + 3 + 3 + 3 + 1 + 4

    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: torch.Tensor, goal_base: torch.Tensor | np.ndarray | None = None) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if goal_base is None:
            goals = self._sample_safe_goals(env_ids.numel())
        else:
            goals = torch.as_tensor(goal_base, dtype=torch.float32, device=self.device)
            if goals.dim() == 1:
                goals = goals.view(1, 3).repeat(env_ids.numel(), 1)
            if goals.shape != (env_ids.numel(), 3):
                raise ValueError(f"goal_base must have shape ({env_ids.numel()}, 3) or (3,), got {tuple(goals.shape)}")

        for local_index, env_id_t in enumerate(env_ids):
            env_index = int(env_id_t.item())
            self._destroy_sim(env_index)
            sim = self._create_sim(render=self.render and env_index == 0)
            self.sims[env_index] = sim
            self.goal_base[env_index] = goals[local_index]
            self._update_goal_marker(env_index)
            self._initialize_history(env_index, sim)

            self._update_goal_marker(env_index)
            self.actions[env_index] = self.action_hist[env_index, -1, :]
            self.last_actions[env_index] = self.actions[env_index]
            self.policy_actions[env_index] = 0.0
            self.last_policy_actions[env_index] = 0.0
            self._reset_episode_buffers(env_index)

    def reset_goal_idx(self, env_ids: torch.Tensor, goal_base: torch.Tensor | np.ndarray | None = None) -> None:
        """Assign a new target while keeping the current Chrono scene/state.

        This is the consecutive-goal reset path: the arm remains wherever Chrono
        ended the previous segment, while episode statistics and success counters
        restart for the new goal.
        """
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        goals = self._resolve_goals(env_ids.numel(), goal_base)
        for local_index, env_id_t in enumerate(env_ids):
            env_index = int(env_id_t.item())
            if self.sims[env_index] is None:
                raise RuntimeError(f"Chrono simulation {env_index} is not initialized")
            self.goal_base[env_index] = goals[local_index]
            self._update_goal_marker(env_index)
            self.last_actions[env_index] = self.actions[env_index]
            self.last_policy_actions[env_index] = self.policy_actions[env_index]
            self._reset_episode_buffers(env_index)
        self._compute_observations()

    def _resolve_goals(self, count: int, goal_base: torch.Tensor | np.ndarray | None) -> torch.Tensor:
        if goal_base is None:
            return self._sample_safe_goals(count)
        goals = torch.as_tensor(goal_base, dtype=torch.float32, device=self.device)
        if goals.dim() == 1:
            goals = goals.view(1, 3).repeat(count, 1)
        if goals.shape != (count, 3):
            raise ValueError(f"goal_base must have shape ({count}, 3) or (3,), got {tuple(goals.shape)}")
        return goals

    def _reset_episode_buffers(self, env_index: int) -> None:
        env_id = torch.tensor([env_index], dtype=torch.long, device=self.device)
        self.unsafe_action_buf[env_index] = False
        self.clearance_buf[env_index] = self.safety.clearance(self.current_q(env_id))[0]
        self.ee_error_buf[env_index] = torch.linalg.norm(
            self.current_ee_base(env_id)[0] - self.goal_base[env_index],
            dim=-1,
        )
        self.episode_min_ee_error[env_index] = self.ee_error_buf[env_index]
        self.success_count_buf[env_index] = 0
        self.contact_buf[env_index] = False
        self.contact_force_buf[env_index] = 0.0
        self.joint_limit_buf[env_index] = False
        self.nonfinite_buf[env_index] = False
        self.contact_kinds[env_index] = None
        self.contact_links[env_index] = []
        self.joint_limit_labels[env_index] = []
        self.episode_length_buf[env_index] = 0
        self.rew_buf[env_index] = 0.0
        self.reset_buf[env_index] = 0
        self.time_out_buf[env_index] = False
        self.episode_reward_sum[env_index] = 0.0
        self.episode_reach_reward_sum[env_index] = 0.0
        self.episode_action_rate_penalty_sum[env_index] = 0.0
        self.episode_success_bonus_sum[env_index] = 0.0
        self.episode_ee_error_sum[env_index] = 0.0
        self.episode_unsafe_sum[env_index] = 0.0

    def _destroy_sim(self, env_index: int) -> None:
        if self.sims[env_index] is not None:
            self.sims[env_index] = None
            gc.collect()

    def _create_sim(self, render: bool = False) -> ChronoArmSim:
        m113, vehicle, terrain, gripper, actuator, collision_links, vis = build_and_prepare(
            render=render,
            settle_time_s=self.pre_roll_time_s,
        )
        driver_inputs = veh.DriverInputs()
        driver_inputs.m_throttle = 0.0
        driver_inputs.m_steering = 0.0
        driver_inputs.m_braking = 1.0
        ee_marker = None
        goal_marker = None
        if self.visual_markers_enabled:
            system = m113.GetSystem()
            ee_marker = self._create_marker_body(
                system,
                radius_m=self.marker_radius_m,
                color=chrono.ChColor(1.0, 0.55, 0.05),
            )
            goal_marker = self._create_marker_body(
                system,
                radius_m=self.marker_radius_m,
                color=chrono.ChColor(1.0, 0.05, 0.05),
            )
            ee_world = gripper_center(gripper)
            ee_marker.SetPos(ee_world)
            goal_marker.SetPos(ee_world)
            self._bind_marker_visuals(vis, system)
        return ChronoArmSim(
            m113=m113,
            vehicle=vehicle,
            terrain=terrain,
            gripper=gripper,
            actuator=actuator,
            collision_links=collision_links,
            driver_inputs=driver_inputs,
            vis=vis,
            ee_marker=ee_marker,
            goal_marker=goal_marker,
        )

    @staticmethod
    def _create_marker_body(system: Any, radius_m: float, color: Any) -> Any:
        body = chrono.ChBody()
        body.SetFixed(True)
        body.EnableCollision(False)
        shape = chrono.ChVisualShapeSphere(float(radius_m))
        shape.SetColor(color)
        body.AddVisualShape(shape)
        system.Add(body)
        return body

    @staticmethod
    def _bind_marker_visuals(vis: Any, system: Any) -> None:
        if vis is None:
            return
        # The tracked-vehicle Irrlicht wrapper is attached to the vehicle during
        # build_and_prepare(). These marker bodies are plain ChBody instances in
        # the same ChSystem, so explicitly attach/bind the system after adding
        # their visual shapes.
        for method_name, args in (
            ("AttachSystem", (system,)),
            ("BindAll", ()),
        ):
            method = getattr(vis, method_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception:
                pass

    @staticmethod
    def _base_to_world(sim: ChronoArmSim, point_base: torch.Tensor | np.ndarray) -> Any:
        if isinstance(point_base, torch.Tensor):
            values = point_base.detach().cpu().numpy()
        else:
            values = np.asarray(point_base, dtype=np.float32)
        return sim.gripper.base.GetFrameRefToAbs().TransformPointLocalToParent(
            chrono.ChVector3d(float(values[0]), float(values[1]), float(values[2]))
        )

    def _update_goal_marker(self, env_index: int) -> None:
        sim = self.sims[env_index]
        if sim is None or sim.goal_marker is None:
            return
        sim.goal_marker.SetPos(self._base_to_world(sim, self.goal_base[env_index]))

    @staticmethod
    def _update_ee_marker(sim: ChronoArmSim, ee_world: Any) -> None:
        if sim.ee_marker is not None:
            sim.ee_marker.SetPos(ee_world)

    def _initialize_history(self, env_index: int, sim: ChronoArmSim) -> None:
        if self.warm_start_context:
            for hist_index in range(self.context_steps):
                self.state_hist[env_index, hist_index] = torch.as_tensor(
                    self._capture_state_np(sim),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.action_hist[env_index, hist_index] = 0.0
                if hist_index < self.context_steps - 1:
                    _substep(
                        sim.m113,
                        sim.terrain,
                        sim.actuator,
                        sim.driver_inputs,
                        self.chrono_steps_per_nn_step,
                        vis=sim.vis,
                    )
            return

        state = torch.as_tensor(self._capture_state_np(sim), dtype=torch.float32, device=self.device)
        self.state_hist[env_index] = state.view(1, -1).repeat(self.context_steps, 1)
        self.action_hist[env_index] = 0.0

    def _capture_state_np(self, sim: ChronoArmSim) -> np.ndarray:
        q, qd = sim.actuator.read_state()
        qcmd = list(sim.actuator.qcmd)
        ee_world = gripper_center(sim.gripper)
        self._update_ee_marker(sim, ee_world)
        ee_base = sim.gripper.base.GetFrameRefToAbs().TransformPointParentToLocal(ee_world)
        values: dict[str, float] = {
            **{f"q_{i}": float(q[i]) for i in range(4)},
            **{f"qd_{i}": float(qd[i]) for i in range(4)},
            **{f"qcmd_{i}": float(qcmd[i]) for i in range(4)},
            "ee_base_x": float(ee_base.x),
            "ee_base_y": float(ee_base.y),
            "ee_base_z": float(ee_base.z),
        }
        return np.asarray([values[field] for field in self.state_fields], dtype=np.float32)

    def _sample_safe_goals(self, count: int) -> torch.Tensor:
        goals = torch.empty(count, 3, dtype=torch.float32, device=self.device)
        pending = torch.ones(count, dtype=torch.bool, device=self.device)
        for _ in range(self.max_goal_sample_attempts):
            pending_ids = pending.nonzero(as_tuple=False).flatten()
            if pending_ids.numel() == 0:
                break
            q = self.goal_q_lo + torch.rand(pending_ids.numel(), 4, device=self.device) * (
                self.goal_q_hi - self.goal_q_lo
            )
            safe = self.safety.is_safe(q)
            if safe.any():
                accepted = pending_ids[safe]
                goals[accepted] = self.kin.ee_base(q[safe])
                pending[accepted] = False
        if pending.any():
            fallback = self.kin.ee_base(self.kin.q_home.to(self.device).view(1, 4)).float()[0]
            goals[pending] = fallback
        return goals

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.last_actions = self.actions.clone()
        self.last_policy_actions = self.policy_actions.clone()
        self.policy_actions = actions.to(device=self.device, dtype=torch.float32)
        bounded = torch.tanh(self.policy_actions)
        raw_dq = bounded * (self.dq_max.view(1, 4) * self.action_scale)

        unsafe_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        min_clearance = torch.full((self.num_envs,), float("inf"), dtype=torch.float32, device=self.device)
        contact_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        contact_force = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        joint_limit_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.contact_kinds = [None for _ in range(self.num_envs)]
        self.contact_links = [[] for _ in range(self.num_envs)]
        self.joint_limit_labels = [[] for _ in range(self.num_envs)]

        with torch.no_grad():
            for _ in range(self.action_repeat):
                current_state = self.state_hist[:, -1, :]
                q = current_state[:, self.q_indices]
                qcmd = current_state[:, self.qcmd_indices]
                safe_dq, unsafe, clearance = self.safety.filter(q, qcmd, raw_dq)
                qcmd_next = torch.clamp(qcmd + safe_dq, self.kin.joint_limits_lo, self.kin.joint_limits_hi)
                unsafe_any |= unsafe
                min_clearance = torch.minimum(min_clearance, clearance.float())

                safe_dq_cpu = safe_dq.detach().cpu().numpy()
                qcmd_next_cpu = qcmd_next.detach().cpu().numpy()
                for env_index in range(self.num_envs):
                    sim = self.sims[env_index]
                    if sim is None:
                        raise RuntimeError(f"Chrono simulation {env_index} is not initialized")
                    sim.actuator.qcmd = qcmd_next_cpu[env_index].astype(float).tolist()
                    _substep(
                        sim.m113,
                        sim.terrain,
                        sim.actuator,
                        sim.driver_inputs,
                        self.chrono_steps_per_nn_step,
                        vis=sim.vis,
                    )

                    state_np = self._capture_state_np(sim)
                    self.state_hist[env_index] = torch.roll(self.state_hist[env_index], shifts=-1, dims=0)
                    self.action_hist[env_index] = torch.roll(self.action_hist[env_index], shifts=-1, dims=0)
                    self.state_hist[env_index, -1] = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
                    self.action_hist[env_index, -1] = torch.as_tensor(
                        safe_dq_cpu[env_index],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.actions[env_index] = self.action_hist[env_index, -1]

                    hit, kind, involved, force = arm_contact(sim.collision_links)
                    q_next = state_np[self.q_indices_np].tolist()
                    limit_hit, limit_labels = joint_limit_violation(q_next)
                    contact_any[env_index] |= bool(hit)
                    contact_force[env_index] = max(float(contact_force[env_index].item()), float(force))
                    joint_limit_any[env_index] |= bool(limit_hit)
                    if hit:
                        self.contact_kinds[env_index] = kind
                        self.contact_links[env_index] = list(involved)
                    if limit_hit:
                        self.joint_limit_labels[env_index] = list(limit_labels)

        self.unsafe_action_buf = unsafe_any
        self.clearance_buf = min_clearance
        self.contact_buf = contact_any
        self.contact_force_buf = contact_force
        self.joint_limit_buf = joint_limit_any
        self.nonfinite_buf = ~torch.isfinite(self.state_hist[:, -1, :]).all(dim=-1)
        self.episode_length_buf += 1

        reward, reward_terms = self._compute_reward()
        dones, time_outs, done_terms = self._compute_dones(reward_terms["ee_error_m"])
        self.rew_buf = reward
        self.reset_buf = dones.long()
        self.time_out_buf = time_outs
        self.episode_reward_sum += reward
        self.episode_reach_reward_sum += reward_terms["reach_reward"]
        self.episode_action_rate_penalty_sum += reward_terms["action_rate_penalty"]
        self.episode_success_bonus_sum += reward_terms["success_bonus"]
        self.episode_ee_error_sum += reward_terms["ee_error_m"]
        self.episode_min_ee_error = torch.minimum(self.episode_min_ee_error, reward_terms["ee_error_m"])
        self.episode_unsafe_sum += unsafe_any.float()

        extras = self._make_extras(reward_terms | done_terms, dones, time_outs)
        return_rewards = reward.clone()
        return_dones = dones.long().clone()
        if self.auto_reset:
            done_env_ids = dones.nonzero(as_tuple=False).flatten()
            if done_env_ids.numel() > 0:
                self.reset_idx(done_env_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, return_rewards, return_dones, self.extras

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_cfg = self.cfg["reward"]
        ee = self.current_ee_base()
        error_vec = self.goal_base - ee
        ee_error = torch.linalg.norm(error_vec, dim=-1)
        self.ee_error_buf = ee_error
        reached = ee_error < float(reward_cfg["success_tolerance_m"])
        self.success_count_buf = torch.where(reached, self.success_count_buf + 1, torch.zeros_like(self.success_count_buf))
        success = self.success_count_buf >= int(reward_cfg["success_steps"])

        action_norm = self.actions / torch.clamp(self.dq_max.view(1, 4), min=1.0e-6)
        last_action_norm = self.last_actions / torch.clamp(self.dq_max.view(1, 4), min=1.0e-6)
        action_rate = torch.sum(torch.square(action_norm - last_action_norm), dim=-1)
        reach_reward = torch.exp(-ee_error / float(reward_cfg["ee_error_scale_m"]))
        action_rate_penalty = -float(reward_cfg["action_rate_weight"]) * action_rate
        success_bonus = float(reward_cfg["success_bonus"]) * success.float()

        reward = reach_reward + action_rate_penalty + success_bonus
        return reward, {
            "ee_error_m": ee_error,
            "action_rate": action_rate,
            "action_rate_penalty": action_rate_penalty,
            "reach_reward": reach_reward,
            "success_bonus": success_bonus,
            "total_reward": reward,
            "reached": reached.float(),
            "success": success.float(),
            **{
                f"close_{self._threshold_label(threshold)}": (ee_error < threshold).float()
                for threshold in self.close_thresholds_m
            },
            "unsafe_action": self.unsafe_action_buf.float(),
            "clearance_m": self.clearance_buf,
        }

    def _compute_dones(self, ee_error: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        reward_cfg = self.cfg["reward"]
        success = self.success_count_buf >= int(reward_cfg["success_steps"])
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones = success | self.contact_buf | self.joint_limit_buf | self.nonfinite_buf | time_outs
        return dones, time_outs, {
            "success_done": success.float(),
            "collision_done": self.contact_buf.float(),
            "joint_limit_done": self.joint_limit_buf.float(),
            "nonfinite_done": self.nonfinite_buf.float(),
            "time_out": time_outs.float(),
            "contact_force_n": self.contact_force_buf,
            "ee_error_for_done_m": ee_error,
        }

    def _make_extras(
        self,
        terms: dict[str, torch.Tensor],
        dones: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> dict[str, Any]:
        log = {
            "/chrono_reach/ee_error_m": terms["ee_error_m"].mean(),
            "/chrono_reach/clearance_m": terms["clearance_m"].mean(),
            "/chrono_reach/unsafe_rate": terms["unsafe_action"].mean(),
            "/chrono_reach/reached_rate": terms["reached"].mean(),
            "/chrono_reach/success_rate": terms["success"].mean(),
            "/chrono_reach/action_rate": terms["action_rate"].mean(),
            "/chrono_reach/nonfinite_done": terms["nonfinite_done"].mean(),
            "/chrono_reach/contact_force_n": terms["contact_force_n"].mean(),
        }
        extras: dict[str, Any] = {
            "observations": {"critic": self.obs_buf},
            "time_outs": time_outs,
            "log": log,
            "chrono_contact_kind": list(self.contact_kinds),
            "chrono_contact_links": [list(links) for links in self.contact_links],
            "chrono_joint_limit_labels": [list(labels) for labels in self.joint_limit_labels],
        }
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            lengths = torch.clamp(self.episode_length_buf[done_env_ids].float(), min=1.0)
            episode = {
                "/episode/reward": self.episode_reward_sum[done_env_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/mean_reach_reward": (self.episode_reach_reward_sum[done_env_ids] / lengths).mean(),
                "/episode/mean_action_rate_penalty": (
                    self.episode_action_rate_penalty_sum[done_env_ids] / lengths
                ).mean(),
                "/episode/success_bonus_sum": self.episode_success_bonus_sum[done_env_ids].mean(),
                "/episode/mean_ee_error_m": (self.episode_ee_error_sum[done_env_ids] / lengths).mean(),
                "/episode/final_ee_error_m": terms["ee_error_m"][done_env_ids].mean(),
                "/episode/min_ee_error_m": self.episode_min_ee_error[done_env_ids].mean(),
                "/episode/success_rate": terms["success_done"][done_env_ids].mean(),
                "/episode/collision_rate": terms["collision_done"][done_env_ids].mean(),
                "/episode/joint_limit_rate": terms["joint_limit_done"][done_env_ids].mean(),
                "/episode/timeout_rate": terms["time_out"][done_env_ids].mean(),
                "/episode/unsafe_actions": (self.episode_unsafe_sum[done_env_ids] / lengths).mean(),
            }
            episode.update(log)
            extras["episode"] = episode
        return extras

    @staticmethod
    def _threshold_label(threshold_m: float) -> str:
        centimeters = threshold_m * 100.0
        if abs(centimeters - round(centimeters)) < 1.0e-6:
            return f"{int(round(centimeters))}cm"
        return f"{centimeters:.1f}cm".replace(".", "p")

    def _compute_observations(self) -> None:
        current_state = self.state_hist[:, -1, :]
        q = current_state[:, self.q_indices]
        qd = current_state[:, self.qd_indices]
        qcmd = current_state[:, self.qcmd_indices]
        ee = self.current_ee_base()
        error = self.goal_base - ee

        q_norm = (q - self.state_mean[self.q_indices]) / self.state_std[self.q_indices]
        qd_norm = (qd - self.state_mean[self.qd_indices]) / self.state_std[self.qd_indices]
        qcmd_norm = (qcmd - self.state_mean[self.qcmd_indices]) / self.state_std[self.qcmd_indices]
        obs_cfg = self.cfg["observation"]
        cart_scale = float(obs_cfg["cartesian_scale_m"])
        clearance = torch.clamp(
            self.safety.clearance(q),
            min=-float(obs_cfg["clearance_clip_m"]),
            max=float(obs_cfg["clearance_clip_m"]),
        ).unsqueeze(-1) / float(obs_cfg["clearance_scale_m"])
        last_action_norm = self.actions / torch.clamp(self.dq_max.view(1, 4), min=1.0e-6)

        self.obs_buf = torch.cat(
            [
                q_norm,
                qd_norm,
                qcmd_norm,
                self.goal_base / cart_scale,
                ee / cart_scale,
                error / cart_scale,
                clearance,
                last_action_norm,
            ],
            dim=-1,
        )
        self.extras = {"observations": {"critic": self.obs_buf}}

    def current_state(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            return self.state_hist[:, -1, :]
        return self.state_hist[env_ids, -1, :]

    def current_q(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.current_state(env_ids)[:, self.q_indices]

    def current_qcmd(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.current_state(env_ids)[:, self.qcmd_indices]

    def current_ee_base(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.current_state(env_ids)[:, self.ee_indices]


__all__ = [
    "ArmReachingChronoEnv",
    "ChronoArmSim",
    "default_chrono_env_cfg",
    "load_checkpoint_metadata",
]
