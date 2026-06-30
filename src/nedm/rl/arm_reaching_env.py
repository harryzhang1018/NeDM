"""Arm end-effector reaching environment backed by the frozen arm NN dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.rl.arm_kinematics import ArmKinematics
from nedm.rl.arm_safety import ArmSafetyFilter
from nedm.rl.defaults import (
    DEFAULT_ARM_DYNAMICS_CHECKPOINT,
    DEFAULT_ARM_GEOMETRY_PATH,
    DEFAULT_ARM_PROCESSED_DATASET_DIR,
)
from nedm.rl.dynamics import load_frozen_dynamics


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 1024,
        "device": "cuda",
        "dynamics_checkpoint": str(DEFAULT_ARM_DYNAMICS_CHECKPOINT),
        "processed_dataset_dir": str(DEFAULT_ARM_PROCESSED_DATASET_DIR),
        "geometry_path": str(DEFAULT_ARM_GEOMETRY_PATH),
        "seed_split": "train",
        "dynamics_context_steps": None,
        "action_repeat": 1,
        "max_episode_steps": 150,
        "auto_reset": True,
        "action_scale": 1.0,
        "goal": {
            "q_lo": [-1.5, 0.1, -0.4, -0.4],
            "q_hi": [1.5, 0.7, 0.4, 0.4],
            "max_sample_attempts": 16,
        },
        "observation": {
            "cartesian_scale_m": 4.0,
            "clearance_scale_m": 0.5,
            "clearance_clip_m": 1.0,
        },
        "logging": {
            "close_thresholds_m": [0.04, 0.1, 0.2, 0.5],
        },
        "safety": {
            "margins": {
                "ground_m": 0.03,
                "vehicle_m": 0.0,
                "self_m": 0.03,
                "joint_rad": 0.0,
            },
            "interpolation_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "vehicle_link_names": ["elbow", "wrist", "endoffactor", "finger_1", "finger_2"],
        },
        "reward": {
            "ee_error_scale_m": 2.0,
            "action_rate_weight": 0.01,
            "success_bonus": 75.0,
            "success_tolerance_m": 0.04,
            "success_steps": 5,
        },
    }


def merge_env_cfg(overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = default_env_cfg()
    if not overrides:
        return cfg
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged_child = dict(cfg[key])
            merged_child.update(value)
            cfg[key] = merged_child
        else:
            cfg[key] = value
    return cfg


class ArmReachingEnv(VecEnv):
    """RSL-RL VecEnv for reach-mode control inside a frozen arm dynamics model."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None) -> None:
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 4
        self.action_repeat = int(self.cfg["action_repeat"])
        self.max_episode_length = int(self.cfg["max_episode_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", True))

        self.dynamics = load_frozen_dynamics(
            checkpoint_path=self.cfg["dynamics_checkpoint"],
            device=self.device,
            processed_dataset_dir=self.cfg.get("processed_dataset_dir"),
        )
        self.model = self.dynamics.model
        self.metadata = self.dynamics.metadata
        self.dt_s = self.dynamics.dt_s
        self.context_steps = self.dynamics.context_steps
        dynamics_context_cfg = self.cfg.get("dynamics_context_steps")
        if dynamics_context_cfg is None:
            self.dynamics_context_steps = self.context_steps
        else:
            self.dynamics_context_steps = int(dynamics_context_cfg)
            if not 1 <= self.dynamics_context_steps <= self.context_steps:
                raise ValueError(
                    f"dynamics_context_steps={self.dynamics_context_steps} must be in "
                    f"[1, {self.context_steps}]"
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

        self.state_mean = self.model.state_mean.to(self.device)
        self.state_std = torch.clamp(self.model.state_std.to(self.device), min=1.0e-6)
        self.action_mean = self.model.action_mean.to(self.device)
        self.action_std = torch.clamp(self.model.action_std.to(self.device), min=1.0e-6)

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

        self.seed_states, self.seed_actions = self._load_seed_prefixes()
        self.num_seed_prefixes = int(self.seed_states.shape[0])

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
        self.extras: dict[str, Any] = {}

        self.num_obs = self._observation_dim()
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.reset()

    @property
    def unwrapped(self) -> "ArmReachingEnv":
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
                "ArmReachingEnv requires the 15-D arm state/action layout. "
                f"missing_state={missing_state}, missing_action={missing_action}"
            )
        if len(self.action_fields) != 4:
            raise ValueError(f"ArmReachingEnv expects 4 action fields, got {self.action_fields}")

    def _load_seed_prefixes(self) -> tuple[torch.Tensor, torch.Tensor]:
        processed_root = Path(self.cfg.get("processed_dataset_dir") or self.dynamics.config["processed_dataset_dir"])
        split = str(self.cfg.get("seed_split", "train"))
        states = np.load(processed_root / f"{split}_states.npy", mmap_mode="r")
        actions = np.load(processed_root / f"{split}_actions.npy", mmap_mode="r")
        starts = np.load(processed_root / f"{split}_episode_starts.npy")
        lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
        valid_episode_ids = np.nonzero(lengths >= self.context_steps)[0]
        if valid_episode_ids.size == 0:
            raise ValueError(f"No {split} seed episodes have length >= context_steps={self.context_steps}")

        seed_states = np.stack(
            [np.asarray(states[starts[i] : starts[i] + self.context_steps], dtype=np.float32) for i in valid_episode_ids],
            axis=0,
        )
        seed_actions = np.stack(
            [np.asarray(actions[starts[i] : starts[i] + self.context_steps], dtype=np.float32) for i in valid_episode_ids],
            axis=0,
        )
        seed_states_t = torch.as_tensor(seed_states, dtype=torch.float32, device=self.device)
        seed_actions_t = torch.as_tensor(seed_actions, dtype=torch.float32, device=self.device)

        q_hist = seed_states_t[:, :, self.q_indices].reshape(-1, 4)
        safe_hist = self.safety.is_safe(q_hist).reshape(seed_states_t.shape[0], self.context_steps).all(dim=1)
        if not bool(safe_hist.any().item()):
            raise ValueError(
                f"No {split} seed prefixes are safe under the configured arm safety filter "
                f"(checked {seed_states_t.shape[0]} prefixes)"
            )
        return seed_states_t[safe_hist], seed_actions_t[safe_hist]

    def _observation_dim(self) -> int:
        return 4 + 4 + 4 + 3 + 3 + 3 + 1 + 4

    def reset(self) -> tuple[torch.Tensor, dict]:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        seed_ids = torch.randint(0, self.num_seed_prefixes, (env_ids.numel(),), device=self.device)
        self.state_hist[env_ids] = self.seed_states[seed_ids]
        self.action_hist[env_ids] = self.seed_actions[seed_ids]
        self._overwrite_ee_channels(env_ids)
        self.goal_base[env_ids] = self._sample_safe_goals(env_ids.numel())

        self.actions[env_ids] = self.action_hist[env_ids, -1, :]
        self.last_actions[env_ids] = self.actions[env_ids]
        self.policy_actions[env_ids] = 0.0
        self.last_policy_actions[env_ids] = 0.0
        self.unsafe_action_buf[env_ids] = False
        self.clearance_buf[env_ids] = self.safety.clearance(self.current_q(env_ids)).float()
        self.ee_error_buf[env_ids] = torch.linalg.norm(self.current_ee_base(env_ids) - self.goal_base[env_ids], dim=-1)
        self.episode_min_ee_error[env_ids] = self.ee_error_buf[env_ids]
        self.success_count_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.reset_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_reach_reward_sum[env_ids] = 0.0
        self.episode_action_rate_penalty_sum[env_ids] = 0.0
        self.episode_success_bonus_sum[env_ids] = 0.0
        self.episode_ee_error_sum[env_ids] = 0.0
        self.episode_unsafe_sum[env_ids] = 0.0

    def _overwrite_ee_channels(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            q_hist = self.state_hist[:, :, self.q_indices]
            ee = self.kin.ee_base(q_hist.reshape(-1, 4)).reshape(self.num_envs, self.context_steps, 3)
            self.state_hist[:, :, self.ee_indices] = ee
            return
        q_hist = self.state_hist[env_ids][:, :, self.q_indices]
        ee = self.kin.ee_base(q_hist.reshape(-1, 4)).reshape(env_ids.numel(), self.context_steps, 3)
        seeded = self.state_hist[env_ids]
        seeded[:, :, self.ee_indices] = ee
        self.state_hist[env_ids] = seeded

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

        with torch.no_grad():
            for _ in range(self.action_repeat):
                safe_dq, unsafe, clearance = self._nn_substep(raw_dq)
                unsafe_any |= unsafe
                min_clearance = torch.minimum(min_clearance, clearance.float())

        self.unsafe_action_buf = unsafe_any
        self.clearance_buf = min_clearance
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

    def _nn_substep(self, raw_dq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_state = self.state_hist[:, -1, :]
        q = current_state[:, self.q_indices]
        qcmd = current_state[:, self.qcmd_indices]
        safe_dq, unsafe, clearance = self.safety.filter(q, qcmd, raw_dq)
        qcmd_next = torch.clamp(qcmd + safe_dq, self.kin.joint_limits_lo, self.kin.joint_limits_hi)

        self.action_hist[:, -1, :] = safe_dq
        k = self.dynamics_context_steps
        delta = self.model.predict_next_delta(self.state_hist[:, -k:, :], self.action_hist[:, -k:, :])
        next_state = current_state + delta
        next_state[:, self.qcmd_indices] = qcmd_next
        next_state[:, self.ee_indices] = self.kin.ee_base(next_state[:, self.q_indices])

        self.state_hist = torch.roll(self.state_hist, shifts=-1, dims=1)
        self.action_hist = torch.roll(self.action_hist, shifts=-1, dims=1)
        self.state_hist[:, -1, :] = next_state
        self.action_hist[:, -1, :] = safe_dq
        self.actions = safe_dq
        return safe_dq, unsafe, clearance

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
        current_state = self.state_hist[:, -1, :]
        q = current_state[:, self.q_indices]
        terms = self.safety.clearance_terms(q)
        geometry_collision = torch.minimum(torch.minimum(terms["ground"], terms["vehicle"]), terms["self"]) < 0.0
        joint_violation = terms["joint"] < 0.0
        success = self.success_count_buf >= int(reward_cfg["success_steps"])
        time_outs = self.episode_length_buf >= self.max_episode_length
        failed = geometry_collision | joint_violation
        dones = success | failed | time_outs
        return dones, time_outs, {
            "success_done": success.float(),
            "collision_done": geometry_collision.float(),
            "joint_limit_done": joint_violation.float(),
            "time_out": time_outs.float(),
            "ee_error_for_done_m": ee_error,
        }

    def _make_extras(
        self,
        terms: dict[str, torch.Tensor],
        dones: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> dict[str, Any]:
        log = {
            "/reach/ee_error_m": terms["ee_error_m"].mean(),
            "/reach/reach_reward": terms["reach_reward"].mean(),
            "/reach/clearance_m": terms["clearance_m"].mean(),
            "/reach/unsafe_rate": terms["unsafe_action"].mean(),
            "/reach/success_rate": terms["success"].mean(),
            "/reach/success_bonus": terms["success_bonus"].mean(),
            "/reach/action_rate": terms["action_rate"].mean(),
            "/reach/action_rate_penalty": terms["action_rate_penalty"].mean(),
            "/reach/total_reward": terms["total_reward"].mean(),
            "/reach/collision_done": terms["collision_done"].mean(),
            "/reach/joint_limit_done": terms["joint_limit_done"].mean(),
        }
        for threshold in self.close_thresholds_m:
            label = self._threshold_label(threshold)
            log[f"/reach/close_{label}_rate"] = terms[f"close_{label}"].mean()
        extras: dict[str, Any] = {
            "observations": {"critic": self.obs_buf},
            "time_outs": time_outs,
            "log": log,
        }
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            lengths = torch.clamp(self.episode_length_buf[done_env_ids].float(), min=1.0)
            episode = {
                "/episode/reward": self.episode_reward_sum[done_env_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/reach_reward_sum": self.episode_reach_reward_sum[done_env_ids].mean(),
                "/episode/mean_reach_reward": (self.episode_reach_reward_sum[done_env_ids] / lengths).mean(),
                "/episode/action_rate_penalty_sum": self.episode_action_rate_penalty_sum[done_env_ids].mean(),
                "/episode/mean_action_rate_penalty": (
                    self.episode_action_rate_penalty_sum[done_env_ids] / lengths
                ).mean(),
                "/episode/success_bonus_sum": self.episode_success_bonus_sum[done_env_ids].mean(),
                "/episode/mean_success_bonus": (self.episode_success_bonus_sum[done_env_ids] / lengths).mean(),
                "/episode/mean_ee_error_m": (self.episode_ee_error_sum[done_env_ids] / lengths).mean(),
                "/episode/final_ee_error_m": terms["ee_error_m"][done_env_ids].mean(),
                "/episode/min_ee_error_m": self.episode_min_ee_error[done_env_ids].mean(),
                "/episode/success_rate": terms["success_done"][done_env_ids].mean(),
                "/episode/collision_rate": terms["collision_done"][done_env_ids].mean(),
                "/episode/joint_limit_rate": terms["joint_limit_done"][done_env_ids].mean(),
                "/episode/timeout_rate": terms["time_out"][done_env_ids].mean(),
                "/episode/unsafe_actions": (self.episode_unsafe_sum[done_env_ids] / lengths).mean(),
            }
            for threshold in self.close_thresholds_m:
                label = self._threshold_label(threshold)
                episode[f"/episode/reached_{label}_rate"] = (
                    self.episode_min_ee_error[done_env_ids] < threshold
                ).float().mean()
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

    def current_ee_base(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.kin.ee_base(self.current_q(env_ids))


__all__ = ["ArmReachingEnv", "default_env_cfg", "merge_env_cfg"]
