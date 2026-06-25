from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv

from nedm.rl.defaults import DEFAULT_RL_DYNAMICS_CHECKPOINT, DEFAULT_RL_REFERENCE_PATH
from nedm.rl.dynamics import FrozenDynamics, load_frozen_dynamics
from nedm.rl.references import ReferenceSet, load_reference_set


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 1024,
        "device": "cuda",
        "dynamics_checkpoint": str(DEFAULT_RL_DYNAMICS_CHECKPOINT),
        "processed_dataset_dir": None,
        "reference_path": str(DEFAULT_RL_REFERENCE_PATH),
        # Terrain-conditioned dynamics checkpoints need a per-env terrain id.
        # None means infer it from reference metadata: single-domain references
        # make every env use that terrain; multi-domain references are split
        # evenly across available terrains.
        "terrain": None,
        "terrain_mix": None,
        # Optional override for old reference files that do not carry metadata
        # domains. Either one terrain for every ref, or one entry per reference.
        "reference_terrain_domains": None,
        "dynamics_context_steps": None,
        "action_repeat": 5,
        "obs_history_steps": 10,
        "reference_preview_steps": 10,
        "max_episode_steps": 180,
        "auto_reset": True,
        "action_low": [-1.0, 0.0, 0.0],
        "action_high": [1.0, 1.0, 1.0],
        "action_center": "dataset_mean",
        "action_scale": [1.0, 0.7, 0.5],
        "steering_rate_limit": None,
        "reward": {
            "position_sigma_m": 2.0,
            "yaw_sigma_rad": 0.35,
            "state_sigma": 1.0,
            "state_error_fields": None,
            "position_weight": 1.0,
            "yaw_weight": 0.8,
            "state_weight": 0.2,
            "action_rate_weight": 0.02,
            "throttle_brake_weight": 0.05,
        },
        "termination": {
            "max_position_error_m": 20.0,
            "max_abs_roll_rad": 0.6,
            "max_abs_pitch_rad": 0.4,
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


class HMMWVNeuralTrackingEnv(VecEnv):
    """RSL-RL vectorized env backed by a frozen HMMWV NN dynamics model."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 3
        self.action_repeat = int(self.cfg["action_repeat"])
        self.obs_history_steps = int(self.cfg["obs_history_steps"])
        self.reference_preview_steps = int(self.cfg["reference_preview_steps"])
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
        self.step_dt = self.dt_s * self.action_repeat
        self.context_steps = self.dynamics.context_steps
        self.num_terrains = int(getattr(self.model, "num_terrains", 0))
        self.terrain_names = self._terrain_names_from_metadata()
        self.terrain_to_id = {name: index for index, name in enumerate(self.terrain_names)}
        self._terrain_lower_to_name = {name.lower(): name for name in self.terrain_names}
        if self.obs_history_steps > self.context_steps:
            raise ValueError(
                f"obs_history_steps={self.obs_history_steps} exceeds dynamics context {self.context_steps}"
            )
        # Number of trailing history tokens actually fed to the dynamics model each
        # substep. The model is trained with block_size=context_steps, but the
        # dynamics is near-Markovian, so a short window gives the same rollout
        # accuracy at a fraction of the transformer compute. None => full context.
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
        self.reference_set = load_reference_set(self.cfg["reference_path"])
        self._validate_reference_set(self.reference_set, self.dynamics)

        self.reference_states = torch.as_tensor(self.reference_set.states, dtype=torch.float32, device=self.device)
        self.reference_actions = torch.as_tensor(self.reference_set.actions, dtype=torch.float32, device=self.device)
        self.reference_poses = torch.as_tensor(self.reference_set.poses, dtype=torch.float32, device=self.device)
        self.num_references = int(self.reference_states.shape[0])
        self.reference_length = int(self.reference_states.shape[1])
        self.reference_terrain_ids = self._resolve_reference_terrain_ids()
        self.env_terrain_ids = self._build_env_terrain_ids()
        self.reference_ids_by_terrain = self._build_reference_ids_by_terrain()
        min_required_length = self.context_steps + self.action_repeat + 1
        if self.reference_length < min_required_length:
            raise ValueError(
                f"Reference length {self.reference_length} is too short for context={self.context_steps} "
                f"and action_repeat={self.action_repeat}"
            )

        max_policy_steps_from_refs = max(1, (self.reference_length - self.context_steps - 1) // self.action_repeat)
        if self.max_episode_length > max_policy_steps_from_refs:
            self.max_episode_length = max_policy_steps_from_refs

        self.state_mean = self.model.state_mean.to(self.device)
        self.state_std = torch.clamp(self.model.state_std.to(self.device), min=1.0e-6)
        self.action_mean = self.model.action_mean.to(self.device)
        self.action_std = torch.clamp(self.model.action_std.to(self.device), min=1.0e-6)
        self.action_low = torch.tensor(self.cfg["action_low"], dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.cfg["action_high"], dtype=torch.float32, device=self.device)
        action_center_cfg = self.cfg.get("action_center", "dataset_mean")
        if action_center_cfg == "dataset_mean":
            self.action_center = self.action_mean.clone()
        else:
            self.action_center = torch.tensor(action_center_cfg, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(self.cfg["action_scale"], dtype=torch.float32, device=self.device)

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
        self.extras: dict[str, Any] = {}

        self.num_obs = self._observation_dim()
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=self.device)
        self.reset()

    @property
    def unwrapped(self) -> "HMMWVNeuralTrackingEnv":
        return self

    def _validate_reference_set(self, reference_set: ReferenceSet, dynamics: FrozenDynamics) -> None:
        if abs(reference_set.dt_s - dynamics.dt_s) > 1.0e-8:
            raise ValueError(f"Reference dt_s={reference_set.dt_s} does not match model dt_s={dynamics.dt_s}")
        dynamics_state_fields = list(dynamics.metadata["state_fields"])
        dynamics_action_fields = list(dynamics.metadata["action_fields"])
        if reference_set.state_fields != dynamics_state_fields:
            raise ValueError(
                "Reference state fields do not match dynamics checkpoint metadata "
                f"({len(reference_set.state_fields)} reference fields vs "
                f"{len(dynamics_state_fields)} checkpoint fields). Rebuild the RL reference set "
                "from the processed dataset used by the dynamics checkpoint."
            )
        if reference_set.action_fields != dynamics_action_fields:
            raise ValueError(
                "Reference action fields do not match dynamics checkpoint metadata "
                f"({len(reference_set.action_fields)} reference fields vs "
                f"{len(dynamics_action_fields)} checkpoint fields)."
            )

    def _terrain_names_from_metadata(self) -> list[str]:
        if self.num_terrains <= 0:
            return []
        terrain_cfg = self.metadata.get("terrain_conditioning") or {}
        terrain_names = [str(name) for name in terrain_cfg.get("terrains", [])]
        if not terrain_names:
            terrain_names = [str(index) for index in range(self.num_terrains)]
        if len(terrain_names) != self.num_terrains:
            raise ValueError(
                "Checkpoint terrain_conditioning metadata has "
                f"{len(terrain_names)} terrain names for num_terrains={self.num_terrains}"
            )
        return terrain_names

    def _terrain_id_from_value(self, value: Any) -> int:
        if self.num_terrains <= 0:
            raise ValueError("terrain was supplied, but the dynamics checkpoint is not terrain-conditioned")
        if isinstance(value, bool):
            raise ValueError("terrain id must not be a bool")
        if isinstance(value, int):
            terrain_id = int(value)
            if 0 <= terrain_id < self.num_terrains:
                return terrain_id
            raise ValueError(f"terrain id {terrain_id} is outside [0, {self.num_terrains})")
        if isinstance(value, (list, tuple)):
            if len(value) != self.num_terrains:
                raise ValueError(
                    f"one-hot terrain key has length {len(value)}, expected {self.num_terrains}"
                )
            values = [float(item) for item in value]
            terrain_id = int(max(range(len(values)), key=lambda index: values[index]))
            if values[terrain_id] <= 0.0:
                raise ValueError(f"one-hot terrain key has no positive entry: {value!r}")
            return terrain_id
        name = str(value).strip()
        if name.isdigit():
            return self._terrain_id_from_value(int(name))
        lower_name = name.lower()
        if lower_name in self._terrain_lower_to_name:
            return self.terrain_to_id[self._terrain_lower_to_name[lower_name]]
        aliases = {
            "rigid": "flat",
            "rigid_terrain": "flat",
            "flat_terrain": "flat",
            "deformable": "crm",
            "soft": "crm",
            "soil": "crm",
        }
        alias = aliases.get(lower_name)
        if alias is not None and alias in self._terrain_lower_to_name:
            return self.terrain_to_id[self._terrain_lower_to_name[alias]]
        if lower_name == "flat" and "rigid" in self._terrain_lower_to_name:
            return self.terrain_to_id[self._terrain_lower_to_name["rigid"]]
        raise ValueError(
            f"Unknown terrain {value!r}; expected one of {self.terrain_names} "
            "(aliases: rigid->flat, deformable/soft/soil->crm)."
        )

    def _looks_like_one_hot_key(self, value: tuple[Any, ...] | list[Any]) -> bool:
        if len(value) != self.num_terrains:
            return False
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return False
        values = [float(item) for item in value]
        return sum(item > 0.0 for item in values) == 1

    def _terrain_tensor_from_value(self, value: Any, *, expected_length: int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.long)
            if tensor.dim() == 0:
                tensor = tensor.view(1).expand(expected_length)
            if tensor.numel() != expected_length:
                raise ValueError(f"terrain_ids has {tensor.numel()} entries; expected {expected_length}")
            return tensor.reshape(expected_length)
        if isinstance(value, (list, tuple)) and self._looks_like_one_hot_key(value):
            ids = [self._terrain_id_from_value(value)] * expected_length
        elif isinstance(value, (list, tuple)) and len(value) == expected_length and all(
            isinstance(item, str) for item in value
        ):
            ids = [self._terrain_id_from_value(item) for item in value]
        elif isinstance(value, (list, tuple)) and len(value) == expected_length and all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        ):
            ids = [self._terrain_id_from_value(item) for item in value]
        else:
            ids = [self._terrain_id_from_value(value)] * expected_length
        if len(ids) != expected_length:
            raise ValueError(f"terrain value produced {len(ids)} entries; expected {expected_length}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _expand_reference_domain_cfg(self, domains_cfg: Any) -> list[Any] | None:
        if domains_cfg is None:
            return None
        if isinstance(domains_cfg, str):
            parts = [part.strip() for part in domains_cfg.split(",") if part.strip()]
            if len(parts) == 1:
                return parts * self.num_references
            if len(parts) == self.num_references:
                return parts
            raise ValueError(
                "reference_terrain_domains must be one terrain name or one per reference; "
                f"got {len(parts)} entries for {self.num_references} references"
            )
        if isinstance(domains_cfg, (list, tuple)):
            if len(domains_cfg) == 1:
                return list(domains_cfg) * self.num_references
            if len(domains_cfg) == self.num_references:
                return list(domains_cfg)
            raise ValueError(
                "reference_terrain_domains must have length 1 or num_references; "
                f"got {len(domains_cfg)} for {self.num_references} references"
            )
        return [domains_cfg] * self.num_references

    def _infer_reference_domain_names(self) -> list[Any] | None:
        cfg_domains = self._expand_reference_domain_cfg(self.cfg.get("reference_terrain_domains"))
        if cfg_domains is not None:
            return cfg_domains

        metadata = self.reference_set.metadata
        for key in ("domains", "terrain_domains", "terrains"):
            domains = metadata.get(key)
            if isinstance(domains, list):
                if len(domains) != self.num_references:
                    raise ValueError(
                        f"Reference metadata {key!r} has {len(domains)} entries for "
                        f"{self.num_references} references"
                    )
                return list(domains)

        for key in ("domain", "terrain"):
            if key in metadata:
                return [metadata[key]] * self.num_references

        source_tokens = " ".join(
            str(metadata.get(key, ""))
            for key in ("source", "source_processed_root", "crm_dataset_dir")
        )
        source_tokens = f"{source_tokens} {self.cfg['reference_path']}".lower()
        if "crm" in source_tokens:
            return ["crm"] * self.num_references
        if "rigid" in source_tokens or "flat" in source_tokens:
            return ["flat"] * self.num_references
        return None

    def _resolve_reference_terrain_ids(self) -> torch.Tensor | None:
        if self.num_terrains <= 0:
            return None
        domain_names = self._infer_reference_domain_names()
        if domain_names is None:
            single_terrain = self.cfg.get("terrain")
            if single_terrain is not None and self.cfg.get("terrain_mix") is None:
                terrain_id = self._terrain_id_from_value(single_terrain)
                return torch.full(
                    (self.num_references,),
                    terrain_id,
                    dtype=torch.long,
                    device=self.device,
                )
            raise ValueError(
                "The dynamics checkpoint is terrain-conditioned, but the reference set does not "
                "identify each reference terrain. Use a reference set with metadata['domains'] "
                "or set reference_terrain_domains/--reference-terrain-domains."
            )
        ids = [self._terrain_id_from_value(domain_name) for domain_name in domain_names]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _parse_terrain_mix(self, mix_cfg: Any) -> list[tuple[int, float]]:
        if mix_cfg is None:
            return []
        items: list[tuple[Any, Any]] = []
        if isinstance(mix_cfg, str):
            for raw_part in mix_cfg.split(","):
                part = raw_part.strip()
                if not part:
                    continue
                if ":" in part:
                    terrain_name, raw_weight = part.split(":", 1)
                    items.append((terrain_name.strip(), float(raw_weight)))
                else:
                    items.append((part, 1.0))
        elif isinstance(mix_cfg, dict):
            items = list(mix_cfg.items())
        elif isinstance(mix_cfg, (list, tuple)):
            for item in mix_cfg:
                if isinstance(item, dict):
                    terrain_name = item.get("terrain", item.get("name"))
                    if terrain_name is None:
                        raise ValueError(f"terrain_mix item is missing terrain/name: {item!r}")
                    weight = item.get("fraction", item.get("weight", item.get("count", 1.0)))
                    items.append((terrain_name, weight))
                else:
                    items.append((item, 1.0))
        else:
            raise ValueError(f"Unsupported terrain_mix value: {mix_cfg!r}")
        if not items:
            raise ValueError("terrain_mix must not be empty")
        parsed = [(self._terrain_id_from_value(name), float(weight)) for name, weight in items]
        if any(weight < 0.0 for _, weight in parsed) or sum(weight for _, weight in parsed) <= 0.0:
            raise ValueError(f"terrain_mix weights must be non-negative with positive sum: {mix_cfg!r}")
        return parsed

    def _allocate_env_terrain_ids(self, terrain_weights: list[tuple[int, float]]) -> torch.Tensor:
        total_weight = sum(weight for _, weight in terrain_weights)
        raw_counts = [self.num_envs * weight / total_weight for _, weight in terrain_weights]
        counts = [int(math.floor(count)) for count in raw_counts]
        remainder = self.num_envs - sum(counts)
        order = sorted(
            range(len(raw_counts)),
            key=lambda index: (raw_counts[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        ids: list[int] = []
        for (terrain_id, _), count in zip(terrain_weights, counts, strict=True):
            ids.extend([terrain_id] * count)
        if len(ids) != self.num_envs:
            raise RuntimeError(f"internal terrain allocation error: {len(ids)} != {self.num_envs}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def _build_env_terrain_ids(self) -> torch.Tensor:
        if self.num_terrains <= 0:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.get("terrain") is not None and self.cfg.get("terrain_mix") is not None:
            raise ValueError("Set either terrain or terrain_mix, not both")
        if self.cfg.get("terrain") is not None:
            return self._terrain_tensor_from_value(self.cfg["terrain"], expected_length=self.num_envs)
        mix = self._parse_terrain_mix(self.cfg.get("terrain_mix"))
        if not mix:
            if self.reference_terrain_ids is None:
                raise ValueError("terrain-conditioned env could not infer terrain ids")
            unique_reference_terrains = [
                terrain_id
                for terrain_id in range(self.num_terrains)
                if bool(torch.any(self.reference_terrain_ids == terrain_id).item())
            ]
            if not unique_reference_terrains:
                raise ValueError("reference set has no terrain ids")
            mix = [(terrain_id, 1.0) for terrain_id in unique_reference_terrains]
        return self._allocate_env_terrain_ids(mix)

    def _build_reference_ids_by_terrain(self) -> dict[int, torch.Tensor]:
        if self.num_terrains <= 0 or self.reference_terrain_ids is None:
            return {
                0: torch.arange(self.num_references, dtype=torch.long, device=self.device),
            }
        result: dict[int, torch.Tensor] = {}
        for terrain_id in range(self.num_terrains):
            ids = (self.reference_terrain_ids == terrain_id).nonzero(as_tuple=False).flatten()
            if ids.numel() > 0:
                result[terrain_id] = ids
        active_terrain_ids = set(int(terrain_id) for terrain_id in self.env_terrain_ids.detach().cpu().tolist())
        missing = sorted(active_terrain_ids - set(result))
        if missing:
            missing_names = [self.terrain_names[terrain_id] for terrain_id in missing]
            raise ValueError(
                f"No reference trajectories are available for active terrain(s): {missing_names}"
            )
        return result

    def terrain_counts(self) -> dict[str, int]:
        if self.num_terrains <= 0:
            return {}
        counts: dict[str, int] = {}
        for terrain_id in self.env_terrain_ids.detach().cpu().tolist():
            name = self.terrain_names[int(terrain_id)]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def reference_terrain_counts(self) -> dict[str, int]:
        if self.num_terrains <= 0 or self.reference_terrain_ids is None:
            return {}
        counts: dict[str, int] = {}
        for terrain_id in self.reference_terrain_ids.detach().cpu().tolist():
            name = self.terrain_names[int(terrain_id)]
            counts[name] = counts.get(name, 0) + 1
        return counts

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
        self.reset_idx(env_ids)
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(
        self,
        env_ids: torch.Tensor,
        reference_ids: torch.Tensor | None = None,
        terrain_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if terrain_ids is not None:
            if self.num_terrains <= 0:
                raise ValueError("terrain_ids were supplied, but the dynamics checkpoint is not terrain-conditioned")
            self.env_terrain_ids[env_ids] = self._terrain_tensor_from_value(
                terrain_ids,
                expected_length=env_ids.numel(),
            )
        if reference_ids is None:
            reference_ids = self._sample_reference_ids(env_ids)
        else:
            reference_ids = reference_ids.to(device=self.device, dtype=torch.long)
            if reference_ids.numel() != env_ids.numel():
                raise ValueError("reference_ids must have the same length as env_ids")
            self._validate_reference_ids_match_terrain(env_ids, reference_ids)

        self.ref_ids[env_ids] = reference_ids
        self.ref_step_buf[env_ids] = self.context_steps - 1
        self.state_hist[env_ids] = self.reference_states[reference_ids, : self.context_steps]
        self.action_hist[env_ids] = self.reference_actions[reference_ids, : self.context_steps]
        self.pose[env_ids] = self.reference_poses[reference_ids, self.context_steps - 1]
        self.actions[env_ids] = self.reference_actions[reference_ids, self.context_steps - 1]
        self.last_actions[env_ids] = self.actions[env_ids]
        self.episode_length_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0
        self.reset_buf[env_ids] = 0
        self.time_out_buf[env_ids] = False
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_pos_error_sum[env_ids] = 0.0
        self.episode_track_reward_sum[env_ids] = 0.0

    def _sample_reference_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.num_terrains <= 0:
            return torch.randint(
                low=0,
                high=self.num_references,
                size=(env_ids.numel(),),
                device=self.device,
            )
        sampled = torch.empty(env_ids.numel(), dtype=torch.long, device=self.device)
        env_terrain_ids = self.env_terrain_ids[env_ids]
        for terrain_id_tensor in torch.unique(env_terrain_ids):
            terrain_id = int(terrain_id_tensor.item())
            terrain_mask = env_terrain_ids == terrain_id
            candidates = self.reference_ids_by_terrain.get(terrain_id)
            if candidates is None or candidates.numel() == 0:
                terrain_name = self.terrain_names[terrain_id]
                raise ValueError(f"No reference trajectories are available for terrain {terrain_name!r}")
            candidate_indices = torch.randint(
                low=0,
                high=candidates.numel(),
                size=(int(terrain_mask.sum().item()),),
                device=self.device,
            )
            sampled[terrain_mask] = candidates[candidate_indices]
        return sampled

    def _validate_reference_ids_match_terrain(self, env_ids: torch.Tensor, reference_ids: torch.Tensor) -> None:
        if self.num_terrains <= 0 or self.reference_terrain_ids is None:
            return
        ref_terrain_ids = self.reference_terrain_ids[reference_ids]
        env_terrain_ids = self.env_terrain_ids[env_ids]
        mismatches = (ref_terrain_ids != env_terrain_ids).nonzero(as_tuple=False).flatten()
        if mismatches.numel() == 0:
            return
        first = int(mismatches[0].item())
        env_id = int(env_ids[first].item())
        ref_id = int(reference_ids[first].item())
        env_terrain = self.terrain_names[int(env_terrain_ids[first].item())]
        ref_terrain = self.terrain_names[int(ref_terrain_ids[first].item())]
        raise ValueError(
            "reference_ids must match each env's terrain id; "
            f"env {env_id} uses terrain {env_terrain!r}, but reference {ref_id} is {ref_terrain!r}. "
            "Pass matching terrain_ids or let reset_idx sample references automatically."
        )

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

        with torch.no_grad():
            for _ in range(self.action_repeat):
                self._nn_substep(driver_actions)

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

    def _nn_substep(self, driver_actions: torch.Tensor) -> None:
        self.action_hist[:, -1, :] = driver_actions
        k = self.dynamics_context_steps
        terrain = self.env_terrain_ids if self.num_terrains > 0 else None
        delta = self.model.predict_next_delta(
            self.state_hist[:, -k:, :],
            self.action_hist[:, -k:, :],
            terrain=terrain,
        )
        next_state = self.state_hist[:, -1, :] + delta
        self.pose = self._integrate_pose(self.pose, next_state)

        self.state_hist = torch.roll(self.state_hist, shifts=-1, dims=1)
        self.action_hist = torch.roll(self.action_hist, shifts=-1, dims=1)
        self.state_hist[:, -1, :] = next_state
        self.action_hist[:, -1, :] = driver_actions
        self.ref_step_buf = torch.clamp(self.ref_step_buf + 1, max=self.reference_length - 1)

    def _integrate_pose(self, pose: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        yaw_rate = next_state[:, self.state_index["yaw_rate_radps"]]
        vx_body = next_state[:, self.state_index["vel_body_x_mps"]]
        vy_body = next_state[:, self.state_index["vel_body_y_mps"]]
        yaw_next = pose[:, 2] + self.dt_s * yaw_rate
        cos_yaw = torch.cos(yaw_next)
        sin_yaw = torch.sin(yaw_next)
        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body
        x_next = pose[:, 0] + self.dt_s * vx_world
        y_next = pose[:, 1] + self.dt_s * vy_world
        return torch.stack([x_next, y_next, yaw_next], dim=-1)

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
        tracking_log = {
            "/tracking/track_reward": reward_terms["track_reward"].mean(),
            "/tracking/position_error_m": reward_terms["position_error_m"].mean(),
            "/tracking/yaw_error_abs_rad": reward_terms["yaw_error_abs_rad"].mean(),
            "/tracking/state_error_norm": reward_terms["state_error_norm"].mean(),
            "/tracking/action_rate": reward_terms["action_rate"].mean(),
            "/tracking/throttle_brake": reward_terms["throttle_brake"].mean(),
        }
        extras: dict[str, Any] = {
            "observations": {"critic": self.obs_buf},
            "time_outs": time_outs,
            "log": tracking_log,
        }
        done_env_ids = dones.nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() > 0:
            lengths = torch.clamp(self.episode_length_buf[done_env_ids].float(), min=1.0)
            episode_log = {
                "/episode/reward": self.episode_reward_sum[done_env_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/mean_pos_error_m": (self.episode_pos_error_sum[done_env_ids] / lengths).mean(),
                "/episode/mean_track_reward": (self.episode_track_reward_sum[done_env_ids] / lengths).mean(),
            }
            episode_log.update(tracking_log)
            extras["episode"] = episode_log
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
