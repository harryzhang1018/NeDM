"""Torch geometric safety filter for the 4-DOF arm.

The arm dynamics model is trained on free-space motion only, so RL rollouts need
a fast shield that keeps commanded joint targets inside the free-space envelope.
This module uses the geometry extracted by ``scripts/extract_arm_geometry.py`` and
implemented by :class:`nedm.rl.arm_kinematics.ArmKinematics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from nedm.rl.arm_kinematics import ArmKinematics


DEFAULT_ADJACENT_LINK_PAIRS = {
    frozenset(("shoulder", "biceps")),
    frozenset(("biceps", "elbow")),
    frozenset(("elbow", "wrist")),
    frozenset(("elbow", "endoffactor")),
    frozenset(("wrist", "endoffactor")),
    frozenset(("wrist", "finger_1")),
    frozenset(("wrist", "finger_2")),
    frozenset(("endoffactor", "finger_1")),
    frozenset(("endoffactor", "finger_2")),
    frozenset(("finger_1", "finger_2")),
}
DEFAULT_INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_VEHICLE_LINK_NAMES = ("elbow", "wrist", "endoffactor", "finger_1", "finger_2")


@dataclass(frozen=True)
class ArmSafetyMargins:
    """Margins applied to raw signed clearances.

    Geometry margins are meters. The joint margin is radians.
    """

    ground_m: float = 0.03
    # The extracted vehicle AABB already has a conservative 0.2 m pad. Keep the
    # extra default margin at zero so the valid mounted/home pose is not blocked.
    vehicle_m: float = 0.0
    self_m: float = 0.03
    joint_rad: float = 0.0

    @classmethod
    def from_value(cls, value: "ArmSafetyMargins | dict[str, Any] | None") -> "ArmSafetyMargins":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        aliases = {
            "ground": "ground_m",
            "vehicle": "vehicle_m",
            "self": "self_m",
            "joint": "joint_rad",
        }
        kwargs = {}
        for key, item in value.items():
            kwargs[aliases.get(key, key)] = float(item)
        return cls(**kwargs)


class ArmSafetyFilter:
    """Batched joint-limit and geometric collision shield.

    ``clearance(q)`` returns the raw minimum signed geometric clearance in meters
    over ground, vehicle boxes, and non-adjacent arm-link boxes. ``safety_margin``
    subtracts configured margins and also includes joint-limit clearance in
    radians. ``filter(q, qcmd, raw_dq)`` clips the proposed command to joint
    limits, checks the interpolated command path, and blocks unsafe commands by
    returning zero delta.
    """

    def __init__(
        self,
        kin: ArmKinematics,
        margins: ArmSafetyMargins | dict[str, Any] | None = None,
        interpolation_alphas: Iterable[float] = DEFAULT_INTERPOLATION_ALPHAS,
        adjacent_link_pairs: Iterable[frozenset[str]] | None = None,
        vehicle_link_names: Iterable[str] | None = DEFAULT_VEHICLE_LINK_NAMES,
    ) -> None:
        self.kin = kin
        self.device = kin.device
        self.dtype = kin.dtype
        self.margins = ArmSafetyMargins.from_value(margins)
        self.alphas = torch.as_tensor(
            list(interpolation_alphas),
            dtype=self.dtype,
            device=self.device,
        )
        if self.alphas.numel() == 0:
            raise ValueError("interpolation_alphas must not be empty")

        adjacent = set(adjacent_link_pairs or DEFAULT_ADJACENT_LINK_PAIRS)
        self.self_pairs = self._build_self_pairs(adjacent)
        if vehicle_link_names is None:
            self.vehicle_link_indices = torch.arange(kin.num_links, dtype=torch.long, device=self.device)
        else:
            vehicle_link_names = list(vehicle_link_names)
            missing = [name for name in vehicle_link_names if name not in kin.link_names]
            if missing:
                raise ValueError(f"Unknown vehicle_link_names: {missing}")
            self.vehicle_link_indices = torch.tensor(
                [kin.link_names.index(name) for name in vehicle_link_names],
                dtype=torch.long,
                device=self.device,
            )

    def _build_self_pairs(self, adjacent: set[frozenset[str]]) -> list[tuple[int, int]]:
        pairs = []
        for i, name_i in enumerate(self.kin.link_names):
            for j in range(i + 1, self.kin.num_links):
                name_j = self.kin.link_names[j]
                if frozenset((name_i, name_j)) not in adjacent:
                    pairs.append((i, j))
        return pairs

    def _as_q_batch(self, q: torch.Tensor) -> torch.Tensor:
        q = q.to(device=self.device, dtype=self.dtype)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q.dim() != 2 or q.shape[-1] != self.kin.num_joints:
            raise ValueError(
                f"q must have shape (B, {self.kin.num_joints}) or ({self.kin.num_joints},), "
                f"got {tuple(q.shape)}"
            )
        return q

    def _link_points_from_transforms(self, transforms: torch.Tensor) -> torch.Tensor:
        world_h = torch.einsum("blij,lpj->blpi", transforms, self.kin.link_pts_ref_h)
        return world_h[..., :3]

    @staticmethod
    def _signed_distance_to_box(points: torch.Tensor, center: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
        """Signed distance from points to an axis-aligned box in the box frame."""
        delta = torch.abs(points - center) - half
        outside = torch.linalg.norm(torch.clamp(delta, min=0.0), dim=-1)
        inside = torch.clamp(torch.amax(delta, dim=-1), max=0.0)
        return outside + inside

    @staticmethod
    def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=-1)
        return torch.einsum("bij,bpj->bpi", transform, points_h)[..., :3]

    def _ground_clearance_from_points(self, points_world: torch.Tensor) -> torch.Tensor:
        z = points_world[..., 2] - self.kin.ground_z
        return z.reshape(points_world.shape[0], -1).amin(dim=1)

    def _vehicle_clearance_from_points(self, points_world: torch.Tensor) -> torch.Tensor:
        if self.kin.vehicle_box_center.numel() == 0 or self.vehicle_link_indices.numel() == 0:
            return torch.full((points_world.shape[0],), float("inf"), dtype=self.dtype, device=self.device)
        selected = points_world[:, self.vehicle_link_indices].reshape(points_world.shape[0], -1, 3)
        distances = self._signed_distance_to_box(
            selected[:, :, None, :],
            self.kin.vehicle_box_center[None, None, :, :],
            self.kin.vehicle_box_half[None, None, :, :],
        )
        return distances.reshape(points_world.shape[0], -1).amin(dim=1)

    def _self_clearance_from_geometry(self, transforms: torch.Tensor, points_world: torch.Tensor) -> torch.Tensor:
        if not self.self_pairs:
            return torch.full((points_world.shape[0],), float("inf"), dtype=self.dtype, device=self.device)

        inv_transforms = torch.linalg.inv(transforms)
        pair_clearances = []
        for i, j in self.self_pairs:
            points_i_in_j = self._transform_points(inv_transforms[:, j], points_world[:, i])
            points_j_in_i = self._transform_points(inv_transforms[:, i], points_world[:, j])
            dist_i_to_j = self._signed_distance_to_box(
                points_i_in_j,
                self.kin.box_center[j],
                self.kin.box_half[j],
            ).amin(dim=1)
            dist_j_to_i = self._signed_distance_to_box(
                points_j_in_i,
                self.kin.box_center[i],
                self.kin.box_half[i],
            ).amin(dim=1)
            pair_clearances.append(torch.minimum(dist_i_to_j, dist_j_to_i))
        return torch.stack(pair_clearances, dim=1).amin(dim=1)

    def joint_limit_clearance(self, q: torch.Tensor) -> torch.Tensor:
        """Minimum signed distance to configured joint limits, in radians."""
        q = self._as_q_batch(q)
        lower = q - self.kin.joint_limits_lo
        upper = self.kin.joint_limits_hi - q
        return torch.minimum(lower, upper).amin(dim=1)

    def clearance_terms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw signed clearances before margins.

        Geometric terms are meters; ``joint`` is radians.
        """
        q = self._as_q_batch(q)
        transforms = self.kin.link_transforms(q)
        points = self._link_points_from_transforms(transforms)
        return {
            "ground": self._ground_clearance_from_points(points),
            "vehicle": self._vehicle_clearance_from_points(points),
            "self": self._self_clearance_from_geometry(transforms, points),
            "joint": self.joint_limit_clearance(q),
        }

    def clearance(self, q: torch.Tensor) -> torch.Tensor:
        """Minimum raw signed geometric clearance in meters."""
        terms = self.clearance_terms(q)
        return torch.minimum(torch.minimum(terms["ground"], terms["vehicle"]), terms["self"])

    def safety_margin_terms(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Signed clearances after configured margins are subtracted."""
        terms = self.clearance_terms(q)
        return {
            "ground": terms["ground"] - self.margins.ground_m,
            "vehicle": terms["vehicle"] - self.margins.vehicle_m,
            "self": terms["self"] - self.margins.self_m,
            "joint": terms["joint"] - self.margins.joint_rad,
        }

    def safety_margin(self, q: torch.Tensor) -> torch.Tensor:
        """Minimum signed margin across geometry and joint limits."""
        terms = self.safety_margin_terms(q)
        return torch.minimum(
            torch.minimum(terms["ground"], terms["vehicle"]),
            torch.minimum(terms["self"], terms["joint"]),
        )

    def is_safe(self, q: torch.Tensor) -> torch.Tensor:
        return self.safety_margin(q) >= 0.0

    def filter(
        self,
        q: torch.Tensor,
        qcmd: torch.Tensor,
        raw_dq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Block unsafe command deltas.

        Returns ``(safe_dq, unsafe, clearance)``. ``safe_dq`` is the clipped
        command delta unless any interpolated command configuration is unsafe, in
        which case it is zero. ``clearance`` is the minimum raw geometric
        clearance along ``[q] + interp(qcmd, qcmd + safe_dq)`` before margins.
        """
        q = self._as_q_batch(q)
        qcmd = self._as_q_batch(qcmd)
        raw_dq = raw_dq.to(device=self.device, dtype=self.dtype)
        if raw_dq.dim() == 1:
            raw_dq = raw_dq.unsqueeze(0)
        if raw_dq.shape != qcmd.shape:
            raise ValueError(f"raw_dq shape {tuple(raw_dq.shape)} does not match qcmd shape {tuple(qcmd.shape)}")
        if q.shape[0] != qcmd.shape[0]:
            raise ValueError(f"q batch {q.shape[0]} does not match qcmd batch {qcmd.shape[0]}")

        qcmd_next = torch.clamp(qcmd + raw_dq, self.kin.joint_limits_lo, self.kin.joint_limits_hi)
        candidate_dq = qcmd_next - qcmd
        path = qcmd[:, None, :] + self.alphas.view(1, -1, 1) * candidate_dq[:, None, :]
        check_q = torch.cat([q[:, None, :], path], dim=1)
        flat_q = check_q.reshape(-1, self.kin.num_joints)

        path_margin = self.safety_margin(flat_q).reshape(q.shape[0], -1).amin(dim=1)
        path_clearance = self.clearance(flat_q).reshape(q.shape[0], -1).amin(dim=1)
        unsafe = path_margin < 0.0
        safe_dq = torch.where(unsafe[:, None], torch.zeros_like(candidate_dq), candidate_dq)
        return safe_dq, unsafe, path_clearance


__all__ = ["ArmSafetyFilter", "ArmSafetyMargins"]
