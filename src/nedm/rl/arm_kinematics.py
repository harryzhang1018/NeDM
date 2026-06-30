"""Batched torch forward kinematics for the 4-DOF arm.

The arm is a SolidWorks import (non-DH joint frames), so the kinematic geometry is
*extracted once* from the live Chrono scene (``scripts/extract_arm_geometry.py``) into a JSON
spec, and reproduced here as a product-of-exponentials (PoE) FK that runs batched on GPU.
This gives every link's pose (not just the end-effector) so the RL safety filter
(``arm_safety.py``) can check link-vs-ground, link-vs-track, and link-vs-link clearance.

PoE around the recorded home configuration ``q_home``: for joint i with world axis ``a_i`` and
a pivot point ``p_i`` on the axis (both read at the home config), rotating by
``θ_i = joint_angle_sign_i * (q_i − q_home_i)`` applies the rigid transform ``exp(ξ_i θ_i)``
to the whole distal subtree, where

    exp(ξ_i θ) = [[ R(a_i, θ),  p_i − R(a_i, θ) p_i ], [0, 1]].

A link with last-upstream joint ``k`` has world pose ``(exp_0 ... exp_k) · H_link`` where
``H_link`` is its home REF→world pose. Twists are evaluated at home, so they are constant and
the product order captures the joint coupling exactly -- robust to the home settle sag.
Chrono's ``ChLinkMotorRotationTorque.GetMotorAngle()`` convention is opposite the frame-Z
axes for this imported arm, so extracted specs default ``joint_angle_signs`` to ``-1`` for
all four joints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _box_points(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """8 corners + center of an axis-aligned box, in the box's own frame (9, 3)."""
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append(center + np.array([sx * half[0], sy * half[1], sz * half[2]]))
    pts.append(center.copy())
    return np.asarray(pts, dtype=np.float64)


class ArmKinematics:
    """Batched PoE forward kinematics + per-link collision sample points."""

    def __init__(self, geom: dict[str, Any], device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=self.device)

        self.q_home = t(geom["q_home"])                                   # (4,)
        self.axes = t([j["axis"] for j in geom["joints"]])                # (4, 3) unit, world@home
        self.pivots = t([j["pivot"] for j in geom["joints"]])             # (4, 3) world@home
        self.num_joints = self.axes.shape[0]
        self.joint_angle_signs = t(geom.get("joint_angle_signs", [-1.0] * self.num_joints))
        if self.joint_angle_signs.shape != self.q_home.shape:
            raise ValueError(
                f"joint_angle_signs shape {tuple(self.joint_angle_signs.shape)} "
                f"does not match q_home shape {tuple(self.q_home.shape)}"
            )

        self.link_names: list[str] = [l["name"] for l in geom["links"]]
        self.link_k = torch.as_tensor([int(l["k"]) for l in geom["links"]],
                                      dtype=torch.long, device=self.device)  # (L,)
        self.H = t([l["H"] for l in geom["links"]]).reshape(-1, 4, 4)        # (L, 4, 4) REF->world@home
        self.num_links = len(self.link_names)

        self.box_center = t([l["box_center"] for l in geom["links"]])       # (L, 3) REF frame
        self.box_half = t([l["box_half"] for l in geom["links"]])           # (L, 3) REF frame

        pts = np.stack([_box_points(np.asarray(l["box_center"]), np.asarray(l["box_half"]))
                        for l in geom["links"]], axis=0)                    # (L, 9, 3)
        ones = np.ones((*pts.shape[:2], 1))
        self.link_pts_ref_h = t(np.concatenate([pts, ones], axis=-1))        # (L, 9, 4) homogeneous

        self.ee_link_idx = self.link_names.index(geom["ee"]["link"])
        ee_local = np.asarray(geom["ee"]["local"], dtype=np.float64)
        self.ee_local_h = t(np.concatenate([ee_local, [1.0]]))               # (4,)

        self.base_to_world = t(geom["base_to_world"]).reshape(4, 4)
        self.world_to_base = torch.linalg.inv(self.base_to_world)
        self.ground_z = float(geom.get("ground_z", 0.0))
        vehicle_boxes = geom.get("vehicle_boxes", [])
        if vehicle_boxes:
            self.vehicle_box_center = t([box["center"] for box in vehicle_boxes])
            self.vehicle_box_half = t([box["half"] for box in vehicle_boxes])
        else:
            self.vehicle_box_center = torch.empty(0, 3, dtype=dtype, device=self.device)
            self.vehicle_box_half = torch.empty(0, 3, dtype=dtype, device=self.device)

        self._skew = torch.stack([self._skew_matrix(self.axes[i]) for i in range(self.num_joints)])  # (4,3,3)
        self.eye3 = torch.eye(3, dtype=dtype, device=self.device)

        # Convenience metadata carried through for the env / goal sampler.
        self.joint_limits_lo = t(geom["joint_limits_lo"])
        self.joint_limits_hi = t(geom["joint_limits_hi"])
        self.dq_max = t(geom["dq_max"])

    @staticmethod
    def _skew_matrix(a: torch.Tensor) -> torch.Tensor:
        x, y, z = a[0], a[1], a[2]
        zero = torch.zeros((), dtype=a.dtype, device=a.device)
        return torch.stack([
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ])

    @classmethod
    def from_json(cls, path: str | Path, device: str | torch.device = "cpu",
                  dtype: torch.dtype = torch.float32) -> "ArmKinematics":
        geom = json.loads(Path(path).read_text())
        return cls(geom, device=device, dtype=dtype)

    def _cumulative_exponentials(self, q: torch.Tensor) -> torch.Tensor:
        """Cumulative joint transforms C[:, i] = exp_0 @ ... @ exp_i, shape (B, J, 4, 4)."""
        q = q.to(device=self.device, dtype=self.dtype)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] != self.num_joints:
            raise ValueError(f"q must have shape (B, {self.num_joints}) or ({self.num_joints},), got {tuple(q.shape)}")
        dq = (q - self.q_home) * self.joint_angle_signs                   # (B, J)
        batch = dq.shape[0]
        sin = torch.sin(dq)                                              # (B, J)
        cos = torch.cos(dq)
        k = self._skew.unsqueeze(0)                                     # (1, J, 3, 3)
        k2 = torch.matmul(self._skew, self._skew).unsqueeze(0)          # (1, J, 3, 3)
        rot = (self.eye3.view(1, 1, 3, 3)
               + sin[..., None, None] * k
               + (1.0 - cos)[..., None, None] * k2)                     # (B, J, 3, 3)
        pivot = self.pivots.view(1, self.num_joints, 3, 1)              # (1, J, 3, 1)
        trans = pivot.squeeze(-1) - torch.matmul(rot, pivot).squeeze(-1)  # (B, J, 3)
        exp = torch.zeros(batch, self.num_joints, 4, 4, dtype=self.dtype, device=self.device)
        exp[:, :, :3, :3] = rot
        exp[:, :, :3, 3] = trans
        exp[:, :, 3, 3] = 1.0
        cumulative = torch.empty_like(exp)
        cumulative[:, 0] = exp[:, 0]
        for i in range(1, self.num_joints):
            cumulative[:, i] = torch.matmul(cumulative[:, i - 1], exp[:, i])
        return cumulative

    def link_transforms(self, q: torch.Tensor) -> torch.Tensor:
        """World REF->world pose of every collision link, shape (B, L, 4, 4)."""
        cumulative = self._cumulative_exponentials(q)                   # (B, J, 4, 4)
        selected = cumulative[:, self.link_k]                           # (B, L, 4, 4)
        return torch.matmul(selected, self.H.unsqueeze(0))              # (B, L, 4, 4)

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`link_transforms`, matching the design-doc name."""
        return self.link_transforms(q)

    def link_points(self, q: torch.Tensor) -> torch.Tensor:
        """World collision sample points per link (8 box corners + center), (B, L, 9, 3)."""
        transforms = self.link_transforms(q)                            # (B, L, 4, 4)
        world_h = torch.einsum("blij,lpj->blpi", transforms, self.link_pts_ref_h)
        return world_h[..., :3]

    def ee_world(self, q: torch.Tensor) -> torch.Tensor:
        """End-effector (grasp-center) position in world frame, (B, 3)."""
        transforms = self.link_transforms(q)                            # (B, L, 4, 4)
        ee_transform = transforms[:, self.ee_link_idx]                  # (B, 4, 4)
        return torch.matmul(ee_transform, self.ee_local_h)[:, :3]

    def ee_base(self, q: torch.Tensor) -> torch.Tensor:
        """End-effector position in the arm-base frame (matches the model's ee_base)."""
        world = self.ee_world(q)
        homogeneous = torch.cat([world, torch.ones_like(world[:, :1])], dim=-1)
        return torch.matmul(homogeneous, self.world_to_base.T)[:, :3]

    def to_base(self, points_world: torch.Tensor) -> torch.Tensor:
        """Map world points (..., 3) into the arm-base frame."""
        homogeneous = torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)
        return torch.matmul(homogeneous, self.world_to_base.T)[..., :3]
