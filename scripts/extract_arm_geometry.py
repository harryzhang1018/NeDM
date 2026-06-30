"""Extract the arm's kinematic + collision geometry from the live Chrono scene.

Builds the settled M113+arm scene (via ``arm_data.build_and_prepare``) and dumps a JSON spec
that ``nedm.rl.arm_kinematics.ArmKinematics`` reproduces as a batched torch FK: per-joint world
axis/pivot at home, per-link home REF->world pose + collision box, the grasp point in the
endeffector frame, the base->world transform, a conservative vehicle (chassis+track) obstacle
box, and the ground plane. Then it self-validates by driving random qcmd and comparing the
torch FK link/EE positions against Chrono's actual frames (max error must be ~mm).

    PYTHONPATH=src conda run -n nedm python scripts/extract_arm_geometry.py \
        --output artifacts/arm_geometry/arm_geometry_v1.json --validate-steps 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pychrono as chrono
import pychrono.vehicle as veh
import torch

from nedm.arm_data import (
    CONTROL_DT,
    DQ_MAX,
    GROUND_PLANE_Z,
    JOINT_LIMITS_HI,
    JOINT_LIMITS_LO,
    STEP_SIZE,
    SmoothCommandSampler,
    build_and_prepare,
    clip_pose,
    gripper_center,
    _substep,
)
from nedm.rl.arm_kinematics import ArmKinematics

# Last-upstream joint index for every collision link (serial chain
# base→shoulder(j0)→biceps(j1)→elbow(j2)→endeffector(j3); wrist locked to endeffector,
# fingers ride it — all distal to j3).
LINK_JOINT_K = {
    "shoulder": 0,
    "biceps": 1,
    "elbow": 2,
    "wrist": 3,
    "endoffactor": 3,
    "finger_1": 3,
    "finger_2": 3,
}
VEHICLE_BOX_PAD_M = 0.2  # conservative padding on the track/chassis AABB


def vec3(v) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def framed_to_matrix(frame) -> np.ndarray:
    """4x4 homogeneous transform from a Chrono ChFramed (rotation columns via quaternion)."""
    q = frame.GetRot()
    p = frame.GetPos()
    cols = [q.Rotate(chrono.ChVector3d(1, 0, 0)),
            q.Rotate(chrono.ChVector3d(0, 1, 0)),
            q.Rotate(chrono.ChVector3d(0, 0, 1))]
    m = np.eye(4, dtype=np.float64)
    for j, c in enumerate(cols):
        m[:3, j] = [c.x, c.y, c.z]
    m[:3, 3] = [p.x, p.y, p.z]
    return m


def vehicle_obstacle_box(vehicle) -> dict[str, list[float]]:
    """Conservative world AABB over the chassis + all track-shoe bodies."""
    pts = [vehicle.GetChassisBody().GetPos()]
    for side in (veh.LEFT, veh.RIGHT):
        assembly = vehicle.GetTrackAssembly(side)
        for i in range(assembly.GetNumTrackShoes()):
            pts.append(assembly.GetTrackShoe(i).GetShoeBody().GetPos())
    arr = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
    lo = arr.min(axis=0) - VEHICLE_BOX_PAD_M
    hi = arr.max(axis=0) + VEHICLE_BOX_PAD_M
    return {"center": ((lo + hi) / 2).tolist(), "half": ((hi - lo) / 2).tolist()}


def extract_geometry(gripper, actuator, collision_links, vehicle) -> dict:
    q_home, _ = actuator.read_state()

    joints = []
    for name, motor in zip(["base", "shoulder", "biceps", "elbow"], actuator.motors):
        frame = motor.GetFrame2Abs()
        axis = frame.GetRot().Rotate(chrono.ChVector3d(0, 0, 1))
        joints.append({"name": name, "axis": vec3(axis), "pivot": vec3(frame.GetPos())})

    links = []
    endeff_matrix = None
    for name, body, center, half in collision_links:
        h = framed_to_matrix(body.GetFrameRefToAbs())
        links.append({
            "name": name,
            "k": LINK_JOINT_K[name],
            "H": h.reshape(-1).tolist(),
            "box_center": [float(c) for c in center],
            "box_half": [float(c) for c in half],
        })
        if name == "endoffactor":
            endeff_matrix = h
    if endeff_matrix is None:
        raise RuntimeError("endoffactor link not found in collision_links")

    # grasp center expressed in the endeffector REF frame (fingers ride it rigidly)
    grasp_world = np.array(vec3(gripper_center(gripper)) + [1.0])
    ee_local = (np.linalg.inv(endeff_matrix) @ grasp_world)[:3]

    return {
        "q_home": [float(v) for v in q_home],
        # Chrono's GetMotorAngle() convention for these imported torque motors is
        # opposite the saved frame-Z axes. ArmKinematics applies these signs to
        # (q - q_home) before evaluating the PoE transforms.
        "joint_angle_signs": [-1.0] * len(joints),
        "joints": joints,
        "links": links,
        "ee": {"link": "endoffactor", "local": ee_local.tolist()},
        "base_to_world": framed_to_matrix(gripper.base.GetFrameRefToAbs()).reshape(-1).tolist(),
        "vehicle_boxes": [vehicle_obstacle_box(vehicle)],
        "ground_z": float(GROUND_PLANE_Z),
        "joint_limits_lo": list(JOINT_LIMITS_LO),
        "joint_limits_hi": list(JOINT_LIMITS_HI),
        "dq_max": list(DQ_MAX),
    }


# Safe quasi-static sweep range (upper/forward, away from ground & track) so the arm
# tracks qcmd closely without contact violating the joints during FK validation.
SAFE_LO = [-1.5, 0.1, -0.4, -0.4]
SAFE_HI = [1.5, 0.7, 0.4, 0.4]


def _compare_fk(kin, actuator, gripper, collision_links, base_frame):
    """Max FK-vs-Chrono error (m) at the current measured q, plus the q used."""
    q, _ = actuator.read_state()
    qt = torch.tensor([q], dtype=torch.float64)
    transforms = kin.link_transforms(qt)[0]
    link_err = 0.0
    for li, (_, body, _, _) in enumerate(collision_links):
        ch = body.GetFrameRefToAbs().GetPos()
        link_err = max(link_err, float(np.linalg.norm(
            transforms[li, :3, 3].numpy() - np.array([ch.x, ch.y, ch.z]))))
    ee_world = gripper_center(gripper)
    ee_err = float(np.linalg.norm(kin.ee_world(qt)[0].numpy() - np.array(vec3(ee_world))))
    ee_b = base_frame.TransformPointParentToLocal(ee_world)
    ee_base_err = float(np.linalg.norm(kin.ee_base(qt)[0].numpy() - np.array(vec3(ee_b))))
    return link_err, ee_err, ee_base_err, q


def _hold(m113, terrain, actuator, target, n_sub, hold_ctrl_steps):
    """Drive qcmd toward target with bounded Δq and hold so q quasi-settles."""
    driver_inputs = veh.DriverInputs()
    driver_inputs.m_braking = 1.0
    for _ in range(hold_ctrl_steps):
        qcmd = list(actuator.qcmd)
        step = [max(-DQ_MAX[j], min(DQ_MAX[j], target[j] - qcmd[j])) for j in range(4)]
        actuator.qcmd = clip_pose([qcmd[j] + step[j] for j in range(4)])
        _substep(m113, terrain, actuator, driver_inputs, n_sub)


def _dump_per_link(kin, actuator, collision_links):
    q, _ = actuator.read_state()
    qt = torch.tensor([q], dtype=torch.float64)
    transforms = kin.link_transforms(qt)[0]
    link_err = 0.0
    print(f"  measured q = {[round(v,3) for v in q]}")
    for li, (name, body, _, _) in enumerate(collision_links):
        ch = body.GetFrameRefToAbs().GetPos()
        fk = transforms[li, :3, 3].numpy()
        err = float(np.linalg.norm(fk - np.array([ch.x, ch.y, ch.z])))
        link_err = max(link_err, err)
        print(f"    {name:11s} chrono=({ch.x:+.3f},{ch.y:+.3f},{ch.z:+.3f}) "
              f"fk=({fk[0]:+.3f},{fk[1]:+.3f},{fk[2]:+.3f}) err={err:.3f}")
    return link_err


def validate(geom, m113, terrain, gripper, actuator, collision_links, steps, seed) -> dict:
    kin = ArmKinematics(geom, device="cpu", dtype=torch.float64)
    rng = random.Random(seed)
    n_sub = max(1, int(round(CONTROL_DT / STEP_SIZE)))
    base_frame = gripper.base.GetFrameRefToAbs()

    # Diagnostic 0: FK(q_home) vs Chrono now (no driving) -- validates H extraction.
    print("[diag] q_home (no driving):")
    diag_link_err = _dump_per_link(kin, actuator, collision_links)
    # Diagnostic 1: single base-yaw move -- isolates base rotation axis/sign.
    print("[diag] after base->0.8 hold:")
    _hold(m113, terrain, actuator, [0.8, 0.4, 0.0, 0.0], n_sub, 60)
    diag_link_err = max(diag_link_err, _dump_per_link(kin, actuator, collision_links))
    # Diagnostic 2: single elbow move.
    print("[diag] after elbow->0.5 hold:")
    _hold(m113, terrain, actuator, [0.0, 0.4, 0.5, 0.0], n_sub, 60)
    diag_link_err = max(diag_link_err, _dump_per_link(kin, actuator, collision_links))

    link_err = diag_link_err
    ee_err = ee_base_err = 0.0
    worst = None
    targets = [[rng.uniform(SAFE_LO[j], SAFE_HI[j]) for j in range(4)] for _ in range(max(0, steps))]
    for target in targets:
        _hold(m113, terrain, actuator, target, n_sub, hold_ctrl_steps=60)
        le, ee, eb, q = _compare_fk(kin, actuator, gripper, collision_links, base_frame)
        if le > link_err:
            worst = {"target": [round(v, 3) for v in target], "q": [round(v, 3) for v in q],
                     "link_err_mm": round(le * 1000, 2)}
        link_err = max(link_err, le)
        ee_err = max(ee_err, ee)
        ee_base_err = max(ee_base_err, eb)
    if worst is not None:
        print(f"worst config: {worst}")
    return {"max_link_err_m": link_err, "max_ee_err_m": ee_err, "max_ee_base_err_m": ee_base_err}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract arm FK/collision geometry from Chrono.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/arm_geometry/arm_geometry_v1.json"))
    parser.add_argument("--validate-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    print("Building M113+arm scene (settle + pin base)...")
    m113, vehicle, terrain, gripper, actuator, collision_links, _ = build_and_prepare(render=False)
    print(f"collision links: {[name for name, *_ in collision_links]}")

    geom = extract_geometry(gripper, actuator, collision_links, vehicle)
    print(f"q_home: {[round(v, 4) for v in geom['q_home']]}")
    print(f"joint axes: {[[round(x, 3) for x in j['axis']] for j in geom['joints']]}")
    print(f"vehicle box: center={[round(x,2) for x in geom['vehicle_boxes'][0]['center']]} "
          f"half={[round(x,2) for x in geom['vehicle_boxes'][0]['half']]}")

    metrics = validate(geom, m113, terrain, gripper, actuator, collision_links,
                       steps=args.validate_steps, seed=args.seed)
    print(f"FK validation (max over {args.validate_steps} random configs): "
          f"link={metrics['max_link_err_m']*1000:.2f} mm, ee={metrics['max_ee_err_m']*1000:.2f} mm, "
          f"ee_base={metrics['max_ee_base_err_m']*1000:.2f} mm")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    geom["validation"] = metrics
    args.output.write_text(json.dumps(geom, indent=2))
    print(f"wrote {args.output}")
    ok = metrics["max_link_err_m"] < 0.01 and metrics["max_ee_err_m"] < 0.01
    print("FK MATCHES Chrono (<1 cm)" if ok else "WARNING: FK error exceeds 1 cm — check axes/signs")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
