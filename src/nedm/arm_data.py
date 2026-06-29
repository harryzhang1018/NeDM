"""Arm-only dynamics data collection for the tracked-vehicle manipulator.

The reach-mode data collector from ``docs/arm-dyn-model.md`` (sections 5-7) and
the second NeDM dynamics study case, parallel to the HMMWV pipeline in
``hmmwv_data.py``. It drives the 4-DOF gripper arm through smooth random
free-space motions while the tracked base is held fixed, and records joint-space
transitions

    x_t = [q_t, qdot_t, qcmd_t]   a_t = Δqcmd_t   ->   x_{t+1} = [q_{t+1}, ...]

so a learned arm dynamics model ``f_arm`` can be trained on free-space motion.

The scene -- M113 tracked vehicle + front-welded ``LRV_Arm`` + flat rigid terrain
-- is built here (ported from lunar-manip's ``TrackedVeh_Builder``). The gripper
arm model and its SolidWorks export/meshes are bundled under ``src/arm_model``.

Four pieces beyond a plain assemble-and-drive scene:

1. **Arm collision geometry.** Only the gripper fingers ship with collision
   shapes; the links do not. We fit one box collision shape per link from the
   link's (already scaled) visual-mesh AABB, in the body reference frame.

2. **Collision detection (arm-vs-track, arm self-collision, arm-vs-ground).**
   Each arm link gets its OWN collision family, whose mask allows colliding with
   the M113 track-shoe family (6) *and* the families of all NON-ADJACENT arm
   links; chain neighbours (which share a joint and legitimately touch) and the
   road wheels / sprocket / idler / chassis are excluded. So a nonzero
   ``body.GetContactForce()`` on a link means it touched a track shoe or folded
   into a non-adjacent link. The flat ground is handled separately and exactly:
   a link hits the ground when its collision-box lowest point reaches the terrain
   plane (z=0), a geometric test that needs no contact identity (pychrono's
   contact callback is broken here) and no force-timing luck. All three are
   termination signals; the self/track label is heuristic (which links report
   force at once), ground is exact.

3. **Compliant PD actuation.** ``LRV_Arm`` drives its joints with
   ``ChLinkMotorRotationAngle``, a *hard angle constraint* -> ``q == qcmd``
   exactly, i.e. no dynamics to learn. The plan instead wants "a low-level PD
   controller in Chrono drives the physical joints toward qcmd" (doc 5.1), so we
   swap the four angle motors for ``ChLinkMotorRotationTorque`` + a per-substep
   PD loop. Now ``q`` lags ``qcmd`` under gravity/inertia and the transitions
   carry real dynamics.

4. **Episode / termination logic + logging.** Every episode starts from a fresh
   Chrono scene at the imported home configuration, then follows smooth random
   ``qcmd`` targets (doc 6.3). One CSV row per control step holds a full
   transition; episodes truncate at ``--max-steps`` or terminate the moment a
   link contacts a track, self-collides, hits the ground, or leaves the configured
   joint range. The terminal row is flagged ``collision=1`` so training keeps it
   out of the free-space set and routes it to safety-filter debugging.

Run in the NeDM conda env:

    conda run -n nedm python -m nedm.arm_data --episodes 4 --max-steps 500

Add ``--render`` to watch one run in the Irrlicht viewer.

NOTE on scope: this collects Chrono ground-truth data and uses Chrono's own
collision detection for termination. The lightweight geometric *safety filter*
used during learned-model RL rollouts (doc 7.2-7.4) is a separate component and
is intentionally not implemented here.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pychrono as chrono
import pychrono.vehicle as veh
from arm_model import LRV_Arm


def repo_root_from_module() -> Path:
    """NeDM repo root (src/nedm/arm_data.py -> parents[2])."""
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Scene geometry (ported from lunar-manip TrackedVeh_Builder)
# ---------------------------------------------------------------------------
INIT_LOC = chrono.ChVector3d(0, 0, 0.9)
INIT_ROT = chrono.ChQuaterniond(1, 0, 0, 0)
# Physics substep. The single-pin M113 track needs a small step (5e-4 s) to stay
# assembled; this also sets the PD control loop's substep.
STEP_SIZE = 5e-4
# Uniform geometric scale of the gripper arm (mass/inertia stay at 1x).
ARM_SCALE = 2.0
# Arm-base mount point in the chassis reference frame, and mount orientation.
ARM_OFFSET = chrono.ChVector3d(-2.5, 0.0, 0.4)
ARM_MOUNT_ROT = chrono.QuatFromAngleZ(math.pi)


# ---------------------------------------------------------------------------
# Simulation / control rates
# ---------------------------------------------------------------------------
# Control / record period. qcmd is updated and one transition is recorded every
# CONTROL_DT seconds; the physics is substepped at STEP_SIZE in between.
CONTROL_DT = 0.02  # 50 Hz
# Seconds to let the rig settle on its tracks before the base is pinned.
SETTLE_TIME = 0.5

# ---------------------------------------------------------------------------
# Collision families (Bullet supports families 0..14 in this build)
# ---------------------------------------------------------------------------
# Occupied M113/terrain families (probed): 0 sprocket+misc, 3 idler, 4 road
# wheels, 6 track shoes, 14 terrain patch. Free: 1,2,5,7,8,9,10,11,12,13.
#
# We detect TWO kinds of unsafe contact during collection and discard the
# transition for both: arm<->track and arm<->arm self-collision. Chrono's
# collision filtering is per-family (not per-pair) and its ReportContactCallback
# is broken in this build, so we cannot read contact identity. Instead each arm
# link gets its OWN family, and its mask allows colliding with:
#   * the track-shoe family (6)              -> arm-vs-track, and
#   * the families of all NON-ADJACENT links -> arm self-collision.
# Chain neighbours (which share a joint and legitimately touch) are excluded, as
# is each link's own family. With that, a nonzero GetContactForce on a link can
# only come from a track shoe or a non-adjacent link -- never terrain, wheels,
# sprocket, idler, or an adjacent joint. The contacting pair is then recovered
# heuristically from which links report force simultaneously.
TRACK_SHOE_FAMILY = 6
# One free family per collision link (in kinematic-chain order).
LINK_FAMILIES = {
    "shoulder": 1,
    "biceps": 2,
    "elbow": 5,
    "wrist": 7,
    "endoffactor": 8,
    "finger_1": 9,
    "finger_2": 10,
}
# Chain-neighbour (and tightly-coupled) pairs whose contact is legitimate and
# must NOT be treated as a self-collision. Everything else is a real self-hit.
ADJACENT_LINK_PAIRS = {
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
# Contact force (N) above which an arm link counts as "in contact" with a track
# shoe or another link. SMC contact within the 3 mm envelope can produce tiny
# forces near grazing; 1 N keeps grazing/noise out while catching real penetration.
CONTACT_FORCE_THRESHOLD_N = 1.0
# Arm-vs-ground is detected GEOMETRICALLY, not via Chrono contact: the terrain is
# a known flat plane at z = GROUND_PLANE_Z, so a link touches the ground exactly
# when its collision-box lowest world point drops to within GROUND_CONTACT_MARGIN
# of the plane. This is exact for a flat plane, unambiguous (force can't tell
# ground from track), and timing-independent (no bounce). The arm is therefore
# not put in the terrain's collision family; it tunnels through, but the step is
# terminated and discarded the instant the box reaches the plane.
GROUND_PLANE_Z = 0.0
GROUND_CONTACT_MARGIN = 0.02
# Shrink factor applied to each link's AABB half-extents (1.0 = exact AABB).
# Slightly < 1 trims the boxy over-coverage at the rounded link ends.
COLLISION_BOX_SHRINK = 0.9

# ---------------------------------------------------------------------------
# Joint model. q indexes the four actuated rotation joints, in motor-internal
# convention (we drive the torque motors directly and read GetMotorAngle, so q
# and qcmd live in the same frame and Δq/qdot are self-consistent):
#   0: base   -> shoulder (yaw of the whole arm)
#   1: shoulder -> biceps (shoulder pitch)
#   2: biceps -> elbow    (elbow pitch)
#   3: elbow  -> endeffector (wrist pitch)
# ---------------------------------------------------------------------------
JOINT_NAMES = ["base", "shoulder", "biceps", "elbow"]
# Soft joint limits (rad), motor convention. Tune to the real workspace; these
# give broad coverage. NB: self-collision needs a deep elbow fold (~170 deg) that
# the +/-90 deg pitch limits do not reach, so widen these to generate self-hits.
JOINT_LIMITS_LO = [-math.pi, -0.5 * math.pi, -0.5 * math.pi, -0.5 * math.pi]
JOINT_LIMITS_HI = [math.pi, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi]
# Actual measured q is also guarded by these limits. qcmd is clipped, but the
# torque-driven joint can overshoot under inertia; terminate those transitions so
# the free-space training set stays inside the intended arm range.
JOINT_LIMIT_EPS = 1e-6
# Max |Δqcmd| per control step (rad). ~0.04 rad at 50 Hz -> ~2 rad/s (doc 6.3
# suggests 1-5 deg per control step).
DQ_MAX = [0.05, 0.04, 0.04, 0.04]

# PD gains / torque clamps per joint (validated stable at STEP_SIZE under the
# 2x-scaled arm's gravity load). Heavier links (shoulder/biceps) need stiffer
# gains and higher torque ceilings.
PD_KP = [4000.0, 30000.0, 20000.0, 4000.0]
PD_KD = [200.0, 2000.0, 1200.0, 200.0]
PD_TAU_MAX = [3000.0, 26000.0, 16000.0, 3000.0]


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def configure_vehicle_data_path() -> None:
    """Point the vehicle data path at the Chrono install's ``vehicle/`` tree.

    The M113 model files (incl. track-shoe collision hulls referenced relative to
    the data root) live there; the arm loads its own bundled meshes.
    """
    veh.SetVehicleDataPath(os.path.join(chrono.GetChronoDataPath(), "vehicle") + os.sep)


def build_scene():
    """Create the M113 tracked vehicle, front-welded arm, and flat terrain.

    Returns (m113, vehicle, terrain, gripper). Mirrors the SMC + high-iteration
    BB-solver configuration that keeps the single-pin track assembled.
    """
    configure_vehicle_data_path()

    m113 = veh.M113()
    m113.SetContactMethod(chrono.ChContactMethod_SMC)
    m113.SetTrackShoeType(veh.TrackShoeType_SINGLE_PIN)
    m113.SetDoublePinTrackShoeType(veh.DoublePinTrackShoeType_ONE_CONNECTOR)
    m113.SetTrackBushings(False)          # SMC iterative solvers can't use bushings
    m113.SetSuspensionBushings(False)
    m113.SetTrackStiffness(False)
    m113.SetDrivelineType(veh.DrivelineTypeTV_BDS)
    m113.SetBrakeType(veh.BrakeType_SHAFTS)
    # SIMPLE engine (zero torque at zero throttle) so the braked rig does not
    # creep; AUTOMATIC_SIMPLE_MAP transmission as in the demo.
    m113.SetEngineType(veh.EngineModelType_SIMPLE)
    m113.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    m113.SetChassisCollisionType(veh.CollisionType_NONE)
    m113.SetChassisFixed(False)
    m113.CreateTrack(True)
    m113.SetInitPosition(chrono.ChCoordsysd(INIT_LOC, INIT_ROT))
    m113.Initialize()

    track_vis = chrono.VisualizationType_MESH
    m113.SetChassisVisualizationType(chrono.VisualizationType_NONE)
    m113.SetSprocketVisualizationType(chrono.VisualizationType_MESH)
    m113.SetIdlerVisualizationType(track_vis)
    m113.SetSuspensionVisualizationType(track_vis)
    m113.SetIdlerWheelVisualizationType(track_vis)
    m113.SetRoadWheelVisualizationType(track_vis)
    m113.SetTrackShoeVisualizationType(track_vis)

    system = m113.GetSystem()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    # BB VI solver with 100 iterations keeps the many single-pin track
    # constraints together (the default ~50 lets shoes drift off the wheels).
    solver = chrono.ChSolverBB()
    solver.SetMaxIterations(100)
    solver.SetOmega(0.8)
    solver.SetSharpnessLambda(1.0)
    system.SetSolver(solver)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)

    vehicle = m113.GetVehicle()

    # Gripper arm welded to the front deck (locked to the chassis body).
    arm_offset = vehicle.GetChassis().GetPos() + ARM_OFFSET
    gripper = LRV_Arm(system, arm_offset, vehicle, scale=ARM_SCALE, mount_rot=ARM_MOUNT_ROT)

    # Flat rigid patch (100 x 100 m) at ground level under the vehicle.
    terrain = veh.RigidTerrain(system)
    minfo = chrono.ChContactMaterialData()
    minfo.mu = 0.9
    minfo.cr = 0.2
    minfo.Y = 2e7
    patch_mat = minfo.CreateMaterial(chrono.ChContactMethod_SMC)
    patch = terrain.AddPatch(
        patch_mat,
        chrono.ChCoordsysd(chrono.ChVector3d(INIT_LOC.x, INIT_LOC.y, 0), chrono.QUNIT),
        100.0, 100.0)
    patch.SetColor(chrono.ChColor(0.5, 0.8, 0.5))
    terrain.Initialize()

    return m113, vehicle, terrain, gripper


def make_vis(vehicle, title="Arm data collection"):
    """Create an Irrlicht tracked-vehicle visual system attached to ``vehicle``."""
    import pychrono.irrlicht as chronoirr  # noqa: F401 (ensures backend is present)

    vis = veh.ChTrackedVehicleVisualSystemIrrlicht()
    vis.SetWindowTitle(title)
    vis.SetWindowSize(1280, 1024)
    vis.SetChaseCamera(chrono.ChVector3d(0.0, 0.0, 1.0), 12.0, 3.0)
    vis.Initialize()
    vis.AddLightWithShadow(INIT_LOC + chrono.ChVector3d(0, 0, 30), INIT_LOC, 20, 1, 60, 50)
    vis.AddTypicalLights()
    vis.AttachVehicle(vehicle)
    return vis


# ---------------------------------------------------------------------------
# Collision geometry
# ---------------------------------------------------------------------------
def _visual_aabb(body):
    """Axis-aligned bounding box of a body's visual mesh, in its REF frame.

    Handles both ``ChVisualShapeTriangleMesh`` (what ``LRV_Arm`` leaves behind
    after geometric scaling, so the vertices are already scaled) and the
    unscaled ``ChVisualShapeModelFile`` case. Returns (min, max) or None.
    """
    vm = body.GetVisualModel()
    if vm is None:
        return None
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    found = False
    for i in range(vm.GetNumShapes()):
        shp = vm.GetShape(i)
        frame = vm.GetShapeFrame(i)
        verts = None
        tm = chrono.CastToChVisualShapeTriangleMesh(shp)
        if tm is not None:
            verts = tm.GetMesh().GetCoordsVertices()
        else:
            mf = chrono.CastToChVisualShapeModelFile(shp)
            if mf is not None:
                tri = chrono.ChTriangleMeshConnected()
                tri.LoadWavefrontMesh(mf.GetFilename(), True, True)
                verts = tri.GetCoordsVertices()
        if verts is None:
            continue
        for k in range(len(verts)):
            v = frame.TransformPointLocalToParent(verts[k])
            lo[0], lo[1], lo[2] = min(lo[0], v.x), min(lo[1], v.y), min(lo[2], v.z)
            hi[0], hi[1], hi[2] = max(hi[0], v.x), max(hi[1], v.y), max(hi[2], v.z)
            found = True
    if not found:
        return None
    return chrono.ChVector3d(*lo), chrono.ChVector3d(*hi)


def add_link_collision_box(body, material, family, allowed_families,
                           shrink=COLLISION_BOX_SHRINK) -> bool:
    """Fit one box collision shape to ``body`` from its visual-mesh AABB.

    The box is placed at the AABB centre in the body REF frame (the arm's visual
    shapes are attached at the REF origin, so visual coordinates == REF
    coordinates). The collision model is assigned ``family`` and restricted to
    collide only with the families in ``allowed_families`` (its own family is not
    included, so self-overlap of one link's shape never reports). Returns True if
    a box was added.
    """
    aabb = _visual_aabb(body)
    if aabb is None:
        return None
    lo, hi = aabb
    size_x = max((hi.x - lo.x) * shrink, 1e-3)
    size_y = max((hi.y - lo.y) * shrink, 1e-3)
    size_z = max((hi.z - lo.z) * shrink, 1e-3)
    center = chrono.ChVector3d(0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z))

    # Replace any pre-existing collision model (e.g. the fingers' contact pads)
    # so every collision link follows the same family rules.
    existing = body.GetCollisionModel()
    if existing is not None:
        existing.Clear()
    body.AddCollisionShape(chrono.ChCollisionShapeBox(material, size_x, size_y, size_z),
                           chrono.ChFramed(center, chrono.QUNIT))
    body.EnableCollision(True)

    cm = body.GetCollisionModel()
    cm.SetFamily(family)
    for fam in range(15):
        cm.DisallowCollisionsWith(fam)  # default-deny everything (incl. own family)
    for fam in allowed_families:
        cm.AllowCollisionsWith(fam)
    # Box geometry in the body REF frame, reused for the geometric ground check.
    return ((center.x, center.y, center.z),
            (0.5 * size_x, 0.5 * size_y, 0.5 * size_z))


def _are_adjacent(a, b):
    return frozenset((a, b)) in ADJACENT_LINK_PAIRS


def setup_arm_collision(gripper, link_families=LINK_FAMILIES,
                        track_family=TRACK_SHOE_FAMILY):
    """Add per-link collision boxes wired for arm-vs-track AND self-collision.

    Each link gets its own family; its mask allows the track-shoe family plus the
    families of all non-adjacent links (the ground is handled geometrically, see
    ``arm_contact``). Returns a list of (name, body, box_center, box_half) for the
    links that now have collision geometry; the box geometry (in the body REF
    frame) feeds both the ``GetContactForce`` poll and the ground min-z check.
    """
    material = chrono.ChContactMaterialSMC()
    material.SetFriction(0.6)
    material.SetYoungModulus(2e7)
    links = []
    for name, family in link_families.items():
        body = getattr(gripper, name)
        allowed = [track_family] + [
            other_fam for other_name, other_fam in link_families.items()
            if other_name != name and not _are_adjacent(name, other_name)
        ]
        geom = add_link_collision_box(body, material, family, allowed)
        if geom is not None:
            center, half = geom
            links.append((name, body, center, half))
    return links


def _link_world_min_z(body, center, half):
    """Lowest world-z of a link's collision box (8 REF-frame corners -> world)."""
    ref = body.GetFrameRefToAbs()
    cx, cy, cz = center
    hx, hy, hz = half
    zmin = math.inf
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                p = ref.TransformPointLocalToParent(
                    chrono.ChVector3d(cx + sx * hx, cy + sy * hy, cz + sz * hz))
                zmin = min(zmin, p.z)
    return zmin


def arm_contact(collision_links, threshold=CONTACT_FORCE_THRESHOLD_N,
                ground_z=GROUND_PLANE_Z + GROUND_CONTACT_MARGIN):
    """Classify arm contact this step.

    Returns (hit, kind, involved, max_force) where:
      * ``hit``      -- any arm link touched a track shoe, the ground, or a
                        non-adjacent link,
      * ``kind``     -- "self" if a non-adjacent link PAIR simultaneously reports
                        contact force (Newton's third law), else "ground" if a
                        link's collision box reaches the terrain plane, else
                        "track", else None,
      * ``involved`` -- the link name(s) describing the contact,
      * ``max_force``-- largest per-link contact-force magnitude (N); ~0 for a
                        purely geometric ground hit.

    Self/track are read from Chrono contact forces; ground is geometric (the
    terrain is a flat plane at GROUND_PLANE_Z). The self/track label is a
    heuristic (no contact identity in this pychrono build); termination fires on
    ``hit`` regardless of the label.
    """
    forces = {}
    ground_links = []
    for name, body, center, half in collision_links:
        forces[name] = body.GetContactForce().Length()
        if _link_world_min_z(body, center, half) <= ground_z:
            ground_links.append(name)
    in_contact = [n for n, f in forces.items() if f > threshold]
    max_force = max(forces.values()) if forces else 0.0

    if not in_contact and not ground_links:
        return False, None, [], max_force

    # Self-collision iff two force-contacting links form a non-adjacent pair.
    for i, a in enumerate(in_contact):
        for b in in_contact[i + 1:]:
            if not _are_adjacent(a, b):
                return True, "self", sorted((a, b)), max_force
    if ground_links:
        return True, "ground", sorted(ground_links), max_force
    worst = max(in_contact, key=lambda n: forces[n])
    return True, "track", [worst], max_force


# ---------------------------------------------------------------------------
# Compliant PD actuation
# ---------------------------------------------------------------------------
class ArmPdActuator:
    """Swaps the arm's rotation-angle motors for torque motors + a PD loop.

    Must be constructed AFTER the arm is fully built and mounted, because the
    joint frames are read from the existing motors' current absolute frames.
    """

    def __init__(self, gripper):
        self.gripper = gripper
        angle_motors = [gripper.motor_base_shoulder, gripper.motor_shoulder_biceps,
                        gripper.motor_biceps_elbow, gripper.motor_elbow_eef]
        body_pairs = [(gripper.base, gripper.shoulder),
                      (gripper.shoulder, gripper.biceps),
                      (gripper.biceps, gripper.elbow),
                      (gripper.elbow, gripper.endoffactor)]
        system = gripper.system

        # Capture the live joint frames, then drop the hard-constraint motors.
        frames = [m.GetFrame2Abs() for m in angle_motors]
        for m in angle_motors:
            system.RemoveLink(m)

        self.motors = []
        for (b1, b2), frame in zip(body_pairs, frames):
            tm = chrono.ChLinkMotorRotationTorque()
            tm.Initialize(b1, b2, frame)
            tm.SetTorqueFunction(chrono.ChFunctionConst(0.0))
            system.Add(tm)
            self.motors.append(tm)

        self.qcmd = [0.0, 0.0, 0.0, 0.0]

    def read_state(self):
        q = [m.GetMotorAngle() for m in self.motors]
        qd = [m.GetMotorAngleDt() for m in self.motors]
        return q, qd

    def apply_pd(self):
        """Compute and set the PD torque for the current ``qcmd`` (one substep)."""
        for i, m in enumerate(self.motors):
            q = m.GetMotorAngle()
            qd = m.GetMotorAngleDt()
            tau = PD_KP[i] * (self.qcmd[i] - q) - PD_KD[i] * qd
            tau = max(-PD_TAU_MAX[i], min(PD_TAU_MAX[i], tau))
            m.SetTorqueFunction(chrono.ChFunctionConst(tau))


def gripper_center(gripper):
    """World position of the grasp point (midpoint between the two fingers)."""
    p1 = gripper.finger_1.GetPos()
    p2 = gripper.finger_2.GetPos()
    return chrono.ChVector3d(0.5 * (p1.x + p2.x), 0.5 * (p1.y + p2.y), 0.5 * (p1.z + p2.z))


# ---------------------------------------------------------------------------
# Smooth random command generation (doc 6.3)
# ---------------------------------------------------------------------------
class SmoothCommandSampler:
    """Move qcmd toward a random target with bounded Δq, resampling the target.

    a_t = clip(q_target - qcmd_t, -dq_max, dq_max), with q_target redrawn every
    ~`resample_every` control steps. This yields realistic, low-jerk actuator
    usage rather than per-step white noise.
    """

    def __init__(self, rng, lo=JOINT_LIMITS_LO, hi=JOINT_LIMITS_HI, dq_max=DQ_MAX,
                 resample_every=50):
        self.rng = rng
        self.lo, self.hi, self.dq_max = lo, hi, dq_max
        self.resample_every = resample_every
        self.target = self._sample_pose()
        self._k = 0

    def _sample_pose(self):
        return [self.rng.uniform(self.lo[i], self.hi[i]) for i in range(4)]

    def next_action(self, qcmd):
        if self._k % self.resample_every == 0:
            self.target = self._sample_pose()
        self._k += 1
        return [max(-self.dq_max[i], min(self.dq_max[i], self.target[i] - qcmd[i]))
                for i in range(4)]


def clip_pose(q, lo=JOINT_LIMITS_LO, hi=JOINT_LIMITS_HI):
    return [max(lo[i], min(hi[i], q[i])) for i in range(4)]


def joint_limit_violation(q, lo=JOINT_LIMITS_LO, hi=JOINT_LIMITS_HI, eps=JOINT_LIMIT_EPS):
    """Return (hit, labels) when measured joint position leaves configured range."""
    labels = []
    for i, value in enumerate(q):
        if value < lo[i] - eps:
            labels.append(f"q_{i}_lo")
        elif value > hi[i] + eps:
            labels.append(f"q_{i}_hi")
    return bool(labels), labels


class EeBinBalancer:
    """Online row balancer using measured gripper-base EE positions.

    This affects which free-space transitions are written, not the Chrono
    simulation itself. Terminal collision rows are still written by the episode
    driver so safety failures are not hidden.
    """

    def __init__(self, grid, bin_cap, bounds_lo, bounds_hi):
        self.grid = tuple(grid)
        self.bin_cap = int(bin_cap)
        self.bounds_lo = tuple(bounds_lo)
        self.bounds_hi = tuple(bounds_hi)
        self.counts = {}
        self.accepted = 0
        self.rejected = 0
        self.out_of_bounds = 0

    @property
    def enabled(self):
        return self.bin_cap > 0 and all(n > 0 for n in self.grid)

    def _bin_index(self, ee_base):
        vals = (ee_base.x, ee_base.y, ee_base.z)
        idx = []
        for value, lo, hi, n in zip(vals, self.bounds_lo, self.bounds_hi, self.grid):
            if value < lo or value > hi:
                return None
            frac = (value - lo) / max(hi - lo, 1e-12)
            idx.append(min(n - 1, max(0, int(frac * n))))
        return tuple(idx)

    def accept(self, ee_base):
        if not self.enabled:
            self.accepted += 1
            return True
        idx = self._bin_index(ee_base)
        if idx is None:
            # Bounds are an operator hint, not a hard data-validity rule.
            self.out_of_bounds += 1
            self.accepted += 1
            return True
        count = self.counts.get(idx, 0)
        if count >= self.bin_cap:
            self.rejected += 1
            return False
        self.counts[idx] = count + 1
        self.accepted += 1
        return True

    def summary(self):
        if not self.enabled:
            return {"enabled": False}
        total_bins = self.grid[0] * self.grid[1] * self.grid[2]
        return {
            "enabled": True,
            "grid": list(self.grid),
            "bin_cap": self.bin_cap,
            "bounds_lo": list(self.bounds_lo),
            "bounds_hi": list(self.bounds_hi),
            "accepted_rows": self.accepted,
            "rejected_rows": self.rejected,
            "out_of_bounds_rows": self.out_of_bounds,
            "occupied_bins": len(self.counts),
            "total_bins": total_bins,
        }


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    episode_id: str
    split: str
    rows: int
    terminated_collision: bool
    collision_kind: str | None       # "self" | "track" | "ground" | "joint_limit" | None
    collision_links: list[str]       # link(s) involved in the terminal contact
    csv_path: Path
    start_q: list[float]
    start_qd: list[float]
    start_qcmd: list[float]


CSV_FIELDS = (
    ["episode_id", "split", "sample_index", "time_s", "collision",
     "collision_kind", "contact_force_n"]
    + [f"q_{j}" for j in range(4)]
    + [f"qd_{j}" for j in range(4)]
    + [f"qcmd_{j}" for j in range(4)]
    + [f"act_{j}" for j in range(4)]
    + [f"qcmd_next_{j}" for j in range(4)]
    + [f"q_next_{j}" for j in range(4)]
    + [f"qd_next_{j}" for j in range(4)]
    + ["ee_x", "ee_y", "ee_z", "ee_base_x", "ee_base_y", "ee_base_z"]
    + ["ee_next_x", "ee_next_y", "ee_next_z",
       "ee_next_base_x", "ee_next_base_y", "ee_next_base_z"]
)


def assign_split(episode_id, validation_ratio):
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) / 0xFFFFFFFF < validation_ratio else "train"


def _substep(m113, terrain, actuator, driver_inputs, n_substeps, vis=None):
    """Advance the physics ``n_substeps`` while holding the PD command + brakes.

    When a visual system is attached, draw one frame per call (~one frame per
    control step) and pump the window's event loop so it stays responsive during
    settle / reset as well as recording.
    """
    system = m113.GetSystem()
    if vis is not None:
        vis.Run()
        vis.BeginScene()
        vis.Render()
        vis.EndScene()
    for _ in range(n_substeps):
        t = system.GetChTime()
        actuator.apply_pd()
        terrain.Synchronize(t)
        m113.Synchronize(t, driver_inputs)
        if vis is not None:
            vis.Synchronize(t, driver_inputs)
        terrain.Advance(STEP_SIZE)
        m113.Advance(STEP_SIZE)
        if vis is not None:
            vis.Advance(STEP_SIZE)


def run_episode(m113, vehicle, terrain, gripper, actuator, collision_links,
                episode_id, validation_ratio, rng, max_steps, output_root, vis=None,
                ee_balancer=None):
    """Run one arm-only episode and write its transition CSV."""
    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_steering = 0.0
    driver_inputs.m_braking = 1.0  # base stays braked (and the chassis is pinned)

    n_sub = max(1, int(round(CONTROL_DT / STEP_SIZE)))

    start_q, start_qd = actuator.read_state()
    start_qcmd = list(actuator.qcmd)
    start_hit, start_kind, start_links, start_force = arm_contact(collision_links)
    if start_hit:
        raise RuntimeError(
            f"{episode_id} fresh home state starts in {start_kind} contact "
            f"({'+'.join(start_links)}), force={start_force:.3f} N"
        )

    sampler = SmoothCommandSampler(rng)
    split = assign_split(episode_id, validation_ratio)
    csv_path = output_root / "episodes" / f"{episode_id}.csv"
    base_frame = gripper.base.GetFrameRefToAbs()

    rows = 0
    terminated = False
    term_kind = None
    term_links = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for i in range(max_steps):
            t = m113.GetSystem().GetChTime()
            q, qd = actuator.read_state()
            qcmd = list(actuator.qcmd)
            ee = gripper_center(gripper)
            ee_b = base_frame.TransformPointParentToLocal(ee)

            action = sampler.next_action(qcmd)
            qcmd_next = clip_pose([qcmd[j] + action[j] for j in range(4)])
            applied = [qcmd_next[j] - qcmd[j] for j in range(4)]

            actuator.qcmd = qcmd_next
            _substep(m113, terrain, actuator, driver_inputs, n_sub, vis=vis)

            q_next, qd_next = actuator.read_state()
            ee_next = gripper_center(gripper)
            ee_next_b = base_frame.TransformPointParentToLocal(ee_next)
            hit, kind, involved, force = arm_contact(collision_links)
            limit_hit, limit_involved = joint_limit_violation(q_next)
            if limit_hit and not hit:
                hit = True
                kind = "joint_limit"
                involved = limit_involved

            row = {
                "episode_id": episode_id, "split": split, "sample_index": i,
                "time_s": t, "collision": int(hit), "collision_kind": kind or "",
                "contact_force_n": force,
                "ee_x": ee.x, "ee_y": ee.y, "ee_z": ee.z,
                "ee_base_x": ee_b.x, "ee_base_y": ee_b.y, "ee_base_z": ee_b.z,
                "ee_next_x": ee_next.x, "ee_next_y": ee_next.y, "ee_next_z": ee_next.z,
                "ee_next_base_x": ee_next_b.x, "ee_next_base_y": ee_next_b.y,
                "ee_next_base_z": ee_next_b.z,
            }
            for j in range(4):
                row[f"q_{j}"] = q[j]
                row[f"qd_{j}"] = qd[j]
                row[f"qcmd_{j}"] = qcmd[j]
                row[f"act_{j}"] = applied[j]
                row[f"qcmd_next_{j}"] = qcmd_next[j]
                row[f"q_next_{j}"] = q_next[j]
                row[f"qd_next_{j}"] = qd_next[j]
            write_row = hit or ee_balancer is None or ee_balancer.accept(ee_b)
            if write_row:
                writer.writerow(row)
                rows += 1

            if hit:
                terminated = True
                term_kind = kind
                term_links = involved
                break

            if vis is not None and not vis.Run():
                break

    meta = {
        "episode_id": episode_id, "split": split, "rows": rows,
        "terminated_collision": terminated, "collision_kind": term_kind,
        "collision_links": term_links,
        "control_dt_s": CONTROL_DT, "step_size_s": STEP_SIZE, "max_steps": max_steps,
        "start_q": start_q, "start_qd": start_qd, "start_qcmd": start_qcmd,
    }
    with (output_root / "episodes" / f"{episode_id}.json").open("w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2)

    return EpisodeResult(
        episode_id, split, rows, terminated, term_kind, term_links, csv_path,
        start_q, start_qd, start_qcmd,
    )


def build_and_prepare(render=False):
    """Build the M113+arm scene, swap to PD motors, add arm collision, settle.

    Returns (m113, vehicle, terrain, gripper, actuator, collision_links, vis).
    """
    m113, vehicle, terrain, gripper = build_scene()
    vis = make_vis(vehicle) if render else None
    if vis is not None:
        vehicle.EnableRealtime(True)  # cap playback to real time so it is watchable

    actuator = ArmPdActuator(gripper)
    collision_links = setup_arm_collision(gripper)

    # Settle on the tracks (arm held at home) then pin the base so subsequent
    # data is arm-only (base pose fixed, base velocity 0 -- doc 6.1).
    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_steering = 0.0
    driver_inputs.m_braking = 1.0
    n_sub = max(1, int(round(CONTROL_DT / STEP_SIZE)))
    for _ in range(int(round(SETTLE_TIME / CONTROL_DT))):
        _substep(m113, terrain, actuator, driver_inputs, n_sub, vis=vis)
    vehicle.GetChassisBody().SetFixed(True)

    return m113, vehicle, terrain, gripper, actuator, collision_links, vis


def collect(episodes, max_steps, seed, output_dir, validation_ratio, render=False,
            dataset_name="arm_dynamics_v1", episode_prefix="arm_ep",
            ee_balance_grid=None, ee_bin_cap=0, ee_bounds_lo=None, ee_bounds_hi=None):
    import random

    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = repo_root_from_module() / output_root
    (output_root / "episodes").mkdir(parents=True, exist_ok=True)

    print(f"  collision links/families: {LINK_FAMILIES}")
    print(f"  termination on: track-shoe (family {TRACK_SHOE_FAMILY}) | self (non-adjacent links) "
          f"| ground (box z <= {GROUND_PLANE_Z + GROUND_CONTACT_MARGIN:.2f} m) | joint limits")
    print("  episode reset: fresh Chrono scene/home pose per episode; q/qdot are measured from motors")
    ee_balancer = None
    if ee_balance_grid is not None and ee_bin_cap > 0:
        ee_balancer = EeBinBalancer(
            ee_balance_grid,
            ee_bin_cap,
            ee_bounds_lo or (-6.0, -6.0, -5.5),
            ee_bounds_hi or (6.0, 6.0, 1.5),
        )
        print("  EE row balancing: "
              f"grid={ee_balancer.grid}, cap/bin={ee_balancer.bin_cap}, "
              f"bounds={ee_balancer.bounds_lo}->{ee_balancer.bounds_hi}")

    results = []
    for ep in range(episodes):
        episode_id = f"{episode_prefix}_{ep:04d}"
        rng = random.Random(seed + ep)
        m113, vehicle, terrain, gripper, actuator, collision_links, vis = build_and_prepare(render)
        result = run_episode(m113, vehicle, terrain, gripper, actuator, collision_links,
                             episode_id, validation_ratio, rng, max_steps, output_root,
                             vis=vis, ee_balancer=ee_balancer)
        if result.terminated_collision:
            flag = f"{result.collision_kind.upper()}-COLLISION@{'+'.join(result.collision_links)}"
        else:
            flag = "full-length"
        start_q_fmt = ", ".join(f"{v:.3f}" for v in result.start_q)
        start_qd_fmt = ", ".join(f"{v:.3f}" for v in result.start_qd)
        print(f"  {episode_id}: {result.rows} rows, {flag}, split={result.split}, "
              f"start_q=[{start_q_fmt}], start_qd=[{start_qd_fmt}]")
        results.append(result)
        keep_running = vis is None or vis.Run()
        del m113, vehicle, terrain, gripper, actuator, collision_links, vis
        gc.collect()
        if not keep_running:
            break

    summary = {
        "dataset_name": dataset_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(results),
        "config": {
            "control_dt_s": CONTROL_DT, "step_size_s": STEP_SIZE,
            "joint_limits_lo": JOINT_LIMITS_LO, "joint_limits_hi": JOINT_LIMITS_HI,
            "dq_max": DQ_MAX, "pd_kp": PD_KP, "pd_kd": PD_KD, "pd_tau_max": PD_TAU_MAX,
            "link_families": LINK_FAMILIES,
            "track_shoe_family": TRACK_SHOE_FAMILY,
            "adjacent_link_pairs": [sorted(p) for p in ADJACENT_LINK_PAIRS],
            "contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
            "ground_plane_z": GROUND_PLANE_Z,
            "ground_contact_margin": GROUND_CONTACT_MARGIN,
            "joint_limit_eps": JOINT_LIMIT_EPS,
            "collision_links": list(LINK_FAMILIES),
            "arm_scale": ARM_SCALE,
            "episode_reset": {
                "mode": "fresh_scene_home_each_episode",
                "random_start_pose": False,
                "qcmd_start": [0.0, 0.0, 0.0, 0.0],
                "state_source": "ChLinkMotorRotationTorque.GetMotorAngle/GetMotorAngleDt",
            },
            "ee_balance": ee_balancer.summary() if ee_balancer else {"enabled": False},
        },
        "episodes": [
            {"episode_id": r.episode_id, "split": r.split, "rows": r.rows,
             "terminated_collision": r.terminated_collision,
             "collision_kind": r.collision_kind, "collision_links": r.collision_links,
             "start_q": r.start_q, "start_qd": r.start_qd, "start_qcmd": r.start_qcmd,
             "csv_path": str(r.csv_path.relative_to(output_root))}
            for r in results
        ],
    }
    with (output_root / "dataset_index.json").open("w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2)

    total = sum(r.rows for r in results)
    n_track = sum(1 for r in results if r.collision_kind == "track")
    n_ground = sum(1 for r in results if r.collision_kind == "ground")
    n_self = sum(1 for r in results if r.collision_kind == "self")
    n_limit = sum(1 for r in results if r.collision_kind == "joint_limit")
    n_full = sum(1 for r in results if not r.terminated_collision)
    print(f"wrote {len(results)} episodes, {total} transitions "
          f"(terminated by: {n_track} track, {n_ground} ground, {n_self} self; "
          f"{n_limit} joint-limit; {n_full} full-length) to {output_root}")
    return results


def _parse_csv_tuple(value, cast, expected_len, name):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must have {expected_len} comma-separated values")
    try:
        return tuple(cast(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {value}") from exc


def parse_int3(value):
    return _parse_csv_tuple(value, int, 3, "int3")


def parse_float3(value):
    return _parse_csv_tuple(value, float, 3, "float3")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect arm-only dynamics data (reach mode).")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max recorded control steps per episode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/arm_data",
                        help="Output root (relative to repo root unless absolute).")
    parser.add_argument("--dataset-name", default="arm_dynamics_v1")
    parser.add_argument("--episode-prefix", default="arm_ep",
                        help="Prefix used for episode IDs inside this output shard.")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--ee-balance-grid", type=parse_int3, default=None,
                        metavar="NX,NY,NZ",
                        help="Enable measured gripper-base EE row balancing with this grid.")
    parser.add_argument("--ee-bin-cap", type=int, default=0,
                        help="Max free-space rows written per EE grid bin; <=0 disables balancing.")
    parser.add_argument("--ee-bounds-lo", type=parse_float3, default=(-6.0, -6.0, -5.5),
                        metavar="X,Y,Z",
                        help="Lower gripper-base EE bounds for row balancing.")
    parser.add_argument("--ee-bounds-hi", type=parse_float3, default=(6.0, 6.0, 1.5),
                        metavar="X,Y,Z",
                        help="Upper gripper-base EE bounds for row balancing.")
    parser.add_argument("--render", action="store_true",
                        help="Open the Irrlicht viewer (debug; runs one window).")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    print("=== Arm-only dynamics data collection ===")
    collect(episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
            output_dir=args.output_dir, validation_ratio=args.validation_ratio,
            render=args.render, dataset_name=args.dataset_name,
            episode_prefix=args.episode_prefix, ee_balance_grid=args.ee_balance_grid,
            ee_bin_cap=args.ee_bin_cap, ee_bounds_lo=args.ee_bounds_lo,
            ee_bounds_hi=args.ee_bounds_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
