"""
model_builder.py -- load the vendored ADAM-U MJCF and adapt it for the benchmark.

The vendored files (pnd_robots/adam_u/scene.xml + adam_u.xml) are never edited
on disk.  They are parsed, patched in memory and compiled from a string:

  1. The 6-DOF "floating_base" free joint is removed, so the lifting column is
     rigidly attached to the world.  The brief calls ADAM-U a fixed-base robot;
     as shipped, the model only rests on the floor through contact.
  2. A site "tcp_right" is added to the last body of the right arm
     ("wristRollRight").  The vendored model has no sites, so there is no
     tool frame to measure until we define one.
  3. Optionally a point-mass payload is welded at the TCP (perturbed plant).

Plain XML editing is used instead of the MjSpec API on purpose: MjSpec changed
between MuJoCo 3.2 and 4.x, while this approach runs unchanged on both.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

# The 19 motors, in the order of the PND low-level SDK (LowCmd.motor_cmd[i]),
# which is also the actuator order in the MJCF.
MOTOR_JOINTS = [
    "waistRoll", "waistPitch", "waistYaw",
    "neckYaw", "neckPitch",
    "shoulderPitch_Left", "shoulderRoll_Left", "shoulderYaw_Left", "elbow_Left",
    "wristYaw_Left", "wristPitch_Left", "wristRoll_Left",
    "shoulderPitch_Right", "shoulderRoll_Right", "shoulderYaw_Right", "elbow_Right",
    "wristYaw_Right", "wristPitch_Right", "wristRoll_Right",
]
RIGHT_ARM_JOINTS = MOTOR_JOINTS[12:19]
ARM_SLICE = slice(12, 19)          # right arm inside the 19-motor vector

TCP_SITE = "tcp_right"
PAYLOAD_BODY = "tcp_payload"


@dataclass
class Robot:
    """A compiled model plus the index arrays the rest of the code needs."""
    model: mujoco.MjModel
    tcp_site_id: int
    qadr: np.ndarray            # (19,) qpos address of every motor joint
    dadr: np.ndarray            # (19,) dof address of every motor joint
    act: np.ndarray             # (19,) actuator id of every motor joint
    torque_limit: np.ndarray    # (19,) N*m, from the actuator ctrlrange
    joint_range: np.ndarray     # (19, 2) rad
    payload_mass_kg: float = 0.0
    xml: str = field(default="", repr=False)

    @property
    def arm_qadr(self):
        return self.qadr[ARM_SLICE]

    @property
    def arm_dadr(self):
        return self.dadr[ARM_SLICE]

    @property
    def arm_range(self):
        return self.joint_range[ARM_SLICE]


def _find_parent(root, predicate):
    for parent in root.iter():
        for child in list(parent):
            if predicate(child):
                return parent, child
    return None, None


def build_patched_xml(scene_xml_path: str, tcp_body: str, tcp_offset, payload_mass_kg: float = 0.0) -> str:
    scene_xml_path = os.path.abspath(scene_xml_path)
    if not os.path.isfile(scene_xml_path):
        raise FileNotFoundError(
            f"ADAM-U scene not found at '{scene_xml_path}'.\n"
            "Run  bash scripts/setup_env.sh  to fetch the pinned pnd_mujoco checkout, "
            "or pass --model-path / set ADAM_U_SCENE_XML.")
    model_dir = os.path.dirname(scene_xml_path)
    scene_root = ET.parse(scene_xml_path).getroot()
    include = scene_root.find("include")
    if include is None:
        raise ValueError("expected scene.xml to <include> the robot file")
    robot_root = ET.parse(os.path.join(model_dir, include.get("file"))).getroot()

    # Mesh paths in the robot file are relative to its own folder.
    compiler = robot_root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    if compiler is None:
        compiler = ET.SubElement(robot_root, "compiler")
    compiler.set("meshdir", os.path.join(model_dir, meshdir))

    # 1. fixed base
    parent, joint = _find_parent(
        robot_root, lambda e: e.tag in ("joint", "freejoint") and
        (e.get("type") == "free" or e.tag == "freejoint"))
    if joint is None:
        raise ValueError("no free joint found; has the vendored model changed?")
    parent.remove(joint)

    # 2. TCP site (+ 3. optional payload)
    wrist = next((b for b in robot_root.iter("body") if b.get("name") == tcp_body), None)
    if wrist is None:
        raise ValueError(f"body '{tcp_body}' not found in the robot model")
    pos = " ".join(f"{v:.6f}" for v in tcp_offset)
    ET.SubElement(wrist, "site", name=TCP_SITE, pos=pos, size="0.008", rgba="1 0.1 0.1 1")
    if payload_mass_kg > 0.0:
        r = 0.03                                    # sphere radius used only for the inertia
        inertia = 0.4 * payload_mass_kg * r * r
        body = ET.SubElement(wrist, "body", name=PAYLOAD_BODY, pos=pos)
        ET.SubElement(body, "inertial", pos="0 0 0", mass=f"{payload_mass_kg:.6f}",
                      diaginertia=f"{inertia:.3e} {inertia:.3e} {inertia:.3e}")

    # Merge the scene (floor, light, options) into the robot document.
    for child in scene_root:
        if child.tag != "include":
            robot_root.append(child)
    return ET.tostring(robot_root, encoding="unicode")


def build_robot(scene_xml_path: str, tcp_body: str = "wristRollRight",
                tcp_offset=(0.0, 0.0, -0.09), payload_mass_kg: float = 0.0) -> Robot:
    xml = build_patched_xml(scene_xml_path, tcp_body, tcp_offset, payload_mass_kg)
    model = mujoco.MjModel.from_xml_string(xml)

    def jid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise ValueError(f"joint '{name}' not found")
        return i

    jids = [jid(n) for n in MOTOR_JOINTS]
    act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in MOTOR_JOINTS])
    if (act < 0).any():
        raise ValueError("a motor actuator is missing; has the vendored model changed?")
    return Robot(
        model=model,
        tcp_site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE),
        qadr=np.array([model.jnt_qposadr[j] for j in jids]),
        dadr=np.array([model.jnt_dofadr[j] for j in jids]),
        act=act,
        torque_limit=np.abs(model.actuator_ctrlrange[act]).max(axis=1),
        joint_range=np.array([model.jnt_range[j] for j in jids]),
        payload_mass_kg=payload_mass_kg,
        xml=xml,
    )
