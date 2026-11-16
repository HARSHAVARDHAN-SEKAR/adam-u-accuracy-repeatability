"""
experiment.py -- shared set-up: configuration, robot models, targets and seeds.
Used by run_experiment.py, the tests and the helper scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

from src.controller import SdkJointController, gains_from_config
from src.kinematics import fk, solve_ik, static_equilibrium
from src.model_builder import ARM_SLICE, Robot, build_robot
from src.perturbations import Condition, condition_from_config

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APPROACHES = {"plus": +1.0, "minus": -1.0}


def load_config(path: str | None = None) -> dict:
    with open(path or os.path.join(REPO_ROOT, "config", "experiment.yaml")) as f:
        return yaml.safe_load(f)


def resolve_model_path(cfg: dict, override: str | None = None) -> str:
    path = override or os.environ.get("ADAM_U_SCENE_XML") or cfg["model"]["scene_xml"]
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


@dataclass
class Target:
    name: str
    pos: np.ndarray            # commanded TCP position, world frame (m)
    quat: np.ndarray           # commanded TCP orientation (wxyz)
    q_cmd: np.ndarray          # (7,) joint command from IK
    ik_residual_m: float
    ik_residual_rad: float
    limit_margin_rad: float


class Setup:
    """Everything a trial needs, built once."""

    def __init__(self, cfg: dict | None = None, model_path: str | None = None):
        self.cfg = cfg or load_config()
        self.model_path = resolve_model_path(self.cfg, model_path)
        tcp = self.cfg["model"]["tcp"]
        self._tcp = (tcp["body"], tuple(tcp["offset_m"]))
        self._plants: dict[float, tuple[Robot, mujoco.MjData]] = {}
        self.nominal = self._build(0.0)
        self.scratch = mujoco.MjData(self.nominal.model)
        self.kp, self.kd = gains_from_config(self.cfg["controller"]["gains"])
        self._controllers: dict[str, SdkJointController] = {}
        self._conditions: dict[str, Condition] = {}
        self._static: dict[tuple, np.ndarray] = {}
        self.targets = self._make_targets()

    def _build(self, payload_kg: float) -> Robot:
        robot = build_robot(self.model_path, self._tcp[0], self._tcp[1], payload_kg)
        self._plants[round(payload_kg, 9)] = (robot, mujoco.MjData(robot.model))
        return robot

    def plant_for(self, condition: Condition):
        """(robot, data) of the simulated plant for this condition. The plant
        with a payload is a separate compiled model; the controller keeps
        using the nominal one."""
        key = round(condition.payload_kg, 9)
        if key not in self._plants:
            self._build(condition.payload_kg)
        return self._plants[key]

    def controller(self, mode: str) -> SdkJointController:
        if mode not in self._controllers:
            self._controllers[mode] = SdkJointController(self.nominal, self.kp, self.kd, mode)
        return self._controllers[mode]

    def condition(self, name: str) -> Condition:
        if name not in self._conditions:
            if name not in self.cfg["conditions"]:
                raise KeyError(f"condition '{name}' is not defined in the config")
            self._conditions[name] = condition_from_config(name, self.cfg["conditions"][name])
        return self._conditions[name]

    def _make_targets(self) -> list[Target]:
        tcfg = self.cfg["targets"]
        centre = np.array(tcfg["centre_config_rad"], dtype=float)
        q19 = np.zeros(19)
        q19[ARM_SLICE] = centre
        p0, quat0 = fk(self.nominal, self.scratch, q19)
        a = 0.4 * float(tcfg["cube_side_m"])       # corners inset by 10 % of the side
        offsets = {"P1": (0, 0, 0), "P2": (a, a, a), "P3": (a, -a, a), "P4": (-a, -a, -a), "P5": (-a, a, -a)}
        d = np.array(tcfg["approach_offset_rad"], dtype=float)
        lo, hi = self.nominal.arm_range[:, 0], self.nominal.arm_range[:, 1]
        targets = []
        for name, off in offsets.items():
            pos = p0 + np.array(off, dtype=float)
            ik = solve_ik(self.nominal, self.scratch, pos, quat0, centre)     # always seeded from the centre
            if not ik.converged:
                raise RuntimeError(f"target {name} is not reachable (IK residual {ik.pos_error_m:.2e} m)")
            if ((ik.q_arm + d) > hi).any() or ((ik.q_arm - d) < lo).any():
                raise RuntimeError(f"start configuration of target {name} violates a joint limit")
            targets.append(Target(name, pos, quat0.copy(), ik.q_arm, ik.pos_error_m, ik.rot_error_rad,
                                  ik.min_limit_margin_rad))
        self.approach_offset = d
        return targets

    def start_config(self, target: Target, approach: str) -> np.ndarray:
        return target.q_cmd + APPROACHES[approach] * self.approach_offset

    def static_prediction(self, target: Target, condition: Condition, mode: str) -> np.ndarray:
        """TCP position predicted without simulating: the joint configuration
        where the PD law balances gravity on this plant (payload and encoder
        offset included, friction and noise not), pushed through forward
        kinematics.  The difference between a trial and this prediction is what
        statics cannot explain: friction, noise, incomplete settling."""
        key = (target.name, condition.name, mode)
        if key not in self._static:
            plant, data = self.plant_for(condition)
            q_cmd19 = np.zeros(19)
            q_cmd19[ARM_SLICE] = target.q_cmd
            off19 = np.zeros(19)
            off19[ARM_SLICE] = condition.encoder_offset_rad
            tau_ff = self.controller(mode).feedforward(q_cmd19)
            q_eq = static_equilibrium(plant, data, q_cmd19, tau_ff, self.kp, off19)
            self._static[key] = fk(plant, data, q_eq)[0]
        return self._static[key]


def trial_seed(base_seed: int, *indices: int) -> np.random.SeedSequence:
    """Seed that depends only on what the trial is, never on when it runs."""
    return np.random.SeedSequence([int(base_seed), *[int(i) for i in indices]])
