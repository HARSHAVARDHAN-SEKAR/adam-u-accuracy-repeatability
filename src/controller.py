"""
controller.py -- the joint controller used to command the arm.

ADAM-U's 19 motors are pure torque actuators in the MJCF: no position servo,
no gravity compensation.  The vendor simulator closes the loop in its SDK
bridge (simulate_python/pndbotics_sdk_py_bridge.py, LowCmdHandler):

    ctrl[i] = tau[i] + kp[i] * (q_cmd[i] - q_meas[i]) + kd[i] * (0 - dq_meas[i])

This module applies exactly that law to all 19 motors, with the gains of the
vendor example (config/experiment.yaml), once per simulation step.  The only
choice left to the user of that interface is the feed-forward torque `tau`:

    mode "pd"      tau = 0                       (the vendor example as shipped)
    mode "pd_gff"  tau = g(q_cmd)                (gravity torque at the commanded
                                                  configuration, from the nominal model)

The vendor bridge is not used directly because it runs over DDS with wall-clock
threads, which makes runs non-repeatable; stepping MuJoCo ourselves with the
same control law keeps every trial deterministic.

The controller never sees the true joint state, only "encoder" readings, which
can carry a fixed offset and noise (right arm only; see perturbations.py).
"""
from __future__ import annotations

import mujoco
import numpy as np

from src.kinematics import gravity_torque
from src.model_builder import ARM_SLICE, MOTOR_JOINTS, Robot
from src.perturbations import Condition

CONTROL_MODES = ("pd", "pd_gff")


def gains_from_config(gain_cfg: dict):
    missing = [j for j in MOTOR_JOINTS if j not in gain_cfg]
    if missing:
        raise ValueError(f"no gains configured for {missing}")
    kp = np.array([gain_cfg[j][0] for j in MOTOR_JOINTS], dtype=float)
    kd = np.array([gain_cfg[j][1] for j in MOTOR_JOINTS], dtype=float)
    return kp, kd


class SdkJointController:
    def __init__(self, nominal: Robot, kp: np.ndarray, kd: np.ndarray, mode: str):
        if mode not in CONTROL_MODES:
            raise ValueError(f"control mode must be one of {CONTROL_MODES}")
        self.mode = mode
        self.kp, self.kd = kp, kd
        self._nominal = nominal                       # the controller's model of the robot
        self._scratch = mujoco.MjData(nominal.model)

    def feedforward(self, q_cmd19: np.ndarray) -> np.ndarray:
        if self.mode == "pd":
            return np.zeros(19)
        return gravity_torque(self._nominal, self._scratch, q_cmd19)

    def torque(self, plant: Robot, data: mujoco.MjData, q_cmd19: np.ndarray, tau_ff19: np.ndarray,
               condition: Condition, rng: np.random.Generator | None) -> np.ndarray:
        q_meas = data.qpos[plant.qadr].copy()
        dq_meas = data.qvel[plant.dadr].copy()
        q_meas[ARM_SLICE] += condition.encoder_offset_rad
        if rng is not None and condition.encoder_noise_rad > 0.0:
            q_meas[ARM_SLICE] += rng.normal(0.0, condition.encoder_noise_rad, 7)
        if rng is not None and condition.velocity_noise_rad_s > 0.0:
            dq_meas[ARM_SLICE] += rng.normal(0.0, condition.velocity_noise_rad_s, 7)
        tau = tau_ff19 + self.kp * (q_cmd19 - q_meas) - self.kd * dq_meas
        return np.clip(tau, -plant.torque_limit, plant.torque_limit)
