"""
kinematics.py -- forward kinematics, gravity torques and inverse kinematics.

Everything here is pure kinematics/statics on a scratch MjData: no simulation
step is taken.  Joint vectors are 19-long, in the SDK motor order
(model_builder.MOTOR_JOINTS); the right arm is entries 12..18.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.model_builder import ARM_SLICE, Robot


def _set_config(robot: Robot, data: mujoco.MjData, q19: np.ndarray) -> None:
    data.qpos[:] = 0.0                      # fingers open / at zero
    data.qpos[robot.qadr] = q19
    data.qvel[:] = 0.0
    mujoco.mj_kinematics(robot.model, data)
    mujoco.mj_comPos(robot.model, data)


def site_pose(robot: Robot, data: mujoco.MjData):
    """World pose of the TCP site from an already-updated MjData: (pos, quat wxyz)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[robot.tcp_site_id])
    return data.site_xpos[robot.tcp_site_id].copy(), quat


def fk(robot: Robot, data: mujoco.MjData, q19: np.ndarray):
    """TCP pose for a motor configuration: (pos (3,), quat wxyz (4,))."""
    _set_config(robot, data, q19)
    return site_pose(robot, data)


def gravity_torque(robot: Robot, data: mujoco.MjData, q19: np.ndarray) -> np.ndarray:
    """Static joint torques (19,) needed to hold configuration q19 against gravity,
    from recursive Newton-Euler on this robot's own mass model."""
    _set_config(robot, data, q19)
    mujoco.mj_comVel(robot.model, data)
    bias = np.zeros(robot.model.nv)
    mujoco.mj_rne(robot.model, data, 0, bias)
    return bias[robot.dadr]


def rotation_error_vec(quat_target: np.ndarray, quat_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector (axis * angle, rad) that takes the current
    orientation onto the target orientation."""
    conj = np.zeros(4)
    mujoco.mju_negQuat(conj, quat_current)
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, quat_target, conj)
    if q_err[0] < 0:                        # shortest rotation
        q_err = -q_err
    vec = np.zeros(3)
    mujoco.mju_quat2Vel(vec, q_err, 1.0)
    return vec


def orientation_angle(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Angle (rad) of the single rotation between two orientations."""
    return float(np.linalg.norm(rotation_error_vec(quat_a, quat_b)))


@dataclass
class IKResult:
    q_arm: np.ndarray           # (7,)
    pos_error_m: float
    rot_error_rad: float
    iterations: int
    converged: bool
    min_limit_margin_rad: float


def solve_ik(robot: Robot, data: mujoco.MjData, target_pos, target_quat, q_seed_arm,
             q_posture19=None, max_iters: int = 300, pos_tol: float = 1e-7,
             rot_tol: float = 1e-7, max_step: float = 0.15) -> IKResult:
    """Damped least-squares IK for the 7-DOF right arm (6-D task, 1 redundant DOF).

    The damping is proportional to the remaining error (Levenberg-Marquardt
    style): large far from the target, where it keeps steps safe near
    singularities, and vanishing close to it, so the residual can be driven far
    below the errors the benchmark is trying to measure.  The redundant DOF is
    not optimised: the solver returns the solution nearest to its seed, and the
    experiment always seeds from the same configuration so that a target always
    maps to the same joint command.
    """
    model = robot.model
    q19 = np.zeros(19) if q_posture19 is None else np.array(q_posture19, dtype=float)
    q = np.array(q_seed_arm, dtype=float)
    lo, hi = robot.arm_range[:, 0], robot.arm_range[:, 1]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)

    converged = False
    pos_err = rot_err = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        q19[ARM_SLICE] = q
        pos, quat = fk(robot, data, q19)
        e_pos = target_pos - pos
        e_rot = rotation_error_vec(target_quat, quat)
        pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
        if pos_err < pos_tol and rot_err < rot_tol:
            converged = True
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, robot.tcp_site_id)
        J = np.vstack([jacp[:, robot.arm_dadr], jacr[:, robot.arm_dadr]])     # (6, 7)
        e = np.concatenate([e_pos, e_rot])
        lam2 = 1e-2 * float(e @ e) + 1e-12
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
        biggest = np.abs(dq).max()
        if biggest > max_step:
            dq *= max_step / biggest
        q = np.clip(q + dq, lo, hi)

    margin = float(np.minimum(q - lo, hi - q).min())
    return IKResult(q, pos_err, rot_err, it, converged, margin)


def static_equilibrium(plant: Robot, data: mujoco.MjData, q_cmd19: np.ndarray, tau_ff19: np.ndarray,
                       kp19: np.ndarray, encoder_offset19: np.ndarray, iters: int = 30) -> np.ndarray:
    """Joint configuration at which the PD law balances gravity, with no friction:

        tau_ff + kp * (q_cmd - (q + offset)) - g_plant(q) = 0

    Solved with Newton's method (finite-difference gravity gradient).  This is an
    independent, simulation-free prediction of where a trial should end; the
    experiment compares it with the simulated result.
    """
    q = q_cmd19 - encoder_offset19
    n, eps = q.size, 1e-6
    for _ in range(iters):
        g0 = gravity_torque(plant, data, q)
        r = tau_ff19 + kp19 * (q_cmd19 - q - encoder_offset19) - g0
        if np.abs(r).max() < 1e-11:
            break
        dg = np.zeros((n, n))
        for j in range(n):
            qj = q.copy()
            qj[j] += eps
            dg[:, j] = (gravity_torque(plant, data, qj) - g0) / eps
        q = q + np.linalg.solve(np.diag(kp19) + dg, r)
    return q
