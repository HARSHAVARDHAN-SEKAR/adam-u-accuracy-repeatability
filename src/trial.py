"""
trial.py -- one measurement cycle:

    reset to the start configuration -> pre-hold -> minimum-jerk move to the
    commanded configuration -> dwell -> read the TCP pose.

Durations are fixed (config: protocol), so every trial simulates the same
amount of time and the measurement instant never depends on a threshold.
The TCP pose is MuJoCo ground truth (the site's world pose), i.e. an ideal
external measuring device.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from src.controller import SdkJointController
from src.kinematics import site_pose
from src.model_builder import ARM_SLICE, Robot
from src.perturbations import Condition
from src.trajectory import joint_setpoint

TRACE_EVERY = 10          # steps between TCP samples kept for the stabilisation time


@dataclass
class TrialResult:
    tcp_pos: np.ndarray            # (3,) m, world frame, at the end of the dwell
    tcp_quat: np.ndarray           # (4,) wxyz
    q_true19: np.ndarray           # true joint positions at the measurement instant
    stabilisation_time_s: float    # after the commanded motion ends
    peak_torque_fraction: float    # max |tau| / limit over the right arm, move + dwell
    saturated: bool
    joint_contrib_m: np.ndarray    # (19, 3) first-order TCP displacement caused by each joint's error
    trace_t: np.ndarray            # dwell trace, for plots
    trace_dev_mm: np.ndarray


def apply_plant_condition(plant: Robot, condition: Condition, sim_cfg: dict | None = None) -> None:
    """Friction is a runtime model parameter; payload is baked in at build time.

    With MuJoCo's default settings joint friction is a soft constraint and the
    joints creep, so for plants with friction the no-slip solver is enabled
    (see config: simulation). Plants without friction keep the model defaults.
    """
    if abs(plant.payload_mass_kg - condition.payload_kg) > 1e-12:
        raise ValueError("plant model was built with a different payload than the condition asks for")
    opt = plant.model.opt
    if not hasattr(plant, "_default_noslip"):
        plant._default_noslip = (int(opt.noslip_iterations), float(opt.noslip_tolerance))
    plant.model.dof_frictionloss[plant.arm_dadr] = condition.friction_nm
    if np.any(condition.friction_nm > 0.0) and sim_cfg:
        opt.noslip_iterations = int(sim_cfg["friction_noslip_iterations"])
        opt.noslip_tolerance = float(sim_cfg["friction_noslip_tolerance"])
    else:
        opt.noslip_iterations, opt.noslip_tolerance = plant._default_noslip


def run_trial(plant: Robot, data: mujoco.MjData, controller: SdkJointController, condition: Condition,
              q_start_arm: np.ndarray, q_cmd_arm: np.ndarray, protocol: dict,
              rng: np.random.Generator | None, sim_cfg: dict | None = None, on_step=None) -> TrialResult:
    model = plant.model
    dt = model.opt.timestep
    apply_plant_condition(plant, condition, sim_cfg)

    q_start19 = np.zeros(19)
    q_start19[ARM_SLICE] = q_start_arm
    q_goal19 = np.zeros(19)
    q_goal19[ARM_SLICE] = q_cmd_arm

    # --- reset: the arm is placed at the start configuration (plus jitter), at rest ---
    mujoco.mj_resetData(model, data)
    q_init = q_start19.copy()
    if rng is not None and condition.start_jitter_rad > 0.0:
        q_init[ARM_SLICE] += rng.normal(0.0, condition.start_jitter_rad, 7)
        q_init[ARM_SLICE] = np.clip(q_init[ARM_SLICE], plant.arm_range[:, 0], plant.arm_range[:, 1])
    data.qpos[plant.qadr] = q_init
    mujoco.mj_forward(model, data)

    arm_limit = plant.torque_limit[ARM_SLICE]
    peak = 0.0

    def step(q_cmd19, tau_ff19, track_peak):
        nonlocal peak
        tau = controller.torque(plant, data, q_cmd19, tau_ff19, condition, rng)
        data.ctrl[plant.act] = tau
        mujoco.mj_step(model, data)
        if track_peak:
            peak = max(peak, float(np.max(np.abs(tau[ARM_SLICE]) / arm_limit)))
        if on_step is not None:
            on_step()

    # --- pre-hold ---
    ff = controller.feedforward(q_start19)
    for _ in range(int(round(protocol["pre_hold_s"] / dt))):
        step(q_start19, ff, False)

    # --- move ---
    n_move = int(round(protocol["move_s"] / dt))
    for k in range(n_move):
        q_cmd19 = joint_setpoint(q_start19, q_goal19, (k + 1) * dt, protocol["move_s"])
        step(q_cmd19, controller.feedforward(q_cmd19), True)

    # --- dwell ---
    ff = controller.feedforward(q_goal19)
    n_dwell = int(round(protocol["dwell_s"] / dt))
    trace_t, trace_p = [], []
    for k in range(n_dwell):
        step(q_goal19, ff, True)
        if (k + 1) % TRACE_EVERY == 0:
            trace_t.append((k + 1) * dt)
            trace_p.append(data.site_xpos[plant.tcp_site_id].copy())

    # --- measure ---
    mujoco.mj_kinematics(model, data)          # make the site pose consistent with the final qpos
    pos, quat = site_pose(plant, data)
    q_true19 = data.qpos[plant.qadr].copy()

    trace_t = np.array(trace_t)
    dev_mm = np.linalg.norm(np.array(trace_p) - pos, axis=1) * 1000.0
    outside = np.nonzero(dev_mm > protocol["stabilisation_band_mm"])[0]
    stab = 0.0 if outside.size == 0 else float(trace_t[min(outside[-1] + 1, len(trace_t) - 1)])

    jacp = np.zeros((3, model.nv))
    mujoco.mj_comPos(model, data)
    mujoco.mj_jacSite(model, data, jacp, None, plant.tcp_site_id)
    contrib = (jacp[:, plant.dadr] * (q_true19 - q_goal19)).T          # (19, 3)

    return TrialResult(pos, quat, q_true19, stab, peak, bool(peak >= 0.999), contrib, trace_t, dev_mm)
