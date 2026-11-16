"""trajectory.py -- minimum-jerk point-to-point motion in joint space."""
from __future__ import annotations

import numpy as np


def min_jerk_scale(t: float, duration: float) -> float:
    """s(t) in [0, 1] with zero velocity and acceleration at both ends."""
    if duration <= 0.0:
        return 1.0
    s = min(max(t / duration, 0.0), 1.0)
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def joint_setpoint(q_start: np.ndarray, q_goal: np.ndarray, t: float, duration: float) -> np.ndarray:
    return q_start + (q_goal - q_start) * min_jerk_scale(t, duration)
