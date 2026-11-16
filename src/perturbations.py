"""
perturbations.py -- the plant conditions ("what the real robot does that the
ideal simulator does not").  Values live in config/experiment.yaml.

  encoder_offset   fixed error of each joint's zero position.  The controller
                   believes q + offset, so the joint really stops at
                   q_command - offset.  Purely systematic.
  payload          point mass at the TCP that the controller's gravity model
                   does not contain.  Systematic, depends on the pose.
  friction         Coulomb friction (MuJoCo "frictionloss") in each arm joint.
                   The joint stops wherever |kp * error| drops below the
                   friction torque, so the result depends on the approach path.
  encoder noise    white noise on the measured joint position and velocity.
  start jitter     the trial starts from a slightly different configuration
                   each time (models whatever the robot did before).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_ARM = 7


def _vec(value, n=N_ARM) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value if value is not None else 0.0, dtype=float))
    if arr.size == 1:
        arr = np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"expected a scalar or {n} values, got {arr.size}")
    return arr


@dataclass(frozen=True)
class Condition:
    name: str
    encoder_offset_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_ARM))
    payload_kg: float = 0.0
    friction_nm: np.ndarray = field(default_factory=lambda: np.zeros(N_ARM))
    encoder_noise_rad: float = 0.0
    velocity_noise_rad_s: float = 0.0
    start_jitter_rad: float = 0.0

    @property
    def is_stochastic(self) -> bool:
        return self.encoder_noise_rad > 0 or self.velocity_noise_rad_s > 0 or self.start_jitter_rad > 0


def condition_from_config(name: str, cfg: dict | None) -> Condition:
    cfg = cfg or {}
    known = {"encoder_offset_deg", "payload_kg", "friction_nm", "encoder_noise_deg",
             "velocity_noise_rad_s", "start_jitter_deg"}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"condition '{name}': unknown keys {sorted(unknown)}")
    return Condition(
        name=name,
        encoder_offset_rad=np.radians(_vec(cfg.get("encoder_offset_deg"))),
        payload_kg=float(cfg.get("payload_kg", 0.0)),
        friction_nm=_vec(cfg.get("friction_nm")),
        encoder_noise_rad=float(np.radians(cfg.get("encoder_noise_deg", 0.0))),
        velocity_noise_rad_s=float(cfg.get("velocity_noise_rad_s", 0.0)),
        start_jitter_rad=float(np.radians(cfg.get("start_jitter_deg", 0.0))),
    )
