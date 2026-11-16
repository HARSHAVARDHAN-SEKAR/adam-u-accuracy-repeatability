#!/usr/bin/env python3
"""
check_setup.py -- load the model and print what the experiment will use.
Run this first: it fails loudly if the model is missing or a target is unreachable.

    python3 scripts/check_setup.py [--model-path PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.experiment import APPROACHES, Setup  # noqa: E402
from src.kinematics import gravity_torque  # noqa: E402
from src.model_builder import ARM_SLICE, MOTOR_JOINTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    s = Setup(model_path=args.model_path)
    m = s.nominal.model
    print(f"MuJoCo {mujoco.__version__}   model: {s.model_path}")
    print(f"degrees of freedom: {m.nv} (free joint removed: {m.nq == m.nv})   actuators: {m.nu}   "
          f"timestep: {m.opt.timestep} s")
    print(f"TCP site 'tcp_right' on body '{s.cfg['model']['tcp']['body']}', offset {s.cfg['model']['tcp']['offset_m']} m\n")

    q19 = np.zeros(19)
    q19[ARM_SLICE] = s.targets[0].q_cmd
    g = gravity_torque(s.nominal, s.scratch, q19)
    print("motor                  kp      kd   limit(Nm)  gravity torque at P1 (Nm)  droop if PD only (deg)")
    for i, name in enumerate(MOTOR_JOINTS):
        print(f"{name:20s} {s.kp[i]:6.1f} {s.kd[i]:6.2f} {s.nominal.torque_limit[i]:9.1f} {g[i]:18.3f} "
              f"{np.degrees(g[i] / s.kp[i]):22.2f}")
    print("  (droop = gravity torque / kp, a first estimate; the experiment measures the real value)\n")

    print("target   x       y       z (m)     IK residual (m)  limit margin (deg)   joint command (rad)")
    for t in s.targets:
        print(f"{t.name}   {t.pos[0]:7.4f} {t.pos[1]:7.4f} {t.pos[2]:7.4f}   {t.ik_residual_m:10.1e} "
              f"{np.degrees(t.limit_margin_rad):14.1f}        {np.round(t.q_cmd, 3)}")
    print(f"\napproach offset (rad): {s.approach_offset}  ->  approaches {list(APPROACHES)}")
    print("\nstatic prediction of the position error (mm), before simulating anything:")
    for mode in s.cfg["design"]["control_modes"]:
        for cname in ("nominal", "encoder_offset_only", "payload_only"):
            errs = [np.linalg.norm(s.static_prediction(t, s.condition(cname), mode) - t.pos) * 1000 for t in s.targets]
            print(f"  {mode:7s} {cname:20s} " + "  ".join(f"{t.name} {e:8.3f}" for t, e in zip(s.targets, errs)))
    print("\nSetup OK.")


if __name__ == "__main__":
    main()
