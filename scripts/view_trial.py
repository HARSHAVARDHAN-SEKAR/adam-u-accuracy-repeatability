#!/usr/bin/env python3
"""
view_trial.py -- watch one trial in the MuJoCo viewer (needs a display).

    python3 scripts/view_trial.py                                   # P1, pd_gff, perturbed, approach plus
    python3 scripts/view_trial.py --mode pd --condition nominal     # see the arm sag without feed-forward
    python3 scripts/view_trial.py --target P3 --approach minus --slow 2
    python3 scripts/view_trial.py --transparent                     # see the target through the hand
    python3 scripts/view_trial.py --overlay                         # live error drawn in the window
    python3 scripts/view_trial.py --headless                        # no window, just print the result

What is drawn:
  blue sphere    the commanded position (the target)
  red dot        the measured point on the palm (the TCP site)
  yellow line    from the palm to the target; its length is the position error

When the hand reaches the target the blue sphere sits inside the palm and the
line shrinks to nothing. Use --transparent to keep the sphere visible then.

The trial is the same code path as the experiment, so the numbers printed at the
end match results/raw_trials.csv.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.experiment import APPROACHES, Setup  # noqa: E402
from src.kinematics import orientation_angle  # noqa: E402
from src.trial import run_trial  # noqa: E402

TARGET_RGBA = np.array([0.15, 0.45, 1.0, 0.85], dtype=np.float32)
LINE_RGBA = np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32)


def draw_markers(scn, target_pos, tcp_pos):
    """Target sphere plus a line from the palm to the target, in the viewer's user scene."""
    scn.ngeom = 0
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]),
                        target_pos, np.eye(3).flatten(), TARGET_RGBA)
    mujoco.mjv_initGeom(scn.geoms[1], mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), LINE_RGBA)
    if np.linalg.norm(tcp_pos - target_pos) > 1e-4:
        mujoco.mjv_connector(scn.geoms[1], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, tcp_pos, target_pos)
        scn.ngeom = 2
    else:
        scn.ngeom = 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--mode", default="pd_gff", choices=["pd", "pd_gff"])
    ap.add_argument("--condition", default="perturbed")
    ap.add_argument("--target", default="P1")
    ap.add_argument("--approach", default="plus", choices=list(APPROACHES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slow", type=float, default=1.0, help="slow-motion factor")
    ap.add_argument("--transparent", action="store_true", help="make the robot see-through")
    ap.add_argument("--show-ui", action="store_true", help="show MuJoCo's side panels")
    ap.add_argument("--overlay", action="store_true",
                    help="also draw the live error inside the window (needs a MuJoCo version with "
                         "viewer.set_texts; without this flag the error is printed in the terminal)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    s = Setup(model_path=args.model_path)
    target = next(t for t in s.targets if t.name == args.target)
    cond = s.condition(args.condition)
    plant, data = s.plant_for(cond)
    rng = np.random.default_rng(args.seed) if cond.is_stochastic else None
    tcp = plant.tcp_site_id

    def run(on_step=None):
        return run_trial(plant, data, s.controller(args.mode), cond, s.start_config(target, args.approach),
                         target.q_cmd, s.cfg["protocol"], rng, sim_cfg=s.cfg["simulation"], on_step=on_step)

    if args.headless:
        res = run()
    else:
        import mujoco.viewer
        dt = plant.model.opt.timestep
        with mujoco.viewer.launch_passive(plant.model, data, show_left_ui=args.show_ui,
                                          show_right_ui=args.show_ui) as viewer:
            has_text = args.overlay and hasattr(viewer, "set_texts")
            state = {"n": 0, "t0": time.perf_counter(), "last_print": -1.0, "framed": 0}
            title = f"{args.mode} / {args.condition} / {args.target} / approach {args.approach}"

            def frame_camera():
                # Frame the arm and the target instead of the whole robot on its column.
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = target.pos + np.array([-0.05, 0.0, -0.05])
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, 140.0, -10.0
                if args.transparent:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1

            def refresh(phase):
                err = float(np.linalg.norm(data.site_xpos[tcp] - target.pos) * 1000.0)
                with viewer.lock():
                    # The viewer resets its camera while it starts up, which can undo a
                    # setting made right after launch, so re-apply it for the first refreshes.
                    # After that the camera is yours: drag to orbit, scroll to zoom.
                    if state["framed"] < 30:
                        frame_camera()
                        state["framed"] += 1
                    draw_markers(viewer.user_scn, target.pos, data.site_xpos[tcp].copy())
                # set_texts must be called outside viewer.lock(): on MuJoCo 3.17 it takes the
                # same lock itself, and calling it while holding the lock freezes the window.
                state["text_tick"] = state.get("text_tick", 0) + 1
                if has_text and state["text_tick"] % 20 == 1:
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"{title}\n{phase}", f"position error {err:9.3f} mm"))
                viewer.sync()
                return err

            def on_step():
                state["n"] += 1
                if state["n"] % 10 == 0:                       # 100 Hz refresh
                    t_sim = state["n"] * dt
                    err = refresh(f"t = {t_sim:5.2f} s")
                    if not has_text and t_sim - state["last_print"] >= 0.5:
                        print(f"  t = {t_sim:5.2f} s   error {err:9.3f} mm", flush=True)
                        state["last_print"] = t_sim
                    wait = state["t0"] + state["n"] * dt * args.slow - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                if not viewer.is_running():
                    raise KeyboardInterrupt

            try:
                res = run(on_step)
            except KeyboardInterrupt:
                print("viewer closed before the trial finished")
                return
            print("trial finished; close the viewer window to exit")
            while viewer.is_running():
                refresh("finished - close the window to exit")
                time.sleep(0.05)

    err = (res.tcp_pos - target.pos) * 1000.0
    print(f"{args.mode} / {args.condition} / {args.target} / approach {args.approach}")
    print(f"  position error {np.linalg.norm(err):.3f} mm  (x {err[0]:+.3f}, y {err[1]:+.3f}, z {err[2]:+.3f})")
    print(f"  orientation error {np.degrees(orientation_angle(target.quat, res.tcp_quat)):.3f} deg")
    print(f"  stabilisation time {res.stabilisation_time_s:.2f} s, peak torque {100 * res.peak_torque_fraction:.0f} % of limit")


if __name__ == "__main__":
    main()
