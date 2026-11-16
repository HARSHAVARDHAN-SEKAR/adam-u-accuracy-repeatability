#!/usr/bin/env python3
"""
gain_sensitivity.py -- how much of the error is the choice of gains?

    python3 scripts/gain_sensitivity.py          # about 2 minutes

The vendor example gives five right-arm joints (shoulder yaw, elbow, three wrist
joints) a stiffness of kp = 9 N*m/rad, twenty times softer than the shoulder
pitch and roll joints.  This script repeats the experiment with that stiffness
raised step by step and everything else unchanged.  It is a what-if measurement,
not a tuning proposal: no claim is made that these gains are safe on hardware.

kd is scaled with sqrt(kp) so the damping ratio of each joint stays the same.
Each cell is one trial (5 targets x 2 approaches); the ideal plant is
deterministic, and on the perturbed plant the scatter between repeats is about
0.02 mm, far below the differences shown here.

Writes results/gain_sensitivity.csv and results/plots/fig7_gain_sensitivity.png.
"""
import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from src.experiment import APPROACHES, REPO_ROOT, Setup, load_config, trial_seed  # noqa: E402
from src.trial import run_trial  # noqa: E402

SOFT_JOINTS = ["shoulderYaw_Right", "elbow_Right", "wristYaw_Right", "wristPitch_Right", "wristRoll_Right"]
KP_VALUES = [9.0, 18.0, 45.0, 90.0, 180.0]
CASES = [("pd", "nominal"), ("pd", "perturbed"), ("pd_gff", "nominal"), ("pd_gff", "perturbed")]


def setup_with_kp(base_cfg: dict, kp: float, model_path=None) -> Setup:
    cfg = copy.deepcopy(base_cfg)
    for joint in SOFT_JOINTS:
        kp0, kd0 = base_cfg["controller"]["gains"][joint]
        cfg["controller"]["gains"][joint] = [kp, kd0 * float(np.sqrt(kp / kp0))]
    return Setup(cfg, model_path)


def measure(base_cfg: dict, model_path=None, kp_values=KP_VALUES):
    rows = []
    for kp in kp_values:
        s = setup_with_kp(base_cfg, kp, model_path)
        for mode, cname in CASES:
            cond = s.condition(cname)
            plant, data = s.plant_for(cond)
            errs, stab, peak, static = [], [], [], []
            for ti, target in enumerate(s.targets):
                for ai, approach in enumerate(APPROACHES):
                    rng = (np.random.default_rng(trial_seed(base_cfg["design"]["seed"], 7, ti, ai))
                           if cond.is_stochastic else None)
                    res = run_trial(plant, data, s.controller(mode), cond, s.start_config(target, approach),
                                    target.q_cmd, s.cfg["protocol"], rng, sim_cfg=s.cfg["simulation"])
                    errs.append(np.linalg.norm(res.tcp_pos - target.pos) * 1000.0)
                    stab.append(res.stabilisation_time_s)
                    peak.append(res.peak_torque_fraction)
                static.append(np.linalg.norm(s.static_prediction(target, cond, mode) - target.pos) * 1000.0)
            rows.append({"soft_joint_kp": kp, "control_mode": mode, "condition": cname,
                         "n_trials": len(errs), "mean_error_mm": float(np.mean(errs)),
                         "max_error_mm": float(np.max(errs)),
                         "static_prediction_mm": float(np.mean(static)),
                         "stab_time_max_s": float(np.max(stab)), "peak_torque_frac_max": float(np.max(peak))})
            r = rows[-1]
            print(f"kp {kp:6.1f}  {mode:7s} {cname:10s} mean error {r['mean_error_mm']:9.3f} mm   "
                  f"static prediction {r['static_prediction_mm']:9.3f} mm   stab max {r['stab_time_max_s']:.2f} s",
                  flush=True)
    return rows


def plot(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import COLOR, INK, MODE_SHORT, PLANT, SURFACE, _fmt  # also applies the plot style

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    marker = {"pd": "o", "pd_gff": "D"}
    for mode, cname in CASES:
        sel = [r for r in rows if r["control_mode"] == mode and r["condition"] == cname]
        x = [r["soft_joint_kp"] for r in sel]
        y = [r["mean_error_mm"] for r in sel]
        ax.plot(x, y, marker=marker[mode], ms=8, color=COLOR[cname], mec=SURFACE, mew=1.5,
                ls="-" if mode == "pd" else (0, (5, 2)))
        # The two "PD only" curves end close together, so their labels are nudged apart.
        dy = {("pd", "perturbed"): 8, ("pd", "nominal"): -8}.get((mode, cname), 0)
        ax.annotate(f"{MODE_SHORT[mode]}, {PLANT[cname]}:  {_fmt(y[0])} to {_fmt(y[-1])} mm", (x[-1], y[-1]),
                    xytext=(10, dy), textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(KP_VALUES)
    ax.set_xticklabels([f"{k:g}" for k in KP_VALUES])
    ax.minorticks_off()
    ax.set_xlim(7.5, 2600)
    ax.spines["bottom"].set_bounds(KP_VALUES[0], KP_VALUES[-1])
    ax.set_xlabel("stiffness kp of shoulder yaw, elbow and wrist joints (N*m/rad); vendor example = 9",
                  loc="left")
    ax.set_ylabel("mean position error (mm)")
    ax.set_title("Position error against the stiffness of the five soft arm joints")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from the existing CSV")
    args = ap.parse_args()
    out = os.path.join(REPO_ROOT, "results", "gain_sensitivity.csv")
    if args.plot_only:
        with open(out) as f:
            rows = [{k: (v if k in ("control_mode", "condition") else float(v)) for k, v in r.items()}
                    for r in csv.DictReader(f)]
    else:
        rows = measure(load_config(), args.model_path)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", os.path.relpath(out, REPO_ROOT))
    fig = os.path.join(REPO_ROOT, "results", "plots", "fig7_gain_sensitivity.png")
    os.makedirs(os.path.dirname(fig), exist_ok=True)
    plot(rows, fig)
    print("wrote", os.path.relpath(fig, REPO_ROOT))


if __name__ == "__main__":
    main()
