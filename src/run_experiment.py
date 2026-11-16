#!/usr/bin/env python3
"""
run_experiment.py -- run the factorial experiment and write the raw results.

    python3 src/run_experiment.py              # main experiment -> results/raw_trials.csv
    python3 src/run_experiment.py --ablation   # one error source at a time -> results/raw_ablation.csv
    python3 src/run_experiment.py --smoke      # 1 repetition, quick end-to-end check (writes to results/smoke/)

Main design (config/experiment.yaml):
    5 targets x 2 approaches x 2 control modes x 2 plant conditions x 10 repetitions = 400 trials.

Each trial resets the simulation, so trials are independent, and the random
numbers of a trial are seeded from what the trial is (mode, condition, target,
approach, repetition), not from when it runs.  The run order is shuffled
anyway (seeded): on hardware it would guard against drift, and here the tests
use it to show that order really has no effect.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.experiment import APPROACHES, REPO_ROOT, Setup, trial_seed  # noqa: E402
from src.kinematics import orientation_angle  # noqa: E402
from src.model_builder import ARM_SLICE, RIGHT_ARM_JOINTS  # noqa: E402
from src.trial import run_trial  # noqa: E402

SHORT = [j.replace("_Right", "") for j in RIGHT_ARM_JOINTS]


def trial_list(setup: Setup, modes, conditions, repetitions):
    trials = []
    for mi, mode in enumerate(modes):
        for cname in conditions:
            ci = sorted(setup.cfg["conditions"]).index(cname)
            for ti, target in enumerate(setup.targets):
                for ai, approach in enumerate(APPROACHES):
                    for rep in range(repetitions):
                        trials.append(dict(mode=mode, condition=cname, target=target, approach=approach,
                                           rep=rep, seed_key=(["pd", "pd_gff"].index(mode), ci, ti, ai, rep)))
    return trials


def execute(setup: Setup, spec: dict, collect_trace: bool = False):
    """Run one trial described by `spec` and return its CSV row (and the dwell trace)."""
    cfg = setup.cfg
    target = spec["target"]
    condition = setup.condition(spec["condition"])
    plant, data = setup.plant_for(condition)
    seq = trial_seed(cfg["design"]["seed"], *spec["seed_key"])
    rng = np.random.default_rng(seq) if condition.is_stochastic else None
    res = run_trial(plant, data, setup.controller(spec["mode"]), condition,
                    setup.start_config(target, spec["approach"]), target.q_cmd, cfg["protocol"], rng,
                    sim_cfg=cfg["simulation"])

    err = (res.tcp_pos - target.pos) * 1000.0
    pred = setup.static_prediction(target, condition, spec["mode"])
    contrib_mm = res.joint_contrib_m * 1000.0
    row = {
        "control_mode": spec["mode"], "condition": spec["condition"], "target": target.name,
        "approach": spec["approach"], "repetition": spec["rep"],
        "seed_entropy": "-".join(str(k) for k in spec["seed_key"]),
        "tgt_x": target.pos[0], "tgt_y": target.pos[1], "tgt_z": target.pos[2],
        "ach_x": res.tcp_pos[0], "ach_y": res.tcp_pos[1], "ach_z": res.tcp_pos[2],
        "ach_qw": res.tcp_quat[0], "ach_qx": res.tcp_quat[1], "ach_qy": res.tcp_quat[2], "ach_qz": res.tcp_quat[3],
        "err_x_mm": err[0], "err_y_mm": err[1], "err_z_mm": err[2],
        "pos_err_mm": float(np.linalg.norm(err)),
        "ori_err_deg": float(np.degrees(orientation_angle(target.quat, res.tcp_quat))),
        "static_pred_err_mm": float(np.linalg.norm(pred - target.pos) * 1000.0),
        "resid_after_static_mm": float(np.linalg.norm(res.tcp_pos - pred) * 1000.0),
        "stab_time_s": res.stabilisation_time_s,
        "peak_torque_frac": res.peak_torque_fraction,
        "saturated": int(res.saturated),
        "ik_residual_um": target.ik_residual_m * 1e6,
    }
    qerr = np.degrees(res.q_true19[ARM_SLICE] - target.q_cmd)
    for name, v in zip(SHORT, qerr):
        row[f"qerr_deg_{name}"] = float(v)
    for name, c in zip(SHORT, contrib_mm[ARM_SLICE]):
        row[f"contrib_mm_{name}"] = float(np.linalg.norm(c))
    row["contrib_mm_waist"] = float(np.linalg.norm(contrib_mm[0:3].sum(axis=0)))
    row["linear_sum_mm"] = float(np.linalg.norm(contrib_mm.sum(axis=0)))
    trace = (res.trace_t, res.trace_dev_mm) if collect_trace else None
    return row, trace


def pnd_commit(model_path: str) -> str:
    try:
        out = subprocess.run(["git", "-C", os.path.dirname(model_path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=None, help="path to pnd_robots/adam_u/scene.xml")
    ap.add_argument("--ablation", action="store_true", help="run one error source at a time")
    ap.add_argument("--smoke", action="store_true", help="1 repetition per cell, output in results/smoke/")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    setup = Setup(model_path=args.model_path)
    design = setup.cfg["design"]
    if args.ablation:
        modes, conditions = [design["ablation_control_mode"]], design["ablation_conditions"]
        reps, stem = design["ablation_repetitions"], "ablation"
    else:
        modes, conditions = design["control_modes"], design["conditions"]
        reps, stem = design["repetitions"], "trials"
    if args.smoke:
        reps = 1
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "smoke" if args.smoke else "")
    os.makedirs(out_dir, exist_ok=True)

    trials = trial_list(setup, modes, conditions, reps)
    order = np.random.default_rng(trial_seed(design["seed"], 999, int(args.ablation))).permutation(len(trials))
    print(f"{len(trials)} trials: modes={modes} conditions={conditions} repetitions={reps}")
    print(f"model: {setup.model_path}\nMuJoCo {mujoco.__version__}")

    rows, traces = [], {}
    t0 = time.time()
    for n, idx in enumerate(order, start=1):
        spec = trials[idx]
        want_trace = spec["rep"] == 0 and spec["target"].name == "P1"
        row, trace = execute(setup, spec, collect_trace=want_trace)
        row = {"run_order": n, **row}
        rows.append(row)
        if trace is not None:
            traces[f"{spec['mode']}|{spec['condition']}|{spec['approach']}"] = trace
        if n % 40 == 0 or n == len(trials):
            print(f"  {n:4d}/{len(trials)}  {time.time() - t0:6.1f} s")

    # Stored in design order, so the file does not depend on the shuffle.
    key = lambda r: (r["control_mode"], r["condition"], r["target"], r["approach"], r["repetition"])  # noqa: E731
    rows.sort(key=key)
    csv_path = os.path.join(out_dir, f"raw_{stem}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(os.path.join(out_dir, f"traces_{stem}.npz"),
                        **{k: np.vstack(v) for k, v in traces.items()})
    with open(os.path.join(out_dir, "targets.json"), "w") as f:
        json.dump([{"name": t.name, "position_m": t.pos.tolist(), "quaternion_wxyz": t.quat.tolist(),
                    "joint_command_rad": t.q_cmd.tolist(), "ik_residual_m": t.ik_residual_m,
                    "ik_residual_rad": t.ik_residual_rad,
                    "joint_limit_margin_deg": float(np.degrees(t.limit_margin_rad))} for t in setup.targets],
                  f, indent=2)
    info = {
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_trials": len(rows), "elapsed_s": round(time.time() - t0, 1),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "pnd_mujoco_commit_found": pnd_commit(setup.model_path),
        "pnd_mujoco_commit_expected": setup.cfg["model"]["pnd_mujoco_commit"],
        "seed": design["seed"], "control_modes": modes, "conditions": conditions, "repetitions": reps,
    }
    with open(os.path.join(out_dir, f"run_info_{stem}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {csv_path}  ({len(rows)} trials, {info['elapsed_s']} s)")
    return rows


if __name__ == "__main__":
    main()
