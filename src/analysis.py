#!/usr/bin/env python3
"""
analysis.py -- turn the raw trial tables into results/summary.json.

    python3 src/analysis.py

Three levels:
  cells   one per (control mode, condition, target, approach): AP, RP over the repetitions
  targets one per (control mode, condition, target): vAP between the two approaches
  groups  one per (control mode, condition): the cells and trials pooled
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import metrics  # noqa: E402
from src.experiment import REPO_ROOT, load_config  # noqa: E402

XYZ = ["ach_x", "ach_y", "ach_z"]
TGT = ["tgt_x", "tgt_y", "tgt_z"]


def summarise(df: pd.DataFrame, pos_tol_mm: float, ori_tol_deg: float, dwell_s: float | None = None) -> dict:
    cells, targets, groups = [], [], []
    for (mode, cond), g in df.groupby(["control_mode", "condition"], sort=False):
        cell_rows, target_rows = [], []
        for target, gt in g.groupby("target"):
            tgt = gt[TGT].iloc[0].to_numpy()
            bary = []
            for approach, gc in gt.groupby("approach"):
                pts = gc[XYZ].to_numpy()
                bary.append(metrics.barycentre(pts))
                l = metrics.radii(pts) * 1000.0
                err = metrics.per_trial_error(pts, tgt) * 1000.0
                cell_rows.append({
                    "control_mode": mode, "condition": cond, "target": target, "approach": approach,
                    "n": int(len(gc)),
                    "AP_mm": metrics.position_accuracy(pts, tgt) * 1000.0,
                    "AP_vector_mm": ((metrics.barycentre(pts) - tgt) * 1000.0).tolist(),
                    "RP_mm": metrics.position_repeatability(pts) * 1000.0,
                    "mean_error_mm": metrics.mean_error(err),
                    "error_spread_mm": metrics.error_spread(err),
                    "rmse_mm": metrics.rmse(err),
                    "error_max_mm": float(err.max()),
                    "radius_mean_mm": float(l.mean()), "radius_max_mm": float(l.max()),
                    "radius_p95_mm": metrics.radius_percentile(pts, 95.0) * 1000.0,
                    "std_xyz_mm": (metrics.axis_std(pts) * 1000.0).tolist(),
                    "ori_err_mean_deg": float(gc["ori_err_deg"].mean()),
                    "ori_err_std_deg": float(gc["ori_err_deg"].std(ddof=1)),
                    "stab_time_mean_s": float(gc["stab_time_s"].mean()),
                })
            target_rows.append({"control_mode": mode, "condition": cond, "target": target,
                                "vAP_mm": metrics.multidirectional_variation(np.array(bary)) * 1000.0})
        cells += cell_rows
        targets += target_rows
        ap = np.array([c["AP_mm"] for c in cell_rows])
        rp = np.array([c["RP_mm"] for c in cell_rows])
        vap = np.array([t["vAP_mm"] for t in target_rows])
        contrib_cols = [c for c in g.columns if c.startswith("contrib_mm_")]
        pct = metrics.percentiles(g["pos_err_mm"])
        std_xyz = np.array([c["std_xyz_mm"] for c in cell_rows])
        # A trial counts as not settled when the TCP was still outside the stabilisation
        # band at the end of the dwell. There is no other way to fail: the timing is fixed.
        not_settled = int((g["stab_time_s"] >= dwell_s - 0.011).sum()) if dwell_s else 0
        groups.append({
            "pos_err_p50_mm": pct["p50"], "pos_err_p90_mm": pct["p90"], "pos_err_p95_mm": pct["p95"],
            "std_xyz_mean_mm": std_xyz.mean(axis=0).tolist(), "std_xyz_max_mm": std_xyz.max(axis=0).tolist(),
            "radius_p95_mean_mm": float(np.mean([c["radius_p95_mm"] for c in cell_rows])),
            "radius_p95_max_mm": float(np.max([c["radius_p95_mm"] for c in cell_rows])),
            "ori_err_p95_deg": float(np.percentile(g["ori_err_deg"], 95)),
            "ori_err_spread_mean_deg": float(np.mean([c["ori_err_std_deg"] for c in cell_rows])),
            "ori_err_spread_max_deg": float(np.max([c["ori_err_std_deg"] for c in cell_rows])),
            "stab_time_p95_s": float(np.percentile(g["stab_time_s"], 95)),
            "peak_torque_frac_mean": float(g["peak_torque_frac"].mean()),
            "n_not_settled": not_settled,
            "control_mode": mode, "condition": cond, "n_trials": int(len(g)),
            "AP_mean_mm": float(ap.mean()), "AP_min_mm": float(ap.min()), "AP_max_mm": float(ap.max()),
            "RP_mean_mm": float(rp.mean()), "RP_min_mm": float(rp.min()), "RP_max_mm": float(rp.max()),
            "vAP_mean_mm": float(vap.mean()), "vAP_max_mm": float(vap.max()),
            "mean_error_mm": metrics.mean_error(g["pos_err_mm"]),
            "error_spread_mm": metrics.error_spread(g["pos_err_mm"]),
            "pos_err_rmse_mm": metrics.rmse(g["pos_err_mm"]), "pos_err_max_mm": float(g["pos_err_mm"].max()),
            "ori_err_mean_deg": float(g["ori_err_deg"].mean()), "ori_err_max_deg": float(g["ori_err_deg"].max()),
            "success_rate": metrics.success_rate(g["pos_err_mm"], g["ori_err_deg"], pos_tol_mm, ori_tol_deg),
            "stab_time_mean_s": float(g["stab_time_s"].mean()), "stab_time_max_s": float(g["stab_time_s"].max()),
            "peak_torque_frac_max": float(g["peak_torque_frac"].max()),
            "n_saturated": int(g["saturated"].sum()),
            "static_pred_err_mean_mm": float(g["static_pred_err_mm"].mean()),
            "resid_after_static_mean_mm": float(g["resid_after_static_mm"].mean()),
            "resid_after_static_max_mm": float(g["resid_after_static_mm"].max()),
            "linearisation_gap_max_mm": float((g["linear_sum_mm"] - g["pos_err_mm"]).abs().max()),
            "contrib_mean_mm": {c.replace("contrib_mm_", ""): float(g[c].mean()) for c in contrib_cols},
        })
    return {"groups": groups, "targets": targets, "cells": cells}


def build_summary(results_dir: str | None = None) -> dict:
    results_dir = results_dir or os.path.join(REPO_ROOT, "results")
    cfg = load_config()
    tol = cfg["tolerance"]
    dwell_s = float(cfg["protocol"]["dwell_s"])
    out = {"tolerance": tol}
    for key, stem in (("main", "trials"), ("ablation", "ablation")):
        path = os.path.join(results_dir, f"raw_{stem}.csv")
        if os.path.isfile(path):
            out[key] = summarise(pd.read_csv(path), tol["position_mm"], tol["orientation_deg"], dwell_s)
            info = os.path.join(results_dir, f"run_info_{stem}.json")
            if os.path.isfile(info):
                with open(info) as f:
                    out[key]["run_info"] = json.load(f)
    if "main" not in out:
        raise FileNotFoundError(f"no raw_trials.csv in {results_dir}; run src/run_experiment.py first")
    return out


def print_groups(title: str, groups: list) -> None:
    print(f"\n{title}")
    print("  ISO-style (where the cluster sits, how tight it is)        "
          "| classic (distribution of the error magnitude)")
    print(f"{'mode':7s} {'condition':20s} {'n':>4s} {'AP mean':>9s} {'RP mean':>9s} {'vAP max':>9s} "
          f"| {'mean err':>9s} {'spread':>9s} {'RMSE':>9s} {'worst':>9s} {'success':>8s}")
    for g in groups:
        print(f"{g['control_mode']:7s} {g['condition']:20s} {g['n_trials']:4d} {g['AP_mean_mm']:9.3f} "
              f"{g['RP_mean_mm']:9.4f} {g['vAP_max_mm']:9.3f} | {g['mean_error_mm']:9.3f} "
              f"{g['error_spread_mm']:9.3f} {g['pos_err_rmse_mm']:9.3f} {g['pos_err_max_mm']:9.3f} "
              f"{g['success_rate']:8.2f}")


def main():
    summary = build_summary()
    path = os.path.join(REPO_ROOT, "results", "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print_groups("Main experiment (mm, deg, s)", summary["main"]["groups"])
    if "ablation" in summary:
        print_groups("One error source at a time (mm, deg, s)", summary["ablation"]["groups"])
    print(f"\nwrote {path}")
    return summary


if __name__ == "__main__":
    main()
