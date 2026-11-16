#!/usr/bin/env python3
"""
plotting.py -- figures from results/raw_*.csv and results/summary.json.

    python3 src/plotting.py        # writes results/plots/*.png

Colour always means the plant (blue = ideal, orange = perturbed); marker shape
means the metric or the approach.  Every number shown is also in
results/summary.json and results/report.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.analysis import build_summary  # noqa: E402
from src.experiment import REPO_ROOT, load_config  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"nominal": "#2a78d6", "perturbed": "#eb6834"}
PLANT = {"nominal": "ideal plant", "perturbed": "perturbed plant"}
MODE = {"pd": "PD only (vendor example gains)", "pd_gff": "PD + gravity feed-forward"}
MODE_SHORT = {"pd": "PD only", "pd_gff": "PD + gravity FF"}
JOINTS = ["waist", "shoulderPitch", "shoulderRoll", "shoulderYaw", "elbow", "wristYaw", "wristPitch", "wristRoll"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2.0,
})


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_accuracy_repeatability(summary, out_dir):
    groups = summary["main"]["groups"]
    floor = 1e-3
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    labels = []
    for row, g in enumerate(groups):
        y = len(groups) - 1 - row
        c = COLOR[g["condition"]]
        labels.append(f"{MODE_SHORT[g['control_mode']]}\n{PLANT[g['condition']]}")
        ax.plot([g["AP_min_mm"], g["AP_max_mm"]], [y + 0.14, y + 0.14], color=c, lw=2, solid_capstyle="round")
        ax.plot(g["AP_mean_mm"], y + 0.14, "o", ms=9, color=c, mec=SURFACE, mew=1.5)
        ax.annotate(f"AP {_fmt(g['AP_mean_mm'])} mm", (g["AP_max_mm"], y + 0.14), xytext=(9, 0),
                    textcoords="offset points", va="center", color=INK)
        if g["RP_mean_mm"] >= floor:
            ax.plot([max(g["RP_min_mm"], floor), g["RP_max_mm"]], [y - 0.14, y - 0.14], color=c, lw=2,
                    solid_capstyle="round")
            ax.plot(g["RP_mean_mm"], y - 0.14, "D", ms=8, color=c, mec=SURFACE, mew=1.5)
            ax.annotate(f"RP {_fmt(g['RP_mean_mm'])} mm", (g["RP_max_mm"], y - 0.14), xytext=(9, 0),
                        textcoords="offset points", va="center", color=INK)
        else:
            ax.plot(floor, y - 0.14, "D", ms=8, color=c, mec=SURFACE, mew=1.5)
            ax.annotate(f"RP below {floor:g} mm (repeats are identical), drawn at the floor", (floor, y - 0.14),
                        xytext=(9, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.6, 3000)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels[::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("millimetres (log scale); line = range over the 10 target x approach cells")
    ax.set_title("Position accuracy (AP) and repeatability (RP) by control mode and plant")
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=INK2, label="AP  position accuracy"),
               Line2D([], [], marker="D", ls="", ms=7, color=INK2, label="RP  position repeatability"),
               Line2D([], [], marker="s", ls="", ms=8, color=COLOR["nominal"], label="ideal plant"),
               Line2D([], [], marker="s", ls="", ms=8, color=COLOR["perturbed"], label="perturbed plant")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.3)
    return _save(fig, out_dir, "fig1_accuracy_vs_repeatability.png")


def fig_tcp_side_view(df, out_dir):
    modes = list(df["control_mode"].unique())
    fig, axes = plt.subplots(1, len(modes), figsize=(3.5 * len(modes), 5.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    targets = df.drop_duplicates("target")
    # Targets that coincide in this view (same X and Z, different Y) share one label.
    merged = targets.groupby([targets["tgt_x"].round(6), targets["tgt_z"].round(6)])["target"].agg(", ".join)
    for ax, mode in zip(axes, modes):
        for cond, g in df[df["control_mode"] == mode].groupby("condition"):
            ax.plot(g["ach_x"] * 1000, g["ach_z"] * 1000, "o", ms=6, color=COLOR[cond], mec=SURFACE, mew=0.8,
                    label=f"reached, {PLANT[cond]}")
        ax.plot(targets["tgt_x"] * 1000, targets["tgt_z"] * 1000, "x", ms=9, mew=2, color=INK, label="commanded")
        for (x, z), name in merged.items():
            ax.annotate(name, (x * 1000, z * 1000), xytext=(8, 5), textcoords="offset points", color=INK2)
        ax.set_title(MODE[mode])
        ax.set_xlabel("X forward (mm)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Z up (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Commanded and reached TCP positions, side view ({len(df)} trials)", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_commanded_vs_reached.png")


def fig_approach_clusters(df, summary, out_dir, mode="pd_gff", cond="perturbed"):
    g = df[(df["control_mode"] == mode) & (df["condition"] == cond)]
    if g.empty:
        return None
    vap = {t["target"]: t["vAP_mm"] for t in summary["main"]["targets"]
           if t["control_mode"] == mode and t["condition"] == cond}
    names = sorted(g["target"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(1.9 * len(names), 4.4), sharex=True, sharey=True)
    marker = {"plus": "o", "minus": "^"}
    for ax, name in zip(np.atleast_1d(axes), names):
        gt = g[g["target"] == name]
        for approach, ga in gt.groupby("approach"):
            ax.plot(ga["err_y_mm"], ga["err_z_mm"], marker[approach], ms=7, ls="", color=COLOR[cond],
                    mec=SURFACE, mew=0.8, label=f"reached, approach '{approach}'")
        ax.plot(0, 0, "x", ms=9, mew=2, color=INK, label="commanded position")
        ax.set_title(f"{name}\nvAP {vap[name]:.1f} mm", fontweight="normal")
        ax.set_xlabel("error Y (mm)")
        ax.set_aspect("equal", adjustable="box")
    np.atleast_1d(axes)[0].set_ylabel("error Z (mm)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(f"Position error of every trial, by approach direction ({MODE_SHORT[mode]}, {PLANT[cond]})",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_approach_clusters.png")


def fig_error_by_joint(summary, out_dir, kp):
    groups = [g for g in summary["main"]["groups"] if g["AP_mean_mm"] > 1.0]
    if not groups:
        return None
    labels = {"waist": f"waist (3 joints), kp {kp['waistYaw']:.0f}-{kp['waistRoll']:.0f}"}
    for j in JOINTS[1:]:
        labels[j] = f"{j}, kp {kp[j + '_Right']:.0f}"
    xmax = max(max(g["contrib_mean_mm"].values()) for g in groups) * 1.22
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 1.4, 3.4), sharey=True)
    y = np.arange(len(JOINTS))[::-1]
    for ax, g in zip(np.atleast_1d(axes), groups):
        vals = [g["contrib_mean_mm"][j] for j in JOINTS]
        ax.barh(y, vals, height=0.55, color=COLOR[g["condition"]])
        for yi, v in zip(y, vals):
            ax.annotate(_fmt(v), (v, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
        ax.set_xlim(0, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[j] for j in JOINTS])
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{MODE_SHORT[g['control_mode']]}, {PLANT[g['condition']]}", fontweight="normal")
        ax.set_xlabel("mean TCP displacement (mm)")
    fig.suptitle("Which joint the position error comes from (joint error x Jacobian, mean over trials)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_error_by_joint.png")


def fig_error_sources(summary, out_dir):
    if "ablation" not in summary:
        return None
    groups = sorted(summary["ablation"]["groups"], key=lambda g: g["AP_mean_mm"])
    floor = 1e-4
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    c = COLOR["perturbed"]
    for y, g in enumerate(groups):
        ax.plot(max(g["AP_mean_mm"], floor), y, "o", ms=9, color=c, mec=SURFACE, mew=1.5)
        ax.plot(max(g["RP_mean_mm"], floor), y, "D", ms=8, color=INK2, mec=SURFACE, mew=1.5)
        ax.plot(max(g["vAP_mean_mm"], floor), y, "^", ms=9, color=INK, mec=SURFACE, mew=1.5)
        ax.annotate(f"AP {_fmt(g['AP_mean_mm'])} mm", (max(g["AP_mean_mm"], g["vAP_mean_mm"]), y),
                    xytext=(10, 0), textcoords="offset points", va="center", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(floor * 0.6, 1500)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g["condition"].replace("_only", "").replace("_", " ") + " only" for g in groups])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"millimetres (log scale), mean over targets; values below {floor:g} mm are drawn at {floor:g}")
    mode = groups[0]["control_mode"]
    ax.set_title(f"One error source at a time ({MODE_SHORT[mode]})")
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, label="AP  accuracy"),
               Line2D([], [], marker="D", ls="", ms=7, color=INK2, label="RP  repeatability"),
               Line2D([], [], marker="^", ls="", ms=8, color=INK, label="vAP  difference between approaches")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, handletextpad=0.3)
    return _save(fig, out_dir, "fig5_error_sources.png")


def fig_settling(results_dir, out_dir, band_mm, dwell_s):
    path = os.path.join(results_dir, "traces_trials.npz")
    if not os.path.isfile(path):
        return None
    traces = np.load(path)
    modes = [m for m in MODE if any(k.startswith(m + "|") for k in traces.files)]
    fig, axes = plt.subplots(1, len(modes), figsize=(4.4 * len(modes), 3.4), sharey=True)
    for ax, mode in zip(np.atleast_1d(axes), modes):
        for cond in ("nominal", "perturbed"):
            key = f"{mode}|{cond}|plus"
            if key not in traces.files:
                continue
            t, dev = traces[key]
            ax.plot(t[:-1], np.maximum(dev[:-1], 1e-5), color=COLOR[cond], label=PLANT[cond])
        ax.axhline(band_mm, color=INK2, lw=1)
        ax.annotate(f"{band_mm} mm band", (dwell_s, band_mm), xytext=(-2, 4), textcoords="offset points",
                    ha="right", color=INK2)
        ax.set_yscale("log")
        ax.set_xlim(0, dwell_s)
        ax.set_title(MODE[mode], fontweight="normal")
        ax.set_xlabel("time after the commanded motion ends (s)")
    np.atleast_1d(axes)[0].set_ylabel("distance from the measured position (mm)")
    np.atleast_1d(axes)[0].legend(loc="lower left")
    fig.suptitle("Settling during the dwell (target P1, approach 'plus', first repetition)", x=0.02, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig6_settling.png")


def _group_rows(df):
    """(label, colour, sub-frame) for each control mode x plant, top row first."""
    rows = []
    for (mode, cond), g in df.groupby(["control_mode", "condition"], sort=False):
        rows.append((f"{MODE_SHORT[mode]}\n{PLANT[cond]}", COLOR[cond], g))
    return rows


def _strip(ax, rows, column, scale=1.0, fmt=_fmt, unit=""):
    """One row of dots per group (every trial), with the median and the 95th percentile marked."""
    rng = np.random.default_rng(0)                      # fixed jitter, so the figure is reproducible
    n = len(rows)
    for i, (label, colour, g) in enumerate(rows):
        y = n - 1 - i
        v = g[column].to_numpy() * scale
        ax.plot(v, y + rng.uniform(-0.16, 0.16, v.size), "o", ms=4.5, color=colour, mec=SURFACE, mew=0.5,
                alpha=0.75)
        med, p95 = np.median(v), np.percentile(v, 95)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=INK, lw=2, solid_capstyle="round")
        ax.plot(p95, y + 0.36, "v", ms=7, color=INK, mec=SURFACE, mew=0.8)
        ax.annotate(f"median {fmt(med)}{unit}, p95 {fmt(p95)}{unit}", (1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=8.5,
                    annotation_clip=False)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.grid(axis="y", visible=False)


STRIP_LEGEND = [Line2D([], [], marker="o", ls="", ms=5, color=INK2, label="one trial"),
                Line2D([], [], marker="|", ls="", ms=12, mew=2, color=INK, label="median"),
                Line2D([], [], marker="v", ls="", ms=7, color=INK, label="95th percentile"),
                Line2D([], [], marker="s", ls="", ms=8, color=COLOR["nominal"], label="ideal plant"),
                Line2D([], [], marker="s", ls="", ms=8, color=COLOR["perturbed"], label="perturbed plant")]


def fig_error_distributions(df, out_dir):
    rows = _group_rows(df)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4))
    _strip(axes[0], rows, "pos_err_mm", unit=" mm")
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-3, 1e3)
    axes[0].set_xlabel("per-trial position error (mm, log scale)")
    axes[0].set_title(f"Position error of every trial ({len(df)} trials)")
    _strip(axes[1], rows, "ori_err_deg", unit=" deg")
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-3, 1e2)
    axes[1].set_xlabel("per-trial orientation error (degrees, log scale)")
    axes[1].set_title("Orientation error of every trial")
    fig.legend(handles=STRIP_LEGEND, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.02, 0.8, 1))
    return _save(fig, out_dir, "fig8_error_distributions.png")


def fig_operational(df, out_dir, dwell_s):
    rows = _group_rows(df)
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4))
    _strip(axes[0], rows, "stab_time_s", fmt=lambda v: f"{v:.2f}", unit=" s")
    axes[0].axvline(dwell_s, color=INK2, lw=1)
    axes[0].annotate("measurement instant", (dwell_s, 3.42), xytext=(-4, 0), textcoords="offset points", ha="right",
                     va="center", color=INK2, fontsize=8.5)
    axes[0].set_xlim(0, dwell_s * 1.02)
    axes[0].set_xlabel("stabilisation time after the commanded motion ends (s)")
    axes[0].set_title("Stabilisation time of every trial (time until the TCP stays within the band)")
    _strip(axes[1], rows, "peak_torque_frac", scale=100.0, fmt=lambda v: f"{v:.0f}", unit=" %")
    axes[1].axvline(100, color=INK2, lw=1)
    axes[1].annotate("actuator limit", (100, 3.42), xytext=(-4, 0), textcoords="offset points", ha="right",
                     va="center", color=INK2, fontsize=8.5)
    axes[1].set_xlim(0, 102)
    axes[1].set_xlabel("highest right-arm joint torque used in the trial (% of the actuator limit)")
    axes[1].set_title("Peak joint torque of every trial")
    fig.legend(handles=STRIP_LEGEND, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.02, 0.8, 1))
    return _save(fig, out_dir, "fig9_operational.png")


def fig_repeatability_spread(summary, out_dir):
    """Scatter of the repeats inside each cell: SD along x, y, z and the 95 % radius.
    Only plants with random input are drawn; on the ideal plant every value is zero."""
    cells = [c for c in summary["main"]["cells"] if c["RP_mm"] > 1e-9]
    if not cells:
        return None
    modes = list(dict.fromkeys(c["control_mode"] for c in cells))
    cats = ["SD x", "SD y", "SD z", "95 % radius", "RP (ISO)"]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, mode in zip(np.atleast_1d(axes), modes):
        sel = [c for c in cells if c["control_mode"] == mode]
        cond = sel[0]["condition"]
        cols = [[c["std_xyz_mm"][0] for c in sel], [c["std_xyz_mm"][1] for c in sel],
                [c["std_xyz_mm"][2] for c in sel], [c["radius_p95_mm"] for c in sel], [c["RP_mm"] for c in sel]]
        for x, v in enumerate(cols):
            v = np.array(v)
            ax.plot(x + rng.uniform(-0.09, 0.09, v.size), v, "o", ms=6, color=COLOR[cond], mec=SURFACE, mew=0.8)
            ax.plot([x - 0.2, x + 0.2], [v.mean(), v.mean()], color=INK, lw=2, solid_capstyle="round")
            ax.annotate(_fmt(v.mean()), (x + 0.22, v.mean()), xytext=(2, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats)
        ax.set_xlim(-0.5, len(cats) - 0.2)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{MODE_SHORT[mode]}, {PLANT[cond]}", fontweight="normal")
    np.atleast_1d(axes)[0].set_ylabel("millimetres")
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=COLOR["perturbed"],
                      label="one target x approach cell (10 repeats)"),
               Line2D([], [], marker="_", ls="", ms=14, mew=2, color=INK, label="mean over the 10 cells")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Repeatability: scatter of the repeats within a cell (ideal plant: all zero, not drawn)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, "fig10_repeatability_spread.png")


def main():
    results_dir = os.path.join(REPO_ROOT, "results")
    out_dir = os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config()
    summary = build_summary(results_dir)          # always from the raw files, never a stale summary.json
    df = pd.read_csv(os.path.join(results_dir, "raw_trials.csv"))
    kp = {j: v[0] for j, v in cfg["controller"]["gains"].items()}
    paths = [
        fig_accuracy_repeatability(summary, out_dir),
        fig_tcp_side_view(df, out_dir),
        fig_approach_clusters(df, summary, out_dir),
        fig_error_by_joint(summary, out_dir, kp),
        fig_error_sources(summary, out_dir),
        fig_settling(results_dir, out_dir, cfg["protocol"]["stabilisation_band_mm"], cfg["protocol"]["dwell_s"]),
        fig_error_distributions(df, out_dir),
        fig_operational(df, out_dir, cfg["protocol"]["dwell_s"]),
        fig_repeatability_spread(summary, out_dir),
    ]
    for p in paths:
        if p:
            print("wrote", os.path.relpath(p, REPO_ROOT))
    return [p for p in paths if p]


if __name__ == "__main__":
    main()
