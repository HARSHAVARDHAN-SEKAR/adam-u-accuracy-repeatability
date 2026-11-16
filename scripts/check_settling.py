#!/usr/bin/env python3
"""
check_settling.py -- is the dwell long enough?

Runs a few trials with a dwell four times longer than the experiment uses and
reports how far the TCP still travels after the instant at which the experiment
would have measured it.

    python3 scripts/check_settling.py [--model-path PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from src.experiment import Setup, load_config  # noqa: E402
from src.trial import run_trial  # noqa: E402

CASES = [("pd_gff", "nominal"), ("pd", "nominal"), ("pd_gff", "payload_only"), ("pd_gff", "friction_only"),
         ("pd_gff", "encoder_noise_only"), ("pd_gff", "perturbed"), ("pd", "perturbed")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    cfg = load_config()
    dwell = cfg["protocol"]["dwell_s"]
    cfg["protocol"]["dwell_s"] = 4 * dwell
    s = Setup(cfg, args.model_path)
    print(f"TCP travel between the measurement instant ({dwell:g} s) and {4 * dwell:g} s of dwell, in mm")
    print(f"{'control':8s} {'plant condition':20s} " + " ".join(f"{t.name + ' ' + a:>10s}" for t in s.targets[:2]
                                                                 for a in ("plus", "minus")))
    worst = 0.0
    for mode, cname in CASES:
        cond = s.condition(cname)
        plant, data = s.plant_for(cond)
        row = []
        for t in s.targets[:2]:
            for approach in ("plus", "minus"):
                rng = np.random.default_rng(11) if cond.is_stochastic else None
                res = run_trial(plant, data, s.controller(mode), cond, s.start_config(t, approach), t.q_cmd,
                                cfg["protocol"], rng, sim_cfg=cfg["simulation"])
                row.append(res.trace_dev_mm[np.searchsorted(res.trace_t, dwell) - 1])
        worst = max(worst, max(row))
        print(f"{mode:8s} {cname:20s} " + " ".join(f"{v:10.4f}" for v in row))
    print(f"\nlargest: {worst:.4f} mm")


if __name__ == "__main__":
    main()
