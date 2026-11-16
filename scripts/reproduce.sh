#!/usr/bin/env bash
# Everything, end to end: fetch the model, install, test, run, analyse, plot, report.
# About 8 minutes on a laptop, almost all of it the two simulation runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== 1/7 fetch the pinned ADAM-U model =="
bash scripts/setup_env.sh

echo "== 2/7 python environment =="
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "== 3/7 tests =="
# The variable stops pytest from loading plugins of a sourced ROS 2 install, which can fail to import.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q

echo "== 4/7 model and target check =="
python3 scripts/check_setup.py
python3 scripts/check_settling.py

echo "== 5/7 experiments =="
python3 src/run_experiment.py
python3 src/run_experiment.py --ablation
python3 scripts/gain_sensitivity.py

echo "== 6/7 analysis and plots =="
python3 src/analysis.py
python3 src/plotting.py

echo "== 7/7 report =="
python3 src/report.py
echo "Done: results/report.md, results/summary.json, results/plots/"
