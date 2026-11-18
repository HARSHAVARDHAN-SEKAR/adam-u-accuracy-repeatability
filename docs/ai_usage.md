# AI usage

The brief allows AI tools and asks for a note on how they were used.

**Proposal (steps 1 to 4).** ChatGPT helped structure the first draft.

**Code (step 5).** The simulation code, tests and first documentation were
written with Claude (Anthropic) in an interactive session.

**What I did myself, on my own machine (Ubuntu 22.04, MuJoCo 3.17.0):**

- Ran the full pipeline with `bash scripts/reproduce.sh`: 30 tests passed and
  the 400 + 250 trials reproduced the reference numbers to three decimals.
- Checked the elbow-sag hand calculation (3.558 N·m / 9 N·m/rad ≈ 22.65°)
  against `scripts/check_setup.py` and the measured PD error.
- Watched and recorded trials in the viewer in both control modes.
- Found problems on my setup that were then fixed: the target was not visible
  in the viewer, the viewer froze on MuJoCo 3.17.0, and pytest crashed on the
  ROS 2 plugins (now `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`).
- Revised the proposal and slides so they describe the experiment that was
  actually run.

**Not verified.** Nothing was run on hardware. The perturbation magnitudes are
assumed values.
