# Assumptions and design decisions

The brief leaves gaps on purpose. This file lists every assumption made to fill
them, with the reason and, where there is one, the measurement behind it.

## 1. The robot

**Fixed base.** The vendored model connects the robot to the world with a free
joint and lets it stand on the floor through contact. The brief calls ADAM-U a
fixed-base robot, so the free joint is removed and the lifting column becomes
rigid with the world. The column height is the model default and is not varied.

**Tool centre point.** The model has no sites, so it has no tool frame. A site
`tcp_right` is added to `wristRollRight`, the last body of the right arm,
90 mm along the hand axis. That is the palm centre: the thumb base sits at
47 mm and the knuckles at 113 to 128 mm. A wrist-fixed point was chosen over a
fingertip so that the result does not depend on finger posture or on which hand
is mounted.

**No edits to vendored files.** The MJCF is parsed, patched in memory and
compiled from a string. Plain XML editing is used instead of MuJoCo's `MjSpec`
because `MjSpec` changed between 3.2 and 4.x.

**Right arm only.** Seven joints: shoulder pitch, roll, yaw, elbow, wrist yaw,
pitch, roll. Waist, neck and left arm are commanded to zero and held by the
same control law. The fingers are left at their default command.

## 2. How the robot is commanded

**Control law.** The 19 motors are torque actuators with no servo inside the
model. The vendor simulator closes the loop in `pndbotics_sdk_py_bridge.py`:

    ctrl = tau + kp * (q_cmd - q) + kd * (dq_cmd - dq),   with dq_cmd = 0

The benchmark applies that law at every 1 ms simulation step.

**Gains.** Taken unchanged from `example/python/adam_U_open_arm.py`: kp 180 on
shoulder pitch and roll, kp 9 on shoulder yaw, elbow and wrists. These are the
only gains PNDbotics publishes for ADAM-U. They may be demonstration values.
All results are conditional on them, and errors scale with 1/kp.

**Two control modes.** `pd` sends `tau = 0`, as the vendor example does.
`pd_gff` sends the gravity torque at the commanded configuration, computed from
the nominal model by recursive Newton-Euler. This is the one improvement the
SDK interface allows without changing gains, and it is the mode a manipulation
stack would reasonably use.

**Not using the vendor bridge directly.** The bridge talks DDS and runs on
wall-clock threads, so two runs are never identical. Stepping MuJoCo directly
with the same law keeps trials deterministic.

**State feedback.** The controller reads joint position and velocity through an
encoder model (true value, plus offset, plus noise). The vendor bridge reads
`sensordata`, which holds the same quantities from the previous step.

**Motion.** Minimum-jerk joint interpolation over 2 s. The vendor example uses
a `tanh` blend over a similar time. No velocity feed-forward is sent, so the
hand is 10 to 20 mm behind when the commanded motion ends and catches up in the
dwell.

## 3. Targets and approaches

**Five targets in a cube.** A 250 mm cube centred on the hand position of the
joint configuration `[-0.5, -0.25, 0, -1.6, 0, 0, 0]` rad, hand in front of the
chest. P1 is the centre, P2 to P5 are the corners of a diagonal plane, inset by
10 % of the side. This is the ISO 9283 pose layout. The orientation is the same
for all five.

**One joint command per target.** Inverse kinematics is damped least squares
with damping proportional to the remaining error, seeded from the cube centre.
Residuals are below 1e-8 m, so the IK contributes nothing to the measured
error. The arm has one redundant degree of freedom; the solver returns the
solution nearest the seed and the same command is used for both approaches.
All commands keep at least 35 degrees from a joint limit.

**Two approaches.** Each trial starts at `q_cmd + d` ("plus") or `q_cmd - d`
("minus") with `d = [0.25, 0.12, 0.20, 0.25, 0.20, 0.15, 0.15]` rad. Every
joint therefore reaches its command from opposite sides in the two approaches,
which is what exposes friction. ISO 9283 uses three orthogonal Cartesian
approach directions; two joint-space ones were judged enough to show the effect.

## 4. Measurement protocol

**Fixed timing.** Hold the start pose 1 s, move 2 s, dwell 4 s, measure. No
settling threshold decides when to measure, so every trial simulates the same
7 s and the measurement instant cannot bias the result.

**Reset before every trial.** The simulation state is reset and the arm placed
at the start pose at rest. Trials are independent of each other and of run
order. The run order is still shuffled with a fixed seed, as it would be on
hardware.

**Ground truth.** The TCP pose is the site pose from MuJoCo, an ideal external
measuring device with no noise. Measurement noise was deliberately left out: it
would widen RP without the robot being any less repeatable.

**Is 4 s enough?** Without friction the hand is within 0.04 mm of its final
position at the measurement instant. With friction and encoder noise together
it is still creeping and moves another 0.07 to 0.2 mm over the following 12 s.
`scripts/check_settling.py` prints these numbers.

## 5. What makes the plant imperfect

A perfect simulator returns perfect numbers, so real-robot effects are added.
Magnitudes are assumptions, set in `config/experiment.yaml`.

| Effect | Model | Value | Expected signature |
|---|---|---|---|
| Encoder zero offset | controller reads `q + offset` | ±0.10 to 0.20 deg per joint, fixed | bias, no scatter |
| Unknown payload | point mass at the TCP, absent from the controller's model | 0.3 kg | bias that depends on pose |
| Joint friction | MuJoCo `frictionloss` with the no-slip solver | 0.2 N·m shoulder, 0.1 elbow, 0.03 wrist | depends on approach direction |
| Encoder noise | white noise on measured position and velocity | 0.005 deg, 0.002 rad/s | scatter |
| Start-pose jitter | random offset of the start configuration | 1 deg per joint (1 sigma) | scatter, if the plant has memory |

Offsets, friction and noise apply to the right arm only.

**Friction needs the no-slip solver.** With default settings MuJoCo's joint
friction is a soft constraint. The arm creeps through it with a time constant
of about 2 s and after 20 s sits where it would without friction. Real Coulomb
friction holds the joint anywhere inside a band of about ±friction/kp. Setting
`noslip_iterations = 200` and `noslip_tolerance = 1e-10` on plants with
friction reproduces that. Plants without friction keep the model defaults.

**Repeats on the ideal plant are identical.** MuJoCo is deterministic, and the
ideal plant has no random input, so its ten repetitions per cell coincide and
RP is zero. They are kept so the design stays balanced and as a standing
determinism check. Repetitions carry information only on stochastic plants.

## 6. Metrics

AP, RP and vAP use the ISO 9283:1998 definitions for position. Differences from
the standard: 10 cycles (not 30), 2 approach directions (not 3), ground-truth
measurement, orientation reported as a single rotation angle, no path metrics.
The results are therefore ISO-style and not ISO-compliant.

Four classic summaries of the per-trial error are reported next to them: mean
error, spread (one standard deviation), RMSE and success rate against the
declared tolerance. They are kept because they are what most people expect to
see, and because the comparison is informative in itself. Mean error can never
be smaller than AP, since AP lets opposing errors cancel; the gap between the
two is a direct measure of how much of the error is scatter rather than bias.
Spread is not the same as RP: spread is the variation of a scalar (the distance
to the target) while RP is spatial scatter in three dimensions, so points spread
evenly over a sphere centred on the target have spread zero and a large RP.
In this benchmark mean error and AP coincide within a millionth of a millimetre, which says
the error is entirely bias.

**Error per joint.** At the measurement instant each joint's error
`q_true - q_cmd` is multiplied by its Jacobian column, giving the TCP
displacement that joint causes to first order. The three waist joints are
combined. The first-order sum differs from the measured error by at most
4.5 mm at 256 mm.

**Tolerance.** 5 mm and 2 degrees are declared in the config for the
"within tolerance" column. REVEL gave no requirement; the values are
illustrative.

## 7. Static prediction

For each control mode, plant and target, the joint angles that satisfy

    tau_ff + kp * (q_cmd - q - offset) = g_plant(q)

are found by Newton's method, without simulating, and pushed through forward
kinematics. This prediction includes gravity, payload and encoder offsets and
excludes friction and noise. It serves two purposes: a known-answer check of
the whole pipeline, and a way to separate what statics explains from what
friction and noise add. The controller's model assumes open fingers; in the
simulation they sag by 1 to 14 milliradians, which is why the floor is 0.02 mm
and not zero.

## 8. Reproducibility

One seed in the config. The random numbers of a trial are derived from the
seed and the trial's identity (mode, condition, target, approach, repetition),
never from its position in the run. The pinned `pnd_mujoco` commit, MuJoCo
version and platform are written to `results/run_info_*.json`.

## 9. Not covered

Path accuracy and path repeatability. Orientation as a varied factor. Left arm,
fingers, grasping. Lifting-column height. Backlash, joint and link elasticity,
link-length and TCP calibration error, thermal drift, communication latency.
A simulated external sensor or a contact fixture as the measuring method.
Gain tuning: `scripts/gain_sensitivity.py` measures how the error changes when
the five soft joints are made stiffer, as a what-if. Stability margins, noise
amplification and behaviour on hardware at those gains were not examined.
