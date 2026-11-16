"""
Simulation tests on the real ADAM-U model.  They are skipped when the model has
not been fetched yet (bash scripts/setup_env.sh).  About 15 s in total.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiment import Setup, load_config, resolve_model_path  # noqa: E402
from src.kinematics import fk, gravity_torque, static_equilibrium  # noqa: E402
from src.model_builder import ARM_SLICE, MOTOR_JOINTS  # noqa: E402
from src.trial import run_trial  # noqa: E402

MODEL = resolve_model_path(load_config())
pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL), reason=f"ADAM-U model not found at {MODEL}")


@pytest.fixture(scope="module")
def setup():
    return Setup()


def trial(setup, mode, cond_name, target, approach, seed=0):
    cond = setup.condition(cond_name)
    plant, data = setup.plant_for(cond)
    rng = np.random.default_rng(seed) if cond.is_stochastic else None
    return run_trial(plant, data, setup.controller(mode), cond, setup.start_config(target, approach),
                     target.q_cmd, setup.cfg["protocol"], rng, sim_cfg=setup.cfg["simulation"])


def test_model_is_fixed_base_with_a_tcp_site(setup):
    m = setup.nominal.model
    assert m.nq == m.nv, "a free joint is still present"
    assert setup.nominal.tcp_site_id >= 0
    assert len(setup.nominal.act) == len(MOTOR_JOINTS) == 19
    assert (setup.nominal.torque_limit > 0).all()


def test_vendored_files_are_not_modified(setup):
    with open(os.path.join(os.path.dirname(MODEL), "adam_u.xml")) as f:
        text = f.read()
    assert "tcp_right" not in text and "floating_base" in text


def test_payload_changes_mass_but_not_structure(setup):
    plant, _ = setup.plant_for(setup.condition("payload_only"))
    nominal = setup.nominal.model
    assert plant.model.nv == nominal.nv and plant.model.nu == nominal.nu
    assert plant.model.body_mass.sum() - nominal.body_mass.sum() == pytest.approx(
        setup.condition("payload_only").payload_kg)


def test_targets_are_reached_by_ik_and_form_the_cube(setup):
    for t in setup.targets:
        q19 = np.zeros(19)
        q19[ARM_SLICE] = t.q_cmd
        pos, _ = fk(setup.nominal, setup.scratch, q19)
        assert np.linalg.norm(pos - t.pos) < 1e-6
        assert t.limit_margin_rad > np.radians(15)
    a = 0.4 * setup.cfg["targets"]["cube_side_m"]
    assert np.allclose(np.abs(setup.targets[1].pos - setup.targets[0].pos), a)


def test_controller_implements_the_vendor_law(setup):
    cond = setup.condition("nominal")
    plant, data = setup.plant_for(cond)
    rng = np.random.default_rng(5)
    data.qpos[plant.qadr] = rng.uniform(-0.1, 0.1, 19)
    data.qvel[plant.dadr] = rng.uniform(-0.1, 0.1, 19)
    q_cmd = rng.uniform(-0.1, 0.1, 19)
    tau_ff = rng.uniform(-1, 1, 19)
    tau = setup.controller("pd").torque(plant, data, q_cmd, tau_ff, cond, None)
    for i in range(19):
        expected = tau_ff[i] + setup.kp[i] * (q_cmd[i] - data.qpos[plant.qadr[i]]) \
            + setup.kd[i] * (0.0 - data.qvel[plant.dadr[i]])
        assert tau[i] == pytest.approx(np.clip(expected, -plant.torque_limit[i], plant.torque_limit[i]))


def test_gravity_feedforward_makes_the_command_an_equilibrium(setup):
    t = setup.targets[0]
    q_cmd = np.zeros(19)
    q_cmd[ARM_SLICE] = t.q_cmd
    tau_ff = gravity_torque(setup.nominal, setup.scratch, q_cmd)
    q_eq = static_equilibrium(setup.nominal, setup.scratch, q_cmd, tau_ff, setup.kp, np.zeros(19))
    assert np.abs(q_eq - q_cmd).max() < 1e-9


def test_same_trial_twice_is_bit_identical_and_order_does_not_matter(setup):
    t0, t1 = setup.targets[0], setup.targets[3]
    a = trial(setup, "pd_gff", "perturbed", t0, "plus", seed=7)
    trial(setup, "pd", "perturbed", t1, "minus", seed=8)          # something else in between
    b = trial(setup, "pd_gff", "perturbed", t0, "plus", seed=7)
    assert np.array_equal(a.tcp_pos, b.tcp_pos)
    assert np.array_equal(a.q_true19, b.q_true19)


def test_ideal_plant_with_feedforward_reaches_the_target(setup):
    t = setup.targets[1]
    res = trial(setup, "pd_gff", "nominal", t, "plus")
    assert np.linalg.norm(res.tcp_pos - t.pos) < 1e-4            # below 0.1 mm
    assert not res.saturated


@pytest.mark.parametrize("mode,cond", [("pd", "nominal"), ("pd_gff", "encoder_offset_only"),
                                       ("pd_gff", "payload_only")])
def test_simulation_agrees_with_static_prediction(setup, mode, cond):
    """Known-answer test: without friction and noise the end pose must match an
    independent static calculation."""
    t = setup.targets[2]
    res = trial(setup, mode, cond, t, "minus")
    predicted = setup.static_prediction(t, setup.condition(cond), mode)
    assert np.linalg.norm(predicted - t.pos) > 5e-4              # the injected error is not trivial (> 0.5 mm)
    assert np.linalg.norm(res.tcp_pos - predicted) < 1e-4        # and the simulation recovers it to 0.1 mm


def test_friction_sticks_and_depends_on_the_approach(setup):
    t = setup.targets[0]
    plus = trial(setup, "pd_gff", "friction_only", t, "plus")
    minus = trial(setup, "pd_gff", "friction_only", t, "minus")
    # With MuJoCo's default (soft) friction the arm creeps about 1 mm in the last
    # second of the dwell and the two approaches converge; with the no-slip
    # solver it stays put, several millimetres from the target.
    last_second = plus.trace_dev_mm[plus.trace_t > setup.cfg["protocol"]["dwell_s"] - 1.0]
    assert last_second.max() < 0.1, "the arm is still creeping through friction"
    assert np.linalg.norm(plus.tcp_pos - t.pos) > 5e-3           # more than 5 mm from the target
    assert np.linalg.norm(plus.tcp_pos - minus.tcp_pos) > 5e-3   # and the approaches end more than 5 mm apart
