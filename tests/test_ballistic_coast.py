import math
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from diagnostics.constraints import OperatingInfeasible
from diagnostics.errors import TrialDomainError
from Flight.Flight import FlightSim
from simulation_types import KinematicsState, PlantOut, SimResult
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle
from test_flight_events import TimedPropulsion


def make_flight(cutoff=0.01):
    cfg = {"aero": {"q_cutoff_Pa": cutoff},
           "simulation": {"dt": .1, "t_end": 10., "fluid_solve_post_shutdown": False},
           "launch": {"altitude": 0., "rail_height": 0.}}
    return FlightSim(cfg, FakeEnvironment(), FakeAero(), FakePropSystem(), FakeVehicle())


def state():
    return KinematicsState(1., .1, 0., 10., 0., 10., math.pi / 2, .02,
                           math.radians(30), 10., 3.)


@pytest.mark.parametrize("cutoff", [-1, float("nan"), float("inf")])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError, match="q_cutoff_Pa"):
        make_flight(cutoff)


def test_low_pressure_bypasses_all_tables_and_preserves_kinematics():
    sim, kin = make_flight(), state()
    sim.aero.evaluate = Mock(side_effect=AssertionError("table queried"))
    sim.aero.axial_distribution = Mock(side_effect=AssertionError("axial table queried"))
    sim.aero.normal_distribution = Mock(side_effect=AssertionError("normal table queried"))
    atm = replace(sim._atmosphere(kin), q=.001, Ma=100)
    out = sim.trial_aero(kin, atm, False)
    assert out.ballistic_coast and np.isnan(out.cp) and np.isnan(out.Cd)
    assert out.A == out.N == out.D == 0
    assert math.degrees(kin.alpha) == pytest.approx(30)
    fluids = sim.prop_system.update(None, atm, {})
    fluids.propulsion = replace(fluids.propulsion, thrust=0., mode="shutdown")
    forces = sim.forces(kin, PlantOut(out, None, fluids), kin.m)
    assert forces["pitch_moment"] == forces["pitch_acceleration"] == 0
    next_kin = sim.correct_kinematics(kin, forces, forces, kin.m, kin.Iyy)
    assert next_kin.q == kin.q
    assert next_kin.theta == pytest.approx(kin.theta + kin.q * kin.dt)
    result = SimResult()
    loads = sim._evaluate_loads(result, kin, atm, out, forces, False)
    for key in ("axial", "axial_aero", "normal", "shear", "bending"):
        np.testing.assert_array_equal(loads[key], 0)
    sim._record_extrema(result, atm, out, kin)
    assert result.max_aoa_deg == pytest.approx(30)
    assert "max_aoa_deg" not in result.constraints
    assert result.min_stability_calibers is None


def test_cutoff_boundary_and_domain_checks_resume():
    sim, kin = make_flight(), state()
    atm = replace(sim._atmosphere(kin), q=.001)
    assert sim.trial_aero(kin, atm, False).ballistic_coast
    with pytest.raises(TrialDomainError, match="AoA"):
        sim.trial_aero(kin, replace(atm, q=.01), False)
    sim.aero.mach = [0., 5.]
    with pytest.raises(TrialDomainError, match="Mach"):
        sim.trial_aero(replace(kin, alpha=0), replace(atm, q=.02, Ma=6), False)
    assert not sim.trial_aero(replace(kin, alpha=0), replace(atm, q=.02), False).ballistic_coast
    assert sim.trial_aero(kin, atm, False).ballistic_coast  # No latched state.


def test_disabled_cutoff_keeps_table_bounds_at_zero_pressure():
    sim, kin = make_flight(0), state()
    with pytest.raises(TrialDomainError, match="AoA"):
        sim.trial_aero(kin, replace(sim._atmosphere(kin), q=0), False)


def test_nonfinite_state_and_explicit_constraint_still_rejected():
    sim, kin = make_flight(), state()
    atm = replace(sim._atmosphere(kin), q=.001)
    with pytest.raises(TrialDomainError):
        sim.trial_aero(replace(kin, alpha=float("nan")), atm, False)
    with pytest.raises(TrialDomainError):
        sim.trial_aero(kin, replace(atm, q=-1), False)
    sim.cfg["constraints"] = {"max_aoa_deg": 15.}
    with pytest.raises(OperatingInfeasible):
        sim.check_commit(kin, kin)


def test_ballistic_run_reaches_apogee_outside_aero_domain():
    sim = make_flight()
    sim.prop_system = TimedPropulsion(cutoff=0.)
    sim.env.wind = lambda h: (5., 0.)
    sim.env.atmosphere = lambda h, v: replace(FakeEnvironment.atmosphere(h, v), q=.001)
    sim.aero.evaluate = Mock(side_effect=AssertionError("table queried"))
    sim.aero.axial_distribution = Mock(side_effect=AssertionError("table queried"))
    sim.aero.normal_distribution = Mock(side_effect=AssertionError("table queried"))
    history = sim.run(v0=10.)
    assert sim.result.apogee_reached
    assert sim.result.max_aoa_deg > 15
    assert all(s["plant"].aero.ballistic_coast for s in history)
    assert all(np.isfinite(s["forces"]["az"]) for s in history)
