import math
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from diagnostics.errors import TrialDomainError, ModelDomainExceeded
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
    return KinematicsState(1.0, 0.0, 10.0, 0.0, 10.0, math.pi / 2, 0.02)


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
    out = sim.trial_aero(atm, math.radians(30), False)
    assert out.ballistic_coast and np.isnan(out.cp) and np.isnan(out.Cd)
    assert out.A == out.N == out.D == 0
    fluids = sim.prop_system.update(None, atm, {})
    fluids.propulsion = replace(fluids.propulsion, thrust=0., mode="shutdown")
    forces = sim.forces(kin, out, fluids.propulsion, sim.mass_properties())
    assert forces["pitch_moment"] == forces["pitch_acceleration"] == 0
    next_kin = sim.predict_kinematics(kin, forces, .1)
    assert next_kin.q == kin.q
    assert next_kin.theta == pytest.approx(kin.theta + kin.q * .1)
    result = SimResult(history=[])
    sample = sim._collect_sample(result, kin, (atm, out, forces, math.radians(30)), fluids, None, sim.mass_properties(), False, True)
    loads = sample['loads']
    for key in ("axial", "axial_aero", "normal", "shear", "bending"):
        np.testing.assert_array_equal(loads[key], 0)
    sim._record_extrema(result, atm, out, math.radians(30), sim.vehicle.cg)
    assert result.max_aoa_deg == pytest.approx(30)
    assert "max_aoa_deg" not in result.constraint_records
    assert result.min_stability_calibers is None


def test_cutoff_boundary_and_domain_checks_resume():
    sim, kin = make_flight(), state()
    atm = replace(sim._atmosphere(kin), q=.001)
    assert sim.trial_aero(atm, math.radians(30), False).ballistic_coast
    with pytest.raises(TrialDomainError, match="AoA"):
        sim.trial_aero(replace(atm, q=.01), math.radians(30), False)
    sim.aero.mach = [0., 5.]
    with pytest.raises(TrialDomainError, match="Mach"):
        sim.trial_aero(replace(atm, q=.02, Ma=6), 0., False)
    assert not sim.trial_aero(replace(atm, q=.02), 0., False).ballistic_coast
    assert sim.trial_aero(atm, math.radians(30), False).ballistic_coast  # No latched state.


def test_disabled_cutoff_keeps_table_bounds_at_zero_pressure():
    sim, kin = make_flight(0), state()
    with pytest.raises(TrialDomainError, match="AoA"):
        sim.trial_aero(replace(sim._atmosphere(kin), q=0), math.radians(30), False)


def test_nonfinite_state_and_explicit_constraint_still_rejected():
    sim, kin = make_flight(), state()
    atm = replace(sim._atmosphere(kin), q=.001)
    with pytest.raises(TrialDomainError):
        sim.trial_aero(atm, float("nan"), False)
    with pytest.raises(TrialDomainError):
        sim.trial_aero(replace(atm, q=-1), math.radians(30), False)
    sim.cfg["constraints"] = {"max_aoa_deg": 15.}
    with pytest.raises(TrialDomainError):
        sim.trial_aero(replace(atm, q=1.), math.radians(30), False)


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
