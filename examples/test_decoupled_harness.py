"""Analytic checks for the experimental interval sampler and event integration."""
from dataclasses import replace
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoupled_flight_harness import DecoupledFlight, Interval, Knot
from diagnostics.errors import TrialDomainError
from simulation_types import FluidOut, KinematicsState, PropulsionOut, SimResult


def knot(t, thrust=160., mode='combusting', mass=16., cg=1., inertia=2.):
    p = PropulsionOut(mode, None, thrust, 1., 1., 1., 1., 1., 1., 2.)
    f = FluidOut({}, {}, {}, p)
    return Knot(t, f, mass, mass*cg, inertia+mass*cg**2, 1., np.array([mass]))


def test_interpolation_mass_moments_and_event_sides():
    a = knot(0., mass=16., cg=1., inertia=2.)
    b = knot(1., mass=8., cg=2., inertia=3.)
    off = knot(1., 0., 'shutdown', mass=8., cg=2., inertia=3.)
    interval = Interval([a, b, off, replace(off, t=2.)])
    _, mass, cg, inertia, *_ = interval.sample(.5)
    assert mass == 12.
    assert cg == pytest.approx(16./12.)
    assert inertia + mass*cg**2 == pytest.approx((18.+35.)/2)
    assert interval.sample(1., 'left')[0].propulsion.thrust == 160.
    assert interval.sample(1., 'right')[0].propulsion.thrust == 0.
    assert interval.sample(1.5)[0].propulsion.mode == 'shutdown'
    with pytest.raises(ValueError):
        interval.sample(2.1)


@pytest.mark.parametrize('cutoff', [.1, .35, .7])
def test_event_impulse_rail_and_flight_retry_without_fluid_calls(cutoff):
    sim = object.__new__(DecoupledFlight)
    sim.cfg = {'launch': {'altitude': 0., 'rail_height': 2.}}
    sim.vehicle = SimpleNamespace()
    sim.env = SimpleNamespace(wind=lambda h: (0., 0.))
    sim.result = SimResult(history=[])
    sim.diag = dict(rk_rhs_calls=0, rk_accepted_steps=0, rk_rejected_steps=0, flight_domain_retries=0)
    sim.rk_rtol, sim.rk_max_step = 1e-9, 1.
    def forbidden(*args, **kwargs):
        raise AssertionError('Flight retry attempted a fluid solve')
    sim.update_fluid = forbidden
    rejected = [False]
    def evaluate(kin, fluid):
        if kin.t > .01 and not rejected[0]:
            rejected[0] = True
            raise TrialDomainError('Exercise flight-only recovery')
        return None, None, dict(ax=0., az=fluid.propulsion.thrust/kin.m-1., pitch_acceleration=0.)
    sim.evaluate_state = evaluate
    sim._record = lambda kin, fluid, rail, before=None: sim.result.history.append({'kinematics': kin})
    sim._collect_sample = lambda *args: {'before': True}
    on = knot(0.)
    off = knot(cutoff, 0., 'shutdown')
    interval = Interval([on, replace(on, t=cutoff), off, replace(off, t=1.)])
    kin = KinematicsState(0., 1., 0., 0., 0., 10., np.pi/2, 0., 0., 16., 2.)
    end, rail = sim._integrate(kin, interval, True, None)
    assert end.vz == pytest.approx(10.+10.*cutoff-1., abs=1e-8)
    assert end.h == pytest.approx(10.-.5+10.*(cutoff-.5*cutoff**2), abs=1e-8)
    assert sim.result.burn_duration == pytest.approx(cutoff, abs=1e-10)
    assert not rail
    assert sim.diag['flight_domain_retries'] == 1


def test_coast_apogee_localization():
    sim = object.__new__(DecoupledFlight)
    sim.cfg = {'launch': {'altitude': 0., 'rail_height': 0.}}
    sim.vehicle = SimpleNamespace()
    sim.env = SimpleNamespace(wind=lambda h: (0., 0.))
    sim.result = SimResult(history=[])
    sim.diag = dict(rk_rhs_calls=0, rk_accepted_steps=0, rk_rejected_steps=0, flight_domain_retries=0)
    sim.rk_rtol, sim.rk_max_step = 1e-9, 1.
    sim.evaluate_state = lambda kin, fluid: (None, None, dict(ax=0., az=-1., pitch_acceleration=0.))
    sim._record = lambda kin, fluid, rail, before=None: sim.result.history.append({'kinematics': kin})
    off = knot(0., 0., 'shutdown')
    interval = Interval([off, replace(off, t=5.)])
    kin = KinematicsState(0., 1., 0., 10., 0., 2., np.pi/2, 0., 0., 16., 2.)
    end, _ = sim._integrate(kin, interval, False, None)
    assert end.t == pytest.approx(2.)
    assert sim.result.apogee == pytest.approx(12.)
