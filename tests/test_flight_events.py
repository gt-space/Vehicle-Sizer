"""Analytic flight events, one-sided propulsion, and transactional plant advances."""
from copy import deepcopy
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from Flight.Flight import FlightSim
from diagnostics.errors import TrialDomainError
from simulation_types import ThermalOut, history_samples
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


class TimedPropulsion(FakePropSystem):
    def __init__(self, cutoff=float('inf')):
        super().__init__()
        self.cutoff = cutoff

    def update(self, dt, atm, heat_rate, commit=True, **kwargs):
        checkpoint = self.checkpoint()
        start, end = self.time, self.time + (dt or 0.)
        out = super().update(dt, atm, heat_rate, commit, **kwargs)
        def at(t, powered):
            sample = deepcopy(replace(out, samples=(), events=()))
            sample.node['tank'].update(mass=5. - min(t, self.cutoff), axial_mass=[5. - min(t, self.cutoff)])
            sample.propulsion = replace(sample.propulsion, thrust=160. if powered else 0.,
                mode='combusting' if powered else 'shutdown', shutdown_reason=None if powered else 'test_cutoff')
            return sample
        samples = [(start, at(start, start < self.cutoff))]
        events = ()
        if start < self.cutoff <= end:
            before, after = at(self.cutoff, True), at(self.cutoff, False)
            events = ({'event': 'shutdown', 'time_s': self.cutoff, 'before': before},)
            samples += [(self.cutoff, before), (self.cutoff, after)]
        out = at(end, end < self.cutoff)
        samples.append((end, out))
        result = replace(out, events=events, samples=tuple(samples))
        if not commit:
            self.restore(checkpoint)
        return result


class ConstantVehicle(FakeVehicle):
    def update_mass_distribution(self, nodes):
        super().update_mass_distribution(nodes)
        self.total_mass = 16.
        self.mass[:] = 8.


class ZeroAlpha(FakeAero):
    pass


class IntegratingThermal:
    def __init__(self):
        self.temperature, self.commits = 300., []

    def trial(self, dt, *args, **kwargs):
        return ThermalOut({'tank': {'cells': {'wall_T': np.array([self.temperature + dt])},
                                   'phases': {'liquid': {'heat_rate': 0.}}}})

    def commit(self, out):
        self.temperature = out.node['tank']['cells']['wall_T'][0]
        self.commits.append(self.temperature)


def flight(monkeypatch, *, cutoff=float('inf'), rail=0., thermal=None, aero=None):
    monkeypatch.setattr(import_module('Flight.Flight'), 'gravity', lambda mass, height: mass)
    return FlightSim({'simulation': {'dt': 1., 't_end': 1., 'fluid_solve_post_shutdown': False},
                      'launch': {'altitude': 0., 'rail_height': rail}},
                     FakeEnvironment(), aero or ZeroAlpha(), TimedPropulsion(cutoff),
                     ConstantVehicle(), thermal)


@pytest.mark.parametrize('cutoff', [.1, .2, .35, 1.])
def test_rail_and_cutoff_order_preserves_burn_and_impulse(monkeypatch, cutoff):
    rail_time = .2
    powered = min(cutoff, rail_time)
    rail_height = 10*rail_time + 9*(.5*powered**2 + powered*(rail_time-powered)) - .5*(rail_time-powered)**2
    sim = flight(monkeypatch, cutoff=cutoff, rail=rail_height)
    history = sim.run(v0=10., compute_loads=False)
    assert sim.result.burn_duration == pytest.approx(cutoff, abs=1e-7)
    assert sim.result.rail_exit_time == pytest.approx(rail_time, abs=1e-7)
    assert sim.result.final_velocity == pytest.approx(9 + 10*cutoff, abs=1e-7)
    expected_h = 10 + 9*(.5*cutoff**2 + cutoff*(1-cutoff)) - .5*(1-cutoff)**2
    assert sim.result.final_altitude == pytest.approx(expected_h, abs=1e-7)
    assert sim.result.burn_complete
    assert sim.prop_system.time == 1.
    assert len(sim.prop_system.calls) == 2  # No fluid re-solves for either event.
    assert all(row['plant'].fluids.propulsion.thrust == 0 for row in history if row['kinematics'].t >= cutoff)
    assert len([e for row in history for e in row['plant'].fluids.events]) == 1


def test_flight_event_does_not_repeat_wall_or_fluid_advance(monkeypatch):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, rail=2.18, thermal=thermal)
    sim.run(v0=10., compute_loads=False)
    assert sim.result.rail_exit_time == pytest.approx(.2, abs=1e-7)
    assert thermal.commits == [301.]
    assert sim.prop_system.time == 1.
    assert [h['kinematics'].t for h in sim.result.history] == [1.]


def test_domain_retry_reuses_solved_fluid_and_commits_once(monkeypatch):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, thermal=thermal)
    original = sim.rhs
    rejected = []
    def rhs(kin, *args):
        if not rejected and kin.t > .1:
            rejected.append(kin.t)
            raise TrialDomainError('test invalid stage')
        return original(kin, *args)
    monkeypatch.setattr(sim, 'rhs', rhs)
    sim.run(v0=10., compute_loads=False)
    assert rejected
    assert sim.result.final_altitude == pytest.approx(14.5)
    assert len(sim.prop_system.calls) == 2  # preview restores its private call log
    assert thermal.commits == [301.]


def test_exhausted_trial_restores_plant_and_leaves_no_output(monkeypatch):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, thermal=thermal)
    original = sim.rhs
    def rhs(kin, *args):
        if kin.t > .1:
            raise TrialDomainError('test invalid stage')
        return original(kin, *args)
    monkeypatch.setattr(sim, 'rhs', rhs)
    with pytest.raises(RuntimeError, match='recovery exhausted'):
        sim.run(v0=10., compute_loads=False)
    assert sim.prop_system.time == 0
    assert thermal.commits == [] and thermal.temperature == 300.
    assert sim.result.history == [] and sim.result.burn_duration == 0.


@pytest.mark.parametrize('powered', [False, True])
def test_terminal_apogee_synchronizes_live_plant_and_wall(monkeypatch, powered):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, cutoff=float('inf') if powered else 0., thermal=thermal)
    if powered:
        monkeypatch.setattr(import_module('Flight.Flight'), 'gravity', lambda mass, height: 11 * mass)
    sim.cfg['simulation']['fluid_solve_post_shutdown'] = True
    history = sim.run(v0=.5, compute_loads=False)
    assert sim.result.apogee_time == pytest.approx(.5, abs=1e-7)
    assert sim.result.apogee == pytest.approx(.125, abs=1e-7)
    assert sim.prop_system.time == pytest.approx(sim.result.apogee_time, abs=1e-8)
    assert thermal.temperature == pytest.approx(300.5, abs=1e-7)
    assert len(thermal.commits) == 1
    assert history[-1]['kinematics'].vz == 0.
    assert sim.result.burn_duration == pytest.approx(.5 if powered else 0., abs=1e-7)
    assert history[-1]['plant'].fluids.node['tank']['mass'] == pytest.approx(4.5 if powered else 5.)


def test_ascent_only_aero_never_queries_reversed_flow(monkeypatch):
    class AscentOnly(ZeroAlpha):
        @staticmethod
        def evaluate(atm, alpha, engine_on):
            assert abs(np.degrees(alpha)) <= 15
            return ZeroAlpha.evaluate(atm, alpha, engine_on)
    sim = flight(monkeypatch, cutoff=0., aero=AscentOnly())
    sim.run(v0=.5, compute_loads=False)
    assert sim.result.apogee == pytest.approx(.125)


def test_post_shutdown_flow_continues_when_enabled(monkeypatch):
    sim = flight(monkeypatch, cutoff=.35)
    sim.cfg['simulation'].update(fluid_solve_post_shutdown=True, t_end=2.)
    sim.run(v0=10., compute_loads=False)
    assert sim.prop_system.time == 2.
    assert len(sim.prop_system.calls) == 3
    assert sim.result.burn_duration == pytest.approx(.35)


def test_real_sundials_event_snapshots_reach_flight(monkeypatch):
    pytest.importorskip('sundials4py')
    from test_prop_system import build, config
    cfg = config('pump_fed', 'regulator')
    cfg['simulation'] = {'dt': .2, 't_end': .4, 'fluid_solve_post_shutdown': False}
    cfg['launch'] = {'altitude': 0., 'rail_height': .1}
    cfg['advanced'] = {'fluid_network': {'max_step': .01}}
    with build(cfg, tabled=True) as prop:
        sim = FlightSim(cfg, FakeEnvironment(), ZeroAlpha(), prop, ConstantVehicle())
        history = sim.run(v0=10., compute_loads=False)
        shutdown = [e for h in history for e in h['plant'].fluids.events if e.get('event') == 'shutdown']
        assert len(shutdown) == 1
        assert shutdown[0]['before'].propulsion.thrust > 0
        assert sim.result.burn_duration == pytest.approx(shutdown[0]['time_s'], abs=1e-6)
        assert sim.result.rail_exit_time < sim.result.burn_duration < .2
        assert prop.network.frozen
        assert history[-1]['kinematics'].t == .4


def test_shared_reporting_preserves_cutoff_and_omits_internal_samples(monkeypatch, tmp_path):
    import csv
    from main import write_events
    from reporting.history import snapshot_row, capture_history
    from reporting.plot_peak_loads import plot_loads
    sim = flight(monkeypatch, cutoff=.35)
    sim.run(v0=10., compute_loads=True)
    samples = list(history_samples(sim.result))
    assert [s['kinematics'].t for s in samples] == pytest.approx([0, .35, .35, 1.])
    assert samples[1]['engine_on'] and not samples[2]['engine_on']
    assert samples[1]['forces']['thrust'] == 160. and samples[2]['forces']['thrust'] == 0.
    assert not any('samples[' in key or 'before_event' in key for key in snapshot_row(samples[2]))
    assert capture_history(sim.result)[1]['propulsion.total_impulse_sampled_Ns'] == pytest.approx(56.)
    path = tmp_path / 'events.csv'
    write_events(sim.result.history, path)
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1 and float(records[0]['time_s']) == .35
    sim.result.design_summary = {'section': {}, 'length': sim.vehicle.length}
    plot_loads(sim.result, tmp_path)
    assert np.load(tmp_path / 'burn_load_history.npz')['time_s'] == pytest.approx([0., .35])


@pytest.mark.parametrize('setting,value', [('rtol', float('nan')), ('position_atol', -1),
    ('max_step', 0), ('event_max_iterations', True), ('trial_max_retries', 1.5)])
def test_invalid_settings_fail_before_fluid_initialization(monkeypatch, setting, value):
    sim = flight(monkeypatch)
    sim.cfg['advanced'] = {'flight': {setting: value}}
    with pytest.raises(ValueError, match='Flight'):
        sim.run(v0=10., compute_loads=False)
    assert sim.prop_system.calls == []


def test_interval_interpolates_mass_moments_and_event_sides(monkeypatch):
    from Flight.integration import sample_interval
    sim = flight(monkeypatch, cutoff=.35)
    initial = sim.update_fluid(None, FakeEnvironment.atmosphere(0., 10.), commit=True)
    final = sim.update_fluid(1., FakeEnvironment.atmosphere(10., 10.), commit=True)
    a = dict(total_mass=4., cg=2., Iyy=8.)
    b = dict(total_mass=2., cg=4., Iyy=10.)
    samples = [(0., initial, a), (1., final, b)]
    _, props = sample_interval(.5, samples)
    assert props['total_mass'] == 3.
    assert props['cg'] == pytest.approx(8/3)
    assert props['Iyy'] == pytest.approx((24+42)/2 - 3*(8/3)**2)
    prepared = sim.prepare_samples(final)
    assert sample_interval(.35, prepared, 'left')[0].thrust == 160.
    assert sample_interval(.35, prepared, 'right')[0].thrust == 0.
    assert sample_interval(.7, prepared)[0].thrust == 0.


def test_adaptive_flight_converges_without_additional_fluid_solves(monkeypatch):
    def run(rtol):
        sim = flight(monkeypatch)
        sim.cfg['advanced'] = {'flight': dict(rtol=rtol, position_atol=rtol, velocity_atol=rtol,
                                             angle_atol=rtol, angular_rate_atol=rtol)}
        # Smooth analytic growing vertical velocity: dv/dt=v, h=e^t-1.
        def rhs(kin, samples, rail, side):
            return [0., kin.vz, 0., kin.vz, 0., 0.]
        sim.rhs = rhs
        sim.run(v0=1., compute_loads=False)
        return sim
    loose, tight = run(1e-3), run(1e-9)
    assert abs(tight.result.final_velocity - np.e) < abs(loose.result.final_velocity - np.e)
    assert tight.result.final_altitude == pytest.approx(np.e - 1, abs=2e-9)
    assert len(loose.prop_system.calls) == len(tight.prop_system.calls) == 2
