"""Event boundaries must conserve trial state and integrate one-sided forces."""
from copy import deepcopy
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from Flight.Flight import FlightSim
from simulation_types import AeroOut, ThermalOut
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


class TimedPropulsion(FakePropSystem):
    def __init__(self, cutoff=float('inf')):
        super().__init__()
        self.time, self.cutoff = 0., cutoff

    def update(self, dt, atm, heat_rate, commit=True, **kwargs):
        snapshot = self.checkpoint()
        start = self.time
        self.time += dt or 0.
        out = super().update(dt, atm, heat_rate, commit, **kwargs)
        out.node['tank']['mass'] = 5. - min(self.time, self.cutoff)
        out.node['tank']['axial_mass'] = [out.node['tank']['mass']]
        out.propulsion = replace(out.propulsion, thrust=160.)
        if self.time >= self.cutoff:
            before = deepcopy(out)
            before.node['tank']['mass'] = 5. - self.cutoff
            before.node['tank']['axial_mass'] = [5. - self.cutoff]
            out.propulsion = replace(out.propulsion, mode='shutdown', thrust=0.,
                                     shutdown_reason='test_cutoff')
            if start < self.cutoff:
                out.events = ({'event': 'shutdown', 'time_s': self.cutoff, 'before': before},)
        if not commit:
            self.restore(snapshot)
        return out


class ConstantVehicle(FakeVehicle):
    def update_mass_distribution(self, nodes):
        super().update_mass_distribution(nodes)
        self.total_mass = 16.
        self.mass[:] = 8.


class ZeroAlpha(FakeAero):
    @staticmethod
    def aoa(time):
        return 0.


class IntegratingThermal:
    def __init__(self):
        self.temperature, self.commits = 300., []

    def trial(self, kin, *args, **kwargs):
        return ThermalOut({'tank': {'cells': {'wall_T': np.array([self.temperature + kin.dt])},
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
def test_rail_and_cutoff_order_burn_and_impulse(monkeypatch, cutoff):
    rail_time = .2
    powered = min(cutoff, rail_time)
    rail_height = (10 * rail_time + 9 * (.5 * powered**2 + powered * (rail_time - powered))
                   - .5 * (rail_time - powered)**2)
    sim = flight(monkeypatch, cutoff=cutoff, rail=rail_height)
    history = sim.run(v0=10., compute_loads=False)
    assert sim.result.burn_duration == pytest.approx(cutoff, abs=1e-6)
    assert sim.result.rail_exit_time == pytest.approx(rail_time, abs=2e-6)
    assert sim.result.final_velocity == pytest.approx(10 + 10 * cutoff - 1, abs=2e-5)
    expected_h = 10 + 9*(.5*cutoff**2 + cutoff*(1-cutoff)) - .5*(1-cutoff)**2
    assert sim.result.final_altitude == pytest.approx(expected_h, abs=2e-5)
    assert sim.result.burn_complete
    assert history[-1]['kinematics'].t == 1.
    events = [e for row in history for e in row['plant'].fluids.events]
    assert len(events) == 1
    assert history[-1]['plant'].fluids.node['tank']['mass'] == pytest.approx(5-cutoff)
    assert all(row['plant'].fluids.propulsion.thrust == 0
               for row in history if row['kinematics'].t >= cutoff)


def test_rail_trials_do_not_commit_mass_heat_or_result_samples(monkeypatch):
    class DiscontinuousAero(ZeroAlpha):
        @staticmethod
        def aoa(time):
            return .1

        @staticmethod
        def evaluate(kin, atmosphere, engine_on):
            return AeroOut(Cd=0., D=0. if kin.alpha == 0 else 50., cp=1.5)
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, rail=2.18, thermal=thermal, aero=DiscontinuousAero())
    history = sim.run(v0=10., compute_loads=False)
    assert sim.result.rail_exit_time == pytest.approx(.2, abs=1e-6)
    assert [h['kinematics'].t for h in history] == pytest.approx([.2, 1.], abs=1e-6)
    assert len(thermal.commits) == 2
    assert thermal.temperature == pytest.approx(301.)
    assert sim.prop_system.time == pytest.approx(1.)
    assert history[-1]['plant'].fluids.node['tank']['mass'] == pytest.approx(4.)
    assert sim.result.burn_duration == pytest.approx(1.)
    assert all(abs(h['kinematics'].alpha) < 1e-12 for h in history)


def test_exhausted_localization_restores_interval_and_thermal(monkeypatch):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, rail=2.18, thermal=thermal)
    sim.cfg['advanced'] = {'flight': {'event_max_iterations': 1}}
    with pytest.raises(RuntimeError, match='event localization failed'):
        sim.run(v0=10., compute_loads=False)
    assert sim.prop_system.time == 0
    assert sim.prop_system.calls[-1]['dt'] is None
    assert thermal.temperature == 300. and thermal.commits == []
    assert sim.result.burn_duration == 0. and sim.result.history == []


def test_real_sundials_shutdown_snapshot_reaches_flight(monkeypatch):
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
        assert 0 < sim.result.burn_duration < .2
        assert sim.result.rail_exit_time < sim.result.burn_duration
        assert prop.network.frozen
        assert history[-1]['kinematics'].t == .4
        assert history[-1]['plant'].fluids.propulsion.thrust == 0.


def test_post_shutdown_flow_is_still_advanced_when_enabled(monkeypatch):
    sim = flight(monkeypatch, cutoff=.35)
    sim.cfg['simulation']['fluid_solve_post_shutdown'] = True
    sim.run(v0=10., compute_loads=False)
    assert sim.prop_system.time == pytest.approx(1.)
    assert sim.result.burn_duration == pytest.approx(.35, abs=1e-6)
    assert sim.result.final_velocity == pytest.approx(12.5, abs=2e-5)


def test_event_snapshots_do_not_break_csv_export(monkeypatch, tmp_path):
    from main import write_events
    import csv
    sim = flight(monkeypatch, cutoff=.35)
    history = sim.run(v0=10., compute_loads=False)
    path = tmp_path / 'events.csv'
    write_events(history, path)
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1
    assert records[0]['event'] == 'shutdown'
    assert float(records[0]['time_s']) == .35
    assert 'before' not in records[0]


def test_apogee_is_localized_before_reversed_airflow_query(monkeypatch):
    class AscentOnly(ZeroAlpha):
        @staticmethod
        def evaluate(kin, atmosphere, engine_on):
            assert abs(np.degrees(kin.alpha)) <= 15
            return ZeroAlpha.evaluate(kin, atmosphere, engine_on)

    sim = flight(monkeypatch, cutoff=0., aero=AscentOnly())
    history = sim.run(v0=.5, compute_loads=False)
    assert sim.result.apogee_time == pytest.approx(.5, abs=1e-6)
    assert sim.result.apogee == pytest.approx(.125, abs=1e-6)
    assert history[-1]['kinematics'].vz >= 0
    assert sim.result.termination == 'apogee'


@pytest.mark.parametrize('angle', [-16., 16.])
def test_aoa_exceedance_is_a_signed_operating_constraint(monkeypatch, angle):
    from constraints import OperatingInfeasible, finalize
    from errors import TrialDomainError
    from simulation_types import KinematicsState, SimResult
    sim = flight(monkeypatch)
    kin = KinematicsState(t=2., dt=1., x=0., h=10., vx=0., vz=10.,
                          theta=np.pi/2 + np.radians(angle), q=0.,
                          alpha=np.radians(angle), m=16., Iyy=3.)
    with pytest.raises(TrialDomainError):
        sim.trial_aero(kin, FakeEnvironment.atmosphere(kin.h, kin.vz), True)
    with pytest.raises(OperatingInfeasible) as caught:
        sim.check_commit(kin, kin)
    assert caught.value.constraints['max_aoa_deg'] == pytest.approx(-1.)
    assert caught.value.time == 2.
    result = finalize(SimResult(termination='infeasible_operating_state',
                               constraints=caught.value.constraints,
                               constraint_times={'max_aoa_deg': 2.}), {'max_aoa_deg': 15.})
    record = result.constraint_records['max_aoa_deg']
    assert record.margin == pytest.approx(-1.)
    assert (record.source, record.units, record.time) == ('Flight.Flight', 'deg', 2.)
    assert result.feasible is False and not result.accepted


@pytest.mark.parametrize('stage', ['predictor', 'corrector'])
def test_invalid_aero_trial_retries_without_committing(monkeypatch, stage):
    thermal = IntegratingThermal()
    seen = []

    class GuardedAero(ZeroAlpha):
        @staticmethod
        def evaluate(kin, atmosphere, engine_on):
            seen.append(abs(np.degrees(kin.alpha)))
            assert seen[-1] <= 15
            return ZeroAlpha.evaluate(kin, atmosphere, engine_on)

    sim = flight(monkeypatch, thermal=thermal, aero=GuardedAero())
    name = 'predict_kinematics' if stage == 'predictor' else 'correct_kinematics'
    original = getattr(sim, name)

    def overshoot(kin, *args):
        endpoint = original(kin, *args)
        return replace(endpoint, theta=np.pi/2 + np.radians(20)) if kin.dt > .5 else endpoint

    monkeypatch.setattr(sim, name, overshoot)
    history = sim.run(v0=10., compute_loads=False)
    assert [row['kinematics'].t for row in history] == pytest.approx([.5, 1.])
    assert sim.prop_system.time == pytest.approx(1.)
    assert sim.result.burn_duration == pytest.approx(1.)
    assert thermal.temperature == pytest.approx(301.)
    assert len(thermal.commits) == 2
    assert max(seen) <= 15


def test_trial_recovery_is_bounded_and_restores_state(monkeypatch):
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, thermal=thermal)
    sim.cfg['advanced'] = {'flight': {'trial_max_retries': 2}}
    original = sim.correct_kinematics
    calls = []

    def invalid(kin, *args):
        calls.append(kin.dt)
        return replace(original(kin, *args), theta=np.pi/2 + np.radians(20))

    monkeypatch.setattr(sim, 'correct_kinematics', invalid)
    with pytest.raises(RuntimeError, match='trial recovery exhausted'):
        sim.run(v0=10., compute_loads=False)
    assert calls == pytest.approx([1., .5, .25])
    assert sim.prop_system.time == 0
    assert sim.result.history == [] and sim.result.burn_duration == 0
    assert thermal.temperature == 300. and thermal.commits == []


def test_mission_aoa_limit_is_checked_only_after_convergence(monkeypatch):
    from constraints import OperatingInfeasible
    from simulation_types import KinematicsState
    sim = flight(monkeypatch)
    sim.cfg['constraints'] = {'max_aoa_deg': 5.}
    kin = KinematicsState(1., .1, 0., 10., 0., 10., np.pi/2 + np.radians(10),
                          0., np.radians(10), 16., 3.)
    sim.trial_aero(kin, sim._atmosphere(kin), True)
    with pytest.raises(OperatingInfeasible):
        sim.check_commit(kin, kin)
    # Descent is a stopping condition before the reversed-flow AoA check.
    event = sim.check_commit(kin, replace(kin, t=1.1, vz=-1., alpha=np.pi))
    assert event['event'] == 'apogee'


def test_converged_violation_rolls_back_before_commit(monkeypatch):
    from constraints import OperatingInfeasible
    thermal = IntegratingThermal()
    sim = flight(monkeypatch, thermal=thermal)
    sim.cfg['constraints'] = {'max_aoa_deg': 5.}
    original = sim.correct_kinematics

    def corrected(kin, *args):
        return replace(original(kin, *args), theta=np.pi/2 + np.radians(10))

    monkeypatch.setattr(sim, 'correct_kinematics', corrected)
    with pytest.raises(OperatingInfeasible) as error:
        sim.run(v0=10., compute_loads=False)
    assert error.value.constraints['max_aoa_deg'] < 0
    assert sim.prop_system.time == 0
    assert thermal.temperature == 300. and thermal.commits == []
    assert sim.result.history == [] and sim.result.burn_duration == 0
