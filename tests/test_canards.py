"""Canard placement, force/moment closure and accepted-step roll integration."""
from dataclasses import replace
from types import SimpleNamespace
from importlib import import_module

import numpy as np
import pytest

from Flight.canards import CanardAerodynamics, CanardSystem, CanardForces, resolve_station
from Flight.flight_forces import Aero
from Flight.Flight import FlightSim
from Flight.loads import Loads
from Flight.roll import RollAnalysis
from simulation_types import AeroOut, AtmosState, KinematicsState
from test_aero import CANDIDATE, FakeDragModel
from test_flight import FakeAero, FakeEnvironment
from test_flight_events import ConstantVehicle, TimedPropulsion
from test_loads import FakeVehicle as LoadVehicle, FakeAero as LoadAero


def test_shock_table_matches_reference_between_nodes():
    aero = CanardAerodynamics({})
    rng = np.random.default_rng(42)
    mach = rng.uniform(2.8, 10., 2000)
    angle = rng.uniform(-np.deg2rad(25.), np.deg2rad(25.), 2000)
    np.testing.assert_allclose(aero.tabulated_shock_expansion_cn(mach, angle),
                               aero.shock_expansion_cn(mach, angle), rtol=2e-5, atol=1e-8)


def test_shock_table_boundaries_symmetry_and_broadcasting():
    aero = CanardAerodynamics({})
    mach = np.array([2.8, 4., 10.])[:, None]
    angle = np.deg2rad(np.array([-25., 0., 25.]))
    result = aero.tabulated_shock_expansion_cn(mach, angle)
    np.testing.assert_allclose(result, aero.shock_expansion_cn(*np.broadcast_arrays(mach, angle)),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(result[:, 0], -result[:, 2])
    np.testing.assert_array_equal(result[:, 1], 0.)
    assert aero.tabulated_shock_expansion_cn(4., 0.).shape == ()


def test_shock_table_preserves_solver_outside_table_and_invalid_domain():
    aero = CanardAerodynamics({})
    mach = np.array([11., 4., 3., 10.])
    angle = np.deg2rad([5., -26., 40., 30.])
    expected = aero.shock_expansion_cn(mach, angle)
    assert np.all(np.isfinite(expected[:2]))
    assert np.all(np.isnan(expected[2:]))
    np.testing.assert_allclose(aero.tabulated_shock_expansion_cn(mach, angle), expected,
                               rtol=0, atol=0, equal_nan=True)


def vehicle():
    return SimpleNamespace(length=5., sections=[
        SimpleNamespace(name='bay', start_station=2., length=.5),
        SimpleNamespace(name='aft', start_station=4., length=1.)])


@pytest.mark.parametrize('location,expected', [({'station_m': 0.}, 0.),
    ({'station_m': 5.}, 5.), ({'section': 'bay'}, 2.),
    ({'section': 'bay', 'offset_m': .2}, 2.2), ({'section': 'aft', 'offset_m': 1.}, 5.)])
def test_station_forms(location, expected):
    assert resolve_station(location, vehicle()) == pytest.approx(expected)


def test_relative_station_follows_section_geometry():
    v = vehicle()
    v.sections[0].start_station += .3
    assert resolve_station({'section': 'bay', 'offset_m': .2}, v) == pytest.approx(2.5)
    assert resolve_station({'station_m': 2.2}, v) == 2.2


@pytest.mark.parametrize('location', [{}, {'station_m': 1., 'section': 'bay'},
    {'station_m': 1., 'offset_m': 0.}, {'station_m': -1.}, {'station_m': 6.},
    {'station_m': float('nan')}, {'section': 'unknown'}, {'section': 'bay', 'offset_m': -.1},
    {'section': 'bay', 'offset_m': .6}, {'section': 'bay', 'offset_m': float('inf')}])
def test_invalid_station(location):
    with pytest.raises(ValueError):
        resolve_station(location, vehicle())


def system(**updates):
    cfg = dict(root_chord_m=.25, max_deflection_deg=10., station_m=2.,
               aero=dict(cd0=.2, oswald=.4, deflection_drag_multiplier=2.))
    cfg.update(updates)
    return CanardSystem(cfg, .3, vehicle())


@pytest.mark.parametrize('mach', [.5, 1.2, 2.8, 2.81, 4., 7., 10.])
@pytest.mark.parametrize('alpha_deg', [-14., 0., 8., 14.])
def test_bracketed_command_harness(mach, alpha_deg):
    from examples.benchmark_canards import bracketed_command_deflection
    canards = system()
    alpha = np.deg2rad(alpha_deg)
    limit = min(np.deg2rad(10.), max(np.deg2rad(22. if mach <= 2.8 else 15.) - abs(alpha), 0.))
    angles = np.linspace(0., limit, 240)
    curve = canards.aero.roll_coefficient(canards.geometry, mach, angles, alpha)
    assert np.all(np.diff(curve) >= -1e-12)
    scale = 100.
    for fraction in [0., .01, .5, .99, 1., 2.]:
        demand = fraction * scale * curve[-1]
        original = canards.command_deflection(mach, alpha, demand, scale, True)
        trial = bracketed_command_deflection(canards, mach, alpha, demand, scale, True)
        assert trial[0] == pytest.approx(original[0], abs=2e-5)
        assert trial[1:] == original[1:]
        if 0 < fraction < 1:
            achieved = scale * float(canards.aero.roll_coefficient(
                canards.geometry, mach, trial[0], alpha))
            assert achieved == pytest.approx(demand, abs=1e-7)
    for active, scale in [(False, 100.), (True, 0.)]:
        assert bracketed_command_deflection(canards, mach, alpha, 1., scale, active) == (
            canards.command_deflection(mach, alpha, 1., scale, active))


@pytest.mark.parametrize('alpha', [-.1, 0., .1])
@pytest.mark.parametrize('mach', [.5, 1., 2., 4.])
def test_canard_force_transform_and_com_moment(alpha, mach):
    canards = system()
    atm = AtmosState(288., 1e5, 1.2, 1.8e-5, 340., 20000., mach)
    out = canards.evaluate(atm, alpha, 1., True, 3.)
    lift = (2 * atm.q * canards.geometry.area * canards.geometry.body_factor
            * float(canards.aero.canard_lift(canards.geometry, mach, alpha)))
    assert out.axial_force_n * np.cos(alpha) + out.normal_force_n * np.sin(alpha) == pytest.approx(out.drag_n)
    assert -out.axial_force_n * np.sin(alpha) + out.normal_force_n * np.cos(alpha) == pytest.approx(lift)
    assert out.pitch_moment_nm == pytest.approx(out.normal_force_n)
    shifted = canards.evaluate(atm, alpha, 1., True, 3.5)
    assert shifted.pitch_moment_nm == pytest.approx(out.pitch_moment_nm + .5*out.normal_force_n)
    assert out.control_roll_moment_nm <= 0 and np.isfinite(out.axial_force_n)


def test_command_reversal_authority_and_passive_canards():
    canards = system()
    atm = AtmosState(288., 1e5, 1.2, 1.8e-5, 340., 20000., 2.)
    positive, negative = [canards.evaluate(atm, .03, sign*1., True, 3.) for sign in (1, -1)]
    assert positive.deflection_rad == -negative.deflection_rad
    assert positive.control_roll_moment_nm == -negative.control_roll_moment_nm
    assert positive.drag_n == negative.drag_n
    passive = canards.evaluate(atm, .03, 1., False, 3.)
    assert passive.deflection_rad == passive.control_roll_moment_nm == 0
    assert passive.normal_force_n != 0 and passive.drag_n > 0
    assert canards.evaluate(atm, .03, 1e6, True, 3.).saturated


def test_combined_cp_and_flight_moment_agree_without_double_counting():
    from test_flight import flight
    sim = flight()
    aero = AeroOut(Cd=.1, D=12., A=10., N=1000., cp=4.,
        canard=CanardForces(station_m=2., drag_n=3., axial_force_n=2.,
                           normal_force_n=100., pitch_moment_nm=100.))
    kin = KinematicsState(0., 0., 0., 1., 0., 0., 0.)
    prop = sim.prop_system.update(None, sim._atmosphere(kin), {}).propulsion
    force = sim.forces(kin, aero, replace(prop, thrust=0.), dict(total_mass=10., cg=3., Iyy=5.))
    assert aero.total_cp == pytest.approx(4200 / 1100)
    assert force['normal_aero'] == 1100.
    assert force['axial_aero'] == 12.
    assert force['pitch_moment'] == pytest.approx(-1100 * (aero.total_cp - 3.))
    assert force['pitch_moment'] == -900.
    aero.canard.normal_force_n = -1000.
    assert np.isnan(aero.total_cp)
    aero.N = aero.canard.normal_force_n = 0.
    assert np.isnan(aero.total_cp)


@pytest.mark.parametrize('station', [0., .2, .5, 1.25, 2.8, 3.])
def test_generic_point_mapping_preserves_force_and_moment(station):
    loads = Loads(LoadVehicle(), LoadAero())
    force = np.zeros(3)
    loads.add_point_load(force, station, 100.)
    assert force.sum() == pytest.approx(100.)
    assert force @ loads.vehicle.station == pytest.approx(100 * station)
    point = ((station, 50., 100.),)
    result = loads.evaluate(0., 0., 0., 50., 0., False, aerodynamic=False, point_loads=point)
    assert result['axial_aero'].sum() == pytest.approx(50.)
    assert result['normal'].sum() == pytest.approx(0., abs=1e-12)
    assert result['normal'] @ (loads.vehicle.station - loads.vehicle.cg) == pytest.approx(0., abs=1e-12)
    assert result['axial'][-1] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize('alpha', [-.09, 0., .09])
def test_aero_derivatives_match_actual_interpolated_moment(alpha):
    aero = Aero({}, CANDIDATE, FakeDragModel())
    cn_alpha, cm_alpha = aero.stability_derivatives(.6, alpha, 2.)
    def values(angle):
        _, _, cn, cp = aero.coefficients(.6, angle, True)
        return np.array([cn, -cn * (cp - 2.) / (CANDIDATE['omld'] * .0254)])
    step = 1e-7
    expected = (values(alpha + step) - values(alpha - step)) / (2 * step)
    np.testing.assert_allclose([cn_alpha, cm_alpha], expected, rtol=1e-5)


class RollAero(FakeAero):
    candidate = CANDIDATE

    @staticmethod
    def fin_normal_force_derivative(mach):
        return 2.

    @staticmethod
    def stability_derivatives(mach, alpha, cg):
        return 2., -1.


def controlled_flight(monkeypatch, cutoff=float('inf')):
    monkeypatch.setattr(import_module('Flight.Flight'), 'gravity', lambda mass, height: mass)
    cfg = dict(simulation=dict(dt=1., t_end=1., fluid_solve_post_shutdown=True),
        launch=dict(altitude=0., rail_height=0.), vehicle=dict(OMLD=.3),
        roll=dict(fin_cant=0., Cl_delta=0., Cl_p=0., M_roll_external=1.),
        canards=dict(root_chord_m=.25, station_m=1., max_deflection_deg=10., control_start_time_s=.35))
    sim = FlightSim(cfg, FakeEnvironment(), RollAero(), TimedPropulsion(cutoff), ConstantVehicle())
    # Known forcing cancellation and axial drag give analytic event-side integrals.
    def evaluate(atm, alpha, forcing, active, cg):
        return CanardForces(station_m=1., axial_force_n=2. if active else 0.,
                            drag_n=2. if active else 0., control_roll_moment_nm=-1. if active else 0.)
    monkeypatch.setattr(sim.canards, 'evaluate', evaluate)
    return sim


def test_activation_is_one_sided_and_roll_independent_of_history(monkeypatch):
    results = []
    for history in (False, True):
        sim = controlled_flight(monkeypatch)
        sim.run(v0=10., compute_loads=False, record_history=history)
        result = sim.result
        assert result.final_velocity == pytest.approx(19. - 2/16 * .65, abs=1e-7)
        assert .35 in result.roll['time']
        assert result.roll['p'][-1] == pytest.approx(.35 / 2)
        assert len(sim.prop_system.calls) == 2
        results.append(result.roll)
    for key in results[0]:
        np.testing.assert_allclose(results[0][key], results[1][key], equal_nan=True)


def test_terminal_retry_does_not_commit_future_roll(monkeypatch):
    sim = controlled_flight(monkeypatch, cutoff=0.)
    sim.canards.control_start = 2.
    sim.run(v0=.5, compute_loads=False)
    assert sim.result.apogee_time == pytest.approx(.5, abs=1e-7)
    roll = sim.result.roll
    assert roll['time'][-1] == pytest.approx(.5, abs=1e-7)
    assert np.all(np.diff(roll['time']) > 0)
    assert roll['p'][-1] == pytest.approx(.25, abs=1e-7)


def test_rail_release_gates_control_after_scheduled_activation(monkeypatch):
    sim = controlled_flight(monkeypatch)
    sim.cfg['launch']['rail_height'] = 2.18  # v0=10, a=9: release at 0.2 s
    sim.canards.control_start = .1
    sim.run(v0=10., compute_loads=False)
    assert sim.result.rail_exit_time == pytest.approx(.2, abs=1e-7)
    assert sim.result.final_velocity == pytest.approx(19. - 2/16 * .8, abs=1e-7)
    np.testing.assert_allclose(sim.result.roll['p'], 0., atol=1e-12)
    for time, moment in zip(sim.result.roll['time'], sim.result.roll['canard_roll_moment']):
        assert moment == (0. if time < sim.result.rail_exit_time else -1.)


def test_rejected_rk_stage_does_not_append_roll_samples(monkeypatch):
    from diagnostics.errors import TrialDomainError
    sim = controlled_flight(monkeypatch)
    rhs, rejected = sim.rhs, []
    def reject_once(kin, *args):
        if not rejected and kin.t > .1:
            rejected.append(kin.t)
            raise TrialDomainError('test stage rejection')
        return rhs(kin, *args)
    monkeypatch.setattr(sim, 'rhs', reject_once)
    sim.run(v0=10., compute_loads=False)
    assert rejected
    assert np.all(np.diff(sim.result.roll['time']) > 0)
    assert sim.result.roll['p'][-1] == pytest.approx(.35 / 2)
    assert len(sim.prop_system.calls) == 2


def test_roll_inertia_interpolates_with_fluid_samples(monkeypatch):
    from Flight.integration import sample_interval
    sim = controlled_flight(monkeypatch)
    fluid = sim.prop_system.update(None, FakeEnvironment.atmosphere(0., 0.), {})
    a = dict(total_mass=10., cg=1., Iyy=3., Ixx=2.)
    b = dict(total_mass=10., cg=1., Iyy=3., Ixx=4.)
    assert sample_interval(.25, [(0., fluid, a), (1., fluid, b)])[1]['Ixx'] == 2.5


def test_roll_damping_uses_supplied_airspeed():
    roll = RollAnalysis(.3, dict(Cl_delta=0., Cl_p=-1.), RollAero())
    atm = FakeEnvironment.atmosphere(10., 100.)
    mass = dict(Ixx=2., Iyy=3.)
    roll.sample(0., atm, 100., mass)
    roll.result['p'][-1] = 1.
    roll.sample(0., atm, 50., mass)
    expected = -atm.q * roll.area * .3 * .3 / (2*50.)
    assert roll.result['damping_moment'] == pytest.approx([expected])


def test_named_sections_are_optional_and_unique():
    from Configs.loader import load_config
    from Vehicle.Vehicle import Vehicle
    v = Vehicle.__new__(Vehicle)
    v.cfg, v.tanks = load_config('Configs/flight_pressure_fed_regulator.yaml'), {}
    definitions = [dict(type='nosecone', name='nose'), dict(type='nosecone')]
    v._resolve_section_masses = lambda: definitions
    assert [s.name for s in v._build_sections()] == ['nose', None]
    definitions[1]['name'] = 'nose'
    with pytest.raises(ValueError, match='unique'):
        v._build_sections()


def test_history_uses_combined_cp_and_total_forces():
    from reporting.history import snapshot_row
    from test_history_capture import sample
    state = sample(1., 100., 1e6)
    state['plant'].aero = AeroOut(Cd=.1, D=10., A=10., N=1000., cp=4.,
        canard=CanardForces(station_m=2., normal_force_n=100.))
    state['forces'].update(thrust_axial=100., axial_aero=20., normal_aero=1100., thrust_normal=2.)
    row = snapshot_row(state)
    assert row['derived.total_cp_m'] == pytest.approx(4200 / 1100)
    assert row['derived.static_stability_margin_calibers'] == pytest.approx((4200/1100 - 1.) / .3)
    assert row['derived.proper_acceleration_g'] == pytest.approx(np.hypot(80., 1102.) / 10 / 9.80665)
