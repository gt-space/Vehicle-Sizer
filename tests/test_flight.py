import unittest
from copy import deepcopy
from dataclasses import replace
import pytest

import numpy as np

from Flight.Flight import FlightSim
from Flight.flight_forces import gravity
from simulation_types import (
    AeroOut,
    AtmosState,
    FluidOut,
    KinematicsState,
    PlantOut,
    PropulsionOut,
    ThermalOut,
)


class FakeEnvironment:
    @staticmethod
    def atmosphere(altitude, velocity):
        return AtmosState(T=288.0, p=101325.0, rho=1.2, mu=1.8e-5,
                          a=340.0, q=altitude, Ma=abs(velocity) / 340.0)


class FakeAero:
    reference_area = 1.0

    @staticmethod
    def evaluate(atmosphere, alpha, engine_on):
        return AeroOut(Cd=0.0, D=0.0, cp=1.5)

    @staticmethod
    def normal_distribution(mach, alpha):
        del mach, alpha
        return {"x": np.array([0.0, 2.0]), "dcn_dx": np.zeros(2)}

    @staticmethod
    def axial_distribution(mach, alpha, engine_on):
        return dict(x=np.array([0., 2.]), dca_dx=np.zeros(2),
                    parts={"base": np.zeros(2)}, point_loads={"base": (2., 0.)}, ca=0.)


class FakePropSystem:
    def __init__(self):
        self.calls = []
        self.time = 0.
        self.last = None

    def checkpoint(self):
        return deepcopy(self.__dict__)

    def restore(self, checkpoint):
        self.__dict__ = deepcopy(checkpoint)

    def update(
        self, dt, atm, heat_rate, commit=True, axial_specific_force=0.0
    ):
        self.calls.append(
            {
                "dt": dt,
                "atm": atm,
                "heat_rate": heat_rate,
                "commit": commit,
                "axial_specific_force": axial_specific_force,
            }
        )
        liquid_mass = 5.0 if dt is None else 4.9
        out = FluidOut(
            node={
                "tank": {
                    "mass": liquid_mass,
                    "tank_id": "tank",
                    "axial_mass": [liquid_mass],
                },
                "copv": {
                    "mass": 1.0,
                    "tank_id": "copv",
                    "axial_mass": [1.0],
                },
                "junction": {"P": 2.0e6},
            },
            branch={},
            mdot={},
            propulsion=PropulsionOut(
                mode="combusting",
                shutdown_reason=None,
                thrust=300.0,
                Pc=2.0e6,
                MR=2.0,
                Cf=1.5,
                cstar=1500.0,
                mdot_ox=1.0,
                mdot_fuel=0.5,
                mdot_nozzle=1.5,
            ),
        )
        start = self.time
        end = start + (dt or 0.)
        out.samples = ((start, replace(self.last or out, samples=(), events=())),
                       (end, replace(out, samples=())))
        if commit:
            self.time, self.last = end, replace(out, samples=())
        return out


class ShutdownPropSystem(FakePropSystem):
    def update(self, dt, atm, heat_rate, commit=True, axial_specific_force=0.):
        out = super().update(dt, atm, heat_rate, commit, axial_specific_force)
        if dt is not None:
            before = replace(out, samples=())
            out.propulsion = replace(out.propulsion, mode='shutdown', thrust=0.,
                                     shutdown_reason='oxidizer_unavailable')
            out.events = ({'event': 'shutdown', 'time_s': self.time, 'before': before},)
            out.samples = out.samples[:1] + ((self.time, before), (self.time, replace(out, samples=())))
        return out


class FakeVehicle:
    def __init__(self):
        self.total_mass = 10.0
        self.Ixx = 2.0
        self.Iyy = 3.0
        self.cg = 1.0
        self.length = 2.0
        self.station = np.array([0.0, 1.0])
        self.cell_edges = np.array([0.0, 1.0, 2.0])
        self.engine_start_station = 1.0
        self.mass = np.array([5.0, 5.0])
        self.EI = np.ones(2)

    def update_mass_distribution(self, node_states):
        self.total_mass = 10.0 + sum(
            sum(state["axial_mass"])
            for state in node_states.values()
            if "axial_mass" in state
        )
        self.mass = np.array([5.0, self.total_mass - 5.0])
        self.cg = float(np.sum(self.mass * self.station) / self.total_mass)


def flight():
    return FlightSim({'simulation': {'dt': .1, 't_end': .3},
                      'launch': {'altitude': 0., 'rail_height': 5.}},
                     FakeEnvironment(), FakeAero(), FakePropSystem(), FakeVehicle())


def test_dynamic_mass_and_synchronized_reports():
    sim = flight()
    reported = []
    history = sim.run(progress=reported.append)
    assert [k.t for k in reported] == pytest.approx([0., .1, .2, .3])
    assert len(sim.prop_system.calls) == 4
    for row in history:
        kin, props = row['kinematics'], row['mass_properties']
        assert set(vars(kin)) == {'t', 'x', 'h', 'vx', 'vz', 'theta', 'q'}
        assert props['total_mass'] == pytest.approx(15.9)
        assert row['atmosphere'].q == kin.h
        assert row['alpha'] == 0
        assert row['plant'].fluids.samples == ()
        assert row['forces']['gravity'] == gravity(props['total_mass'], kin.h)
        assert row['forces']['az'] == row['forces']['net'] / props['total_mass']
        assert set(row['loads']) == {'station', 'axial', 'axial_aero', 'normal', 'shear', 'bending'}


def test_mass_interpolation_integrates_variable_acceleration():
    sim = flight()
    sim.cfg['simulation']['t_end'] = .1
    history = sim.run(compute_loads=False)
    # Mass falls linearly 16 -> 15.9 kg; analytic thrust impulse / mass.
    expected_v = 300 * np.log(16 / 15.9) - gravity(1., 0.) * .1
    assert history[-1]['kinematics'].vz == pytest.approx(expected_v, abs=1e-7)


def test_rail_constraint_and_air_relative_angle():
    sim = flight()
    kin = KinematicsState(1., 2., 1., 3., 4., .4, .5)
    constrained = sim.constrain_to_rail(kin)
    assert (constrained.x, constrained.vx, constrained.theta, constrained.q) == (0., 0., np.pi/2, 0.)
    assert (constrained.h, constrained.vz) == (kin.h, kin.vz)
    gamma = np.deg2rad(179.)
    kin = KinematicsState(0., 0., 0., 100*np.cos(gamma), 100*np.sin(gamma), np.deg2rad(-179), 0.)
    assert sim.flight_kinematics(kin)[4] == pytest.approx(np.deg2rad(2.))


def test_force_rotation_and_restoring_moment():
    sim = flight()
    kin = KinematicsState(0., 0., 0., 1., 0., 0., 0.)
    fluid = sim.prop_system.update(None, FakeEnvironment.atmosphere(0., 0.), {})
    prop = replace(fluid.propulsion, thrust=100.)
    aero = AeroOut(Cd=.1, D=12., Ca=.1, A=10., Cn=.2, N=20., cp=2.)
    force = sim.forces(kin, aero, prop, dict(total_mass=10., cg=1., Iyy=5.))
    assert force['Fx'] == 90.
    assert force['Fz'] == 20. - gravity(10., 0.)
    assert force['pitch_moment'] == -20.
    assert force['pitch_acceleration'] == -4.


def test_fluid_update_passes_heat_and_preview_policy():
    sim = flight()
    atm = FakeEnvironment.atmosphere(0., 0.)
    thermal = ThermalOut({'tank': {'phases': {'gas': {'heat_rate': 25.}}}})
    for commit in (False, True):
        sim.update_fluid(.1, atm, thermal, axial_specific_force=12., commit=commit)
        assert sim.prop_system.calls[-1]['heat_rate'] == {'tank': {'gas': 25.}}
        assert sim.prop_system.calls[-1]['commit'] is commit
        assert sim.prop_system.calls[-1]['axial_specific_force'] == 12.


def test_engine_aero_mode_uses_combustion_not_residual_thrust():
    sim = flight()
    prop = sim.prop_system.update(None, FakeEnvironment.atmosphere(0., 0.), {}).propulsion
    assert sim.engine_on(prop)
    assert not sim.engine_on(replace(prop, mode='shutdown', thrust=50.))
