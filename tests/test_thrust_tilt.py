import math
from dataclasses import replace

import numpy as np
import pytest

from Flight.Flight import FlightSim
from simulation_types import AeroOut, KinematicsState, PlantOut
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


def flight(tilt=None):
    cfg = {} if tilt is None else {"engine": {"thrust_tilt_deg": tilt}}
    vehicle = FakeVehicle()
    vehicle.engine_start_station = 1.75
    return FlightSim(cfg, FakeEnvironment(), FakeAero(), FakePropSystem(), vehicle)


@pytest.mark.parametrize("tilt", [None, 0., .4, -.4])
@pytest.mark.parametrize("theta", [0., math.pi / 2, 2.1])
def test_thrust_direction_moment_and_fluid_head(tilt, theta):
    sim = flight(tilt)
    kin = KinematicsState(1., .1, 0., 10., 0., 10., theta, 0., 0., 10., 3.)
    fluids = sim.prop_system.update(None, sim._atmosphere(kin), {})
    # Retain thrust torque even when the aero cutoff is active.
    aero = AeroOut(Cd=float("nan"), D=0., ballistic_coast=True)
    result = sim.forces(kin, PlantOut(aero, None, fluids), kin.m)
    thrust = fluids.propulsion.thrust
    delta = math.radians(tilt or 0.)
    assert result['Fx'] == pytest.approx(thrust * math.cos(theta + delta), abs=1e-12)
    assert result['Fz'] + result['gravity'] == pytest.approx(thrust * math.sin(theta + delta), abs=1e-12)
    assert result['pitch_moment'] == pytest.approx(-.75 * thrust * math.sin(delta))
    assert result['pitch_acceleration'] == pytest.approx(result['pitch_moment'] / kin.Iyy)
    assert result['axial_specific_force'] == pytest.approx(thrust * math.cos(delta) / kin.m)
    assert math.hypot(result['thrust_axial'], result['thrust_normal']) == pytest.approx(thrust)

    # Aerodynamic and engine moments add; shutdown removes only the engine term.
    aero = AeroOut(Cd=.2, D=2., A=2., N=4., cp=1.5)
    result = sim.forces(kin, PlantOut(aero, None, fluids), kin.m)
    assert result['pitch_moment'] == pytest.approx(result['thrust_moment'] - 2.)
    fluids.propulsion = replace(fluids.propulsion, thrust=0., mode='shutdown')
    result = sim.forces(kin, PlantOut(aero, None, fluids), kin.m)
    assert result['pitch_moment'] == -2.
    assert result['thrust_axial'] == result['thrust_normal'] == result['thrust_moment'] == 0.


@pytest.mark.parametrize('tilt', [float('nan'), float('inf')])
def test_invalid_tilt(tilt):
    with pytest.raises(ValueError, match='thrust_tilt_deg'):
        flight(tilt)


def test_tilted_engine_loads_remain_when_aero_is_off():
    from Flight.loads import Loads
    from test_loads import FakeVehicle as LoadVehicle

    vehicle = LoadVehicle()
    vehicle.engine_start_station = 2.25
    loads = Loads(vehicle, FakeAero())
    tilt, thrust = math.radians(.4), 10000.
    lateral = thrust * math.sin(tilt)
    result = loads.evaluate(0., 0., 0., 0., thrust * math.cos(tilt), True,
                            aerodynamic=False, thrust_normal=lateral)
    np.testing.assert_allclose(result['normal'], lateral * np.array([.125, -.25, .125]))
    assert result['normal'].sum() == pytest.approx(0., abs=1e-12)
    assert np.dot(result['normal'], vehicle.station - vehicle.cg) == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(result['axial'], loads.get_axial_load(np.zeros(3), thrust * math.cos(tilt)))
    assert np.max(np.abs(result['bending'])) > 0
    # With aero enabled, inertial relief is applied to both contributions.
    from test_loads import FakeAero as LoadAero
    loads.aero = LoadAero()
    combined = loads.get_normal_load(10., .5, .1, thrust_normal=lateral)
    np.testing.assert_allclose(combined, result['normal'] + loads.get_normal_load(10., .5, .1))
