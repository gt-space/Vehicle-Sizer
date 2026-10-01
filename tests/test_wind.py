import math
from dataclasses import replace

import pytest

from Flight.environment import Environment, WindProfile
from Flight.Flight import FlightSim
from simulation_types import AeroOut, KinematicsState
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


def test_profile_interpolates_components_and_is_zero_outside(tmp_path):
    path = tmp_path / "wind.csv"
    path.write_text("altitude_m,wind_x_m_s,wind_z_m_s\n0,2,-1\n100,10,3\n")
    profile = WindProfile(path)
    assert profile.wind(50) == (6, 1)
    assert profile.wind(0) == (2, -1)
    assert profile.wind(100) == (10, 3)
    assert profile.wind(-1) == profile.wind(101) == (0, 0)


@pytest.mark.parametrize("data", [
    "height,vx,vz\n0,0,0\n100,0,0\n",
    "altitude_m,wind_x_m_s,wind_z_m_s\n0,0,0\n",
    "altitude_m,wind_x_m_s,wind_z_m_s\n0,0,0\n0,1,0\n",
    "altitude_m,wind_x_m_s,wind_z_m_s\n100,0,0\n0,1,0\n",
    "altitude_m,wind_x_m_s,wind_z_m_s\n0,nan,0\n100,1,0\n",
    "altitude_m,wind_x_m_s,wind_z_m_s\n0,no,0\n100,1,0\n",
])
def test_profile_rejects_invalid_data(tmp_path, data):
    path = tmp_path / "wind.csv"
    path.write_text(data)
    with pytest.raises(ValueError):
        WindProfile(path)


class WindEnvironment(FakeEnvironment):
    def __init__(self, wind_x=0, wind_z=0):
        self.components = (wind_x, wind_z)

    def wind(self, altitude):
        return self.components

    def atmosphere(self, altitude, velocity):
        atm = super().atmosphere(altitude, velocity)
        return replace(atm, q=0.5 * atm.rho * velocity**2)


def flight(env, aero=None):
    return FlightSim(
        {"simulation": {"dt": 0.1, "t_end": 0.3},
         "launch": {"altitude": 0, "rail_height": 0}},
        env, aero or FakeAero(), FakePropSystem(), FakeVehicle(),
    )


def test_air_relative_mach_pressure_and_alpha():
    sim = flight(WindEnvironment(5, 3))
    kin = KinematicsState(0, .1, 0, 50, 0, 100, math.pi / 2, 0, 0, 10, 3)
    kin = sim._kinematic_boundary(kin, False)
    atm = sim._atmosphere(kin)
    assert kin.alpha == pytest.approx(math.pi / 2 - math.atan2(97, -5))
    assert atm.Ma == pytest.approx(math.hypot(-5, 97) / 340)
    assert atm.q == pytest.approx(.5 * 1.2 * (5**2 + 97**2))
    # The branch retains a constrained zero-AoA rail model.
    assert sim._kinematic_boundary(kin, True).alpha == 0


class NormalAero(FakeAero):
    @staticmethod
    def evaluate(kinematics, atmosphere, engine_on):
        return AeroOut(Cd=0, D=0, N=atmosphere.q * kinematics.alpha * .01, cp=1.5)


def test_wind_changes_trajectory_and_history():
    calm = flight(WindEnvironment(), NormalAero()).run(v0=100, compute_loads=False)
    windy = flight(WindEnvironment(5), NormalAero()).run(v0=100, compute_loads=False)
    assert windy[-1]["kinematics"].x > calm[-1]["kinematics"].x + .001
    for state in windy:
        kin, wind = state["kinematics"], state["wind"]
        assert wind["wind_x"] == 5
        assert wind["airspeed"] == pytest.approx(math.hypot(kin.vx - 5, kin.vz))
        assert kin.alpha == pytest.approx(kin.theta - wind["gamma_air"])
        assert state["atmosphere"].Ma == pytest.approx(wind["airspeed"] / 340)


def test_zero_wind_matches_atmosphere_only_provider():
    old = flight(FakeEnvironment()).run(v0=100)
    zero = flight(FakeEnvironment())
    zero.env.wind = lambda h: (0, 0)
    new = zero.run(v0=100)
    assert [s["kinematics"] for s in old] == [s["kinematics"] for s in new]


def test_environment_defaults_to_no_wind():
    env = Environment(h_max=100, dh=100)
    assert env.wind(50) == (0, 0)
