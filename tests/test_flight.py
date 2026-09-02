import unittest

from Flight.Flight import FlightSim
from Flight.types import AeroOut, AtmosState, KinematicsState, ThermalOut


class FakeEnvironment:
    @staticmethod
    def atmosphere(altitude, velocity):
        return AtmosState(T=288.0, p=101325.0, rho=1.2, mu=1.8e-5,
                          a=340.0, q=0.0, Ma=0.0)


class FakeAero:
    @staticmethod
    def aoa(kinematics, atmosphere, on_rail):
        return 0.0

    @staticmethod
    def evaluate(kinematics, atmosphere):
        return AeroOut(D=0.0, heat_bc={})


class FakePropSystem:
    def __init__(self):
        self.calls = []

    def update(self, dt, atm, heat_flux, commit=True):
        self.calls.append(
            {"dt": dt, "atm": atm, "heat_flux": heat_flux, "commit": commit}
        )
        liquid_mass = 5.0 if dt is None else 4.9
        return {
            "node": {
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
            "branch": {},
            "td_state": {"tank": {}, "copv": {}},
            "mdot": {},
            "propulsion": {"thrust": 300.0},
        }


class FakeVehicle:
    def __init__(self):
        self.total_mass = 10.0
        self.Ixx = 2.0

    def update_mass_distribution(self, node_states):
        self.total_mass = 10.0 + sum(
            sum(state["axial_mass"])
            for state in node_states.values()
            if "axial_mass" in state
        )


class FlightSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.prop_system = FakePropSystem()
        self.flight = FlightSim(
            cfg={
                "simulation": {"dt": 0.1, "t_end": 0.1},
                "launch": {"altitude": 0.0, "rail_length": 5.0},
            },
            env=FakeEnvironment(),
            aero=FakeAero(),
            prop_system=self.prop_system,
            vehicle=FakeVehicle(),
        )

    def test_run_uses_prop_system_and_dynamic_node_mass(self):
        history = self.flight.run()

        self.assertEqual(len(history), 1)
        self.assertEqual(self.prop_system.calls[0]["dt"], None)
        self.assertFalse(self.prop_system.calls[0]["commit"])
        self.assertEqual(self.prop_system.calls[1]["dt"], 0.1)
        self.assertTrue(self.prop_system.calls[1]["commit"])
        self.assertAlmostEqual(history[0]["kinematics"].m, 15.9)
        self.assertGreater(history[0]["kinematics"].v, 0.0)

    def test_step_plant_passes_node_heat_flux(self):
        atmosphere = FakeEnvironment.atmosphere(0.0, 0.0)
        kinematics = KinematicsState(
            t=0.0, dt=0.1, h=0.0, v=0.0, w=0.0, alpha=0.0,
            m=16.0, Ixx=2.0,
        )
        thermal = ThermalOut(
            wall_T=300.0,
            heat_flux_to_fluids={"tank": 25.0},
        )

        self.flight.step_plant(
            kinematics,
            atmosphere,
            FakeAero.evaluate(kinematics, atmosphere),
            thermal_out=thermal,
            commit=False,
        )

        self.assertEqual(self.prop_system.calls[-1]["heat_flux"], {"tank": 25.0})
        self.assertFalse(self.prop_system.calls[-1]["commit"])


if __name__ == "__main__":
    unittest.main()
