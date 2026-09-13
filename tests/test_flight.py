import unittest

import numpy as np

from Flight.Flight import FlightSim
from Flight.flight_forces import gravity
from Flight.types import (
    AeroOut,
    AtmosState,
    FluidOut,
    KinematicsState,
    PropulsionOut,
    ThermalOut,
)


class FakeEnvironment:
    @staticmethod
    def atmosphere(altitude, velocity):
        return AtmosState(T=288.0, p=101325.0, rho=1.2, mu=1.8e-5,
                          a=340.0, q=altitude, Ma=abs(velocity) / 340.0)


class FakeAero:
    @staticmethod
    def aoa(time):
        return 0.1 + 0.01 * time

    @staticmethod
    def evaluate(kinematics, atmosphere, engine_on):
        return AeroOut(Cd=0.0, D=0.0, heat_bc={})


class FakePropSystem:
    def __init__(self):
        self.calls = []

    def update(
        self, dt, atm, heat_flux, commit=True, axial_specific_force=0.0
    ):
        self.calls.append(
            {
                "dt": dt,
                "atm": atm,
                "heat_flux": heat_flux,
                "commit": commit,
                "axial_specific_force": axial_specific_force,
            }
        )
        liquid_mass = 5.0 if dt is None else 4.9
        return FluidOut(
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
            td_state={"tank": {}, "copv": {}},
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


class FakeVehicle:
    def __init__(self):
        self.total_mass = 10.0
        self.Ixx = 2.0
        self.Iyy = 3.0
        self.cg = 1.0
        self.station = np.array([0.0, 1.0])
        self.mass = np.array([5.0, 5.0])

    def update_mass_distribution(self, node_states):
        self.total_mass = 10.0 + sum(
            sum(state["axial_mass"])
            for state in node_states.values()
            if "axial_mass" in state
        )
        self.mass = np.array([5.0, self.total_mass - 5.0])
        self.cg = float(np.sum(self.mass * self.station) / self.total_mass)


class FlightSkeletonTests(unittest.TestCase):
    def setUp(self):
        self.prop_system = FakePropSystem()
        self.flight = FlightSim(
            cfg={
                "simulation": {"dt": 0.1, "t_end": 0.1},
                "launch": {"altitude": 0.0, "rail_height": 5.0},
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
        self.assertAlmostEqual(
            self.prop_system.calls[0]["axial_specific_force"],
            gravity(10.0, 0.0) / 10.0,
        )
        self.assertAlmostEqual(
            self.prop_system.calls[1]["axial_specific_force"], 300.0 / 16.0
        )
        self.assertAlmostEqual(history[0]["kinematics"].m, 15.9)
        start_acceleration = (300.0 - gravity(16.0, 0.0)) / 16.0
        end_acceleration = history[0]["forces"]["acceleration"]
        self.assertAlmostEqual(
            history[0]["kinematics"].v,
            0.5 * (start_acceleration + end_acceleration) * 0.1,
        )
        self.assertAlmostEqual(
            history[0]["kinematics"].h,
            0.5 * history[0]["kinematics"].v * 0.1,
        )

    def test_history_is_synchronized_at_end_of_step(self):
        history = self.flight.run()
        result = history[0]
        kin = result["kinematics"]

        self.assertAlmostEqual(result["atmosphere"].q, kin.h)
        self.assertEqual(len(self.prop_system.calls), 2)
        self.assertAlmostEqual(
            result["plant"].fluids.node["tank"]["mass"],
            4.9,
        )
        self.assertTrue(result["on_rail"])
        self.assertEqual(kin.alpha, 0.0)

    def test_scheduled_aoa_begins_at_rail_exit(self):
        launch_altitude = self.flight.cfg["launch"]["altitude"]
        rail_height = self.flight.cfg["launch"]["rail_height"]
        rail_end = launch_altitude + rail_height

        self.assertEqual(self.flight.angle_of_attack(5.0, rail_end - 0.01), 0.0)
        self.assertAlmostEqual(
            self.flight.angle_of_attack(5.0, rail_end),
            FakeAero.aoa(5.0),
        )

    def test_endpoint_uses_scheduled_aoa_after_crossing_rail_end(self):
        self.flight.cfg["launch"]["rail_height"] = 0.001

        result = self.flight.run()[0]

        self.assertFalse(result["on_rail"])
        self.assertAlmostEqual(
            result["kinematics"].alpha,
            FakeAero.aoa(result["kinematics"].t),
        )

    def test_engine_on_uses_combustion_mode_not_residual_thrust(self):
        propulsion = self.prop_system.update(
            dt=None,
            atm=FakeEnvironment.atmosphere(0.0, 0.0),
            heat_flux={},
            commit=False,
        ).propulsion
        self.assertTrue(self.flight.engine_on(propulsion))

        propulsion.mode = "shutdown"
        propulsion.thrust = 50.0

        self.assertFalse(self.flight.engine_on(propulsion))

    def test_history_records_end_forces_and_mass_distribution(self):
        result = self.flight.run()[0]
        kin = result["kinematics"]
        mass = result["mass_properties"]
        forces = result["forces"]

        self.assertAlmostEqual(mass["total_mass"], kin.m)
        self.assertAlmostEqual(mass["cg"], self.flight.vehicle.cg)
        np.testing.assert_array_equal(mass["station"], self.flight.vehicle.station)
        np.testing.assert_array_equal(mass["axial_mass"], self.flight.vehicle.mass)
        self.assertAlmostEqual(forces["gravity"], gravity(kin.m, kin.h))
        self.assertAlmostEqual(forces["acceleration"], forces["net"] / kin.m)

    def test_step_fluids_passes_node_heat_flux(self):
        atmosphere = FakeEnvironment.atmosphere(0.0, 0.0)
        thermal = ThermalOut(
            wall_T=300.0,
            heat_flux_to_fluids={"tank": 25.0},
        )

        self.flight.step_fluids(
            0.1,
            atmosphere,
            thermal_out=thermal,
            commit=False,
            axial_specific_force=12.0,
        )

        self.assertEqual(self.prop_system.calls[-1]["heat_flux"], {"tank": 25.0})
        self.assertFalse(self.prop_system.calls[-1]["commit"])
        self.assertEqual(self.prop_system.calls[-1]["axial_specific_force"], 12.0)


if __name__ == "__main__":
    unittest.main()
