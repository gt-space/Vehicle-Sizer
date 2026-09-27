import unittest
from copy import deepcopy

import numpy as np

from Flight.Flight import FlightSim
from Flight.flight_forces import gravity
from simulation_types import (
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
    reference_area = 1.0

    @staticmethod
    def aoa(time):
        return 0.1 + 0.01 * time

    @staticmethod
    def evaluate(kinematics, atmosphere, engine_on):
        return AeroOut(Cd=0.0, D=0.0)

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


class ShutdownPropSystem(FakePropSystem):
    def update(self, dt, atm, heat_rate, commit=True, axial_specific_force=0.0):
        output = super().update(dt, atm, heat_rate, commit, axial_specific_force)
        output.mdot = {"OX_INJ": 1.0, "FUEL_INJ": 0.5, "NOZZLE": 1.5}
        if dt is not None:
            output.propulsion.mode = "shutdown"
            output.propulsion.shutdown_reason = "oxidizer_unavailable"
            output.events = ({"event": "dryout"},)
        return output


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

    def test_progress_reports_initial_and_completed_states(self):
        reported = []
        history = self.flight.run(progress=reported.append)
        self.assertEqual(reported[0].t, 0.0)
        self.assertEqual([kin.t for kin in reported[1:]],
                         [state["kinematics"].t for state in history])
        self.assertEqual(len(reported), len(history) + 1)

    def test_run_uses_prop_system_and_dynamic_node_mass(self):
        history = self.flight.run()

        self.assertEqual(len(history), 1)
        self.assertEqual(self.prop_system.calls[0]["dt"], None)
        self.assertTrue(self.prop_system.calls[0]["commit"])
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
            heat_rate={},
            commit=False,
        ).propulsion
        self.assertTrue(self.flight.engine_on(propulsion))

        propulsion.mode = "shutdown"
        propulsion.thrust = 50.0

        self.assertFalse(self.flight.engine_on(propulsion))

    def test_disabling_post_shutdown_solve_freezes_zero_flow_state(self):
        propulsion = ShutdownPropSystem()
        flight = FlightSim(
            cfg={
                "simulation": {
                    "dt": 0.1,
                    "t_end": 0.3,
                    "fluid_solve_post_shutdown": False,
                },
                "launch": {"altitude": 0.0, "rail_height": 5.0},
            },
            env=FakeEnvironment(),
            aero=FakeAero(),
            prop_system=propulsion,
            vehicle=FakeVehicle(),
        )

        history = flight.run(v0=100.0)

        # Rail exit inserts an accepted event boundary before the usual samples.
        self.assertEqual(len(propulsion.calls), 2)
        self.assertEqual(len(history), 4)
        for state in history:
            fluids = state["plant"].fluids
            self.assertTrue(all(mdot == 0.0 for mdot in fluids.mdot.values()))
            self.assertEqual(fluids.propulsion.thrust, 0.0)
        self.assertEqual(history[0]["plant"].fluids.events, ({"event": "dryout"},))
        self.assertEqual(history[1]["plant"].fluids.events, ())

    def test_history_records_end_forces_and_mass_distribution(self):
        result = self.flight.run()[0]
        kin = result["kinematics"]
        mass = result["mass_properties"]
        forces = result["forces"]

        self.assertAlmostEqual(mass["total_mass"], kin.m)
        self.assertAlmostEqual(mass["cg"], self.flight.vehicle.cg)
        np.testing.assert_array_equal(mass["station"], self.flight.vehicle.station)
        np.testing.assert_array_equal(mass["axial_mass"], self.flight.vehicle.mass)
        self.assertEqual(mass["length"], self.flight.vehicle.length)
        self.assertEqual(set(result["loads"]), {"station", "axial", "axial_aero", "normal", "shear", "bending"})
        self.assertAlmostEqual(forces["gravity"], gravity(kin.m, kin.h))
        self.assertAlmostEqual(forces["acceleration"], forces["net"] / kin.m)

    def test_trial_fluid_passes_node_heat_rate(self):
        atmosphere = FakeEnvironment.atmosphere(0.0, 0.0)
        thermal = ThermalOut(
            node={
                "tank": {
                    "cells": {"wall_T": np.array([300.0])},
                    "phases": {"gas": {"heat_rate": 25.0}},
                }
            }
        )

        self.flight.trial_fluid(
            0.1,
            atmosphere,
            thermal_out=thermal,
            axial_specific_force=12.0,
        )

        self.assertEqual(
            self.prop_system.calls[-1]["heat_rate"], {"tank": {"gas": 25.0}}
        )
        self.assertFalse(self.prop_system.calls[-1]["commit"])
        self.assertEqual(self.prop_system.calls[-1]["axial_specific_force"], 12.0)


if __name__ == "__main__":
    unittest.main()
